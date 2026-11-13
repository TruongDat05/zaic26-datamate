"""Chạy lại đối chứng đột biến cho MỌI ứng viên của gói, trên snapshot upstream đã sửa lỗi (bằng chứng cho offline_candidates/1).

    DATAMATE_UPSTREAM=<snapshot> python eval/sqlv1/mutation_replay.py --snapshot <snapshot> [--procs 12]
      → outputs/synth_sql_v1/sft_v2/mutation_report.jsonl + mutation_summary.json

Mỗi shop: bản gốc dựng lại từ CSV của gói bằng snapshot.rebuild_shop (layer đã ghi: view/nợ tất định), 4 bản đột biến mức shop
(world.GENERIC) + 1 bản nhắm mọi đặc tả của shop (world.m_target), ghi file bằng cùng bộ ghi, nạp bằng loader của snapshot.
SQL chạy bằng guard của controller (harness_control/exec_guard: DB chỉ đọc, đúng một SELECT), kiểu hoá và chấm bằng scorer cố định
(chặt, không làm tròn hộ). Gold của mỗi bản đột biến tính lại bằng oracle độc lập (world.gold) trên bản ghi đã đột biến.
Đạt = bản gốc khớp gold (2 lần chạy giống nhau) + mọi bản đột biến tính được đều khớp gold mới + ≥ 2 bản được kiểm.
Một bản đột biến không đổi đáp án KHÔNG chứng minh SQL không ghi cứng: báo riêng n_changed / n_changed_ok (đột biến đổi
đáp án mà SQL vẫn đúng).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

FOXEL = controller_root()


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True, help="thư mục code-only của revision upstream (build/sft_v2/snapshot_<commit>)")
    ap.add_argument("--pkg", type=Path, default=FOXEL / "build" / "synthetic_sql_v1")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--shops", nargs="*")
    ap.add_argument("--subset", choices=("thinking", "retemplated"), help="only thinking.jsonl / only retemplated templates -> mutation_report.<subset>.jsonl")
    ap.add_argument("--merge", choices=("thinking", "retemplated"), help="merge mutation_report.<subset>.jsonl into mutation_report.jsonl (same snapshot), no replay")
    ap.add_argument("--tmp", type=Path, default=Path("/dev/shm/sftv2_mut"))
    return ap.parse_args()


A = _args() if __name__ == "__main__" else None
if A is not None:
    os.environ["DATAMATE_UPSTREAM"] = str(A.snapshot.resolve())
from sqlv1.up import debts, ingest, views, UPSTREAM  # noqa: E402,F401
from sqlv1 import package as PK, render as RD, shops as SH, world as WB  # noqa: E402

sys.path.append(str(FOXEL))  # harness_control (controller): SAU snapshot, để datamate vẫn là của snapshot
sys.path.append(str(FOXEL / "training"))
import duckdb  # noqa: E402
from harness_control import evaluate as E, exec_guard, scorer  # noqa: E402
import sft_format as F  # noqa: E402
from datamate.snapshot import rebuild_shop  # noqa: E402

WORK = SH.SYNTH_ROOT / "outputs" / "synth_sql_v1"
OUT = WORK / os.environ.get("SQLV1_OUT", "sft_v2")  # sft_q35: the Qwen3.5 build (new snapshot, thinking candidates)


def load_candidates(pkg: Path, names=("solutions.jsonl", "thinking.jsonl")) -> list[dict]:
    """candidates/solutions.jsonl (+ candidates/thinking.jsonl: thinking traces of eval/sqlv1/think.py, when present)."""
    out = []
    for name in names:
        p = pkg / "candidates" / name
        if p.exists():
            out += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


def candidate_id(c: dict) -> str:
    m = c["metadata"]
    return f"{c['id']}|{m['provider']}|{m['attempt']}|{m['skeleton_sha']}"


def run_twice(db, sql: str):
    """Kết quả guard (2 lần, phải giống nhau) hoặc lỗi."""
    out = []
    for _ in range(2):
        try:
            out.append(exec_guard.run({"sql": sql}, db, 20, 2000))
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
    return out[0] if out[0] == out[1] else {"status": "error", "error": "nondeterministic_result"}


def judge(db, kind: str, sql: str, gold: dict) -> tuple[str, str | None]:
    if kind not in scorer.TYPES:
        return "fail", f"unknown type {kind}"
    ex = run_twice(db, sql)
    if ex["status"] != "ok":
        return "error", ex["error"]
    try:
        ans = {"type": "unanswerable"} if kind == "unanswerable" else scorer.typed(kind, ex["columns"], ex["rows"])
    except ValueError as e:
        return "fail", f"shape: {e}"[:200]
    return ("ok", None) if scorer.verified_score(ans, gold) == 1 else ("fail", json.dumps(ans, ensure_ascii=False, default=str)[:160])


def build_db(root: Path, bid: str, layer: dict, path: Path):
    if path.exists():
        path.unlink()
    rebuild_shop(root, bid, layer, str(path))
    return exec_guard.open_db(duckdb, str(path))


def mutant_layer(root: Path, bid: str, written: dict) -> dict:
    """SELECT view/nợ tất định cho bản đột biến, từ cột mà loader của snapshot đọc được (như verify.mutant_db)."""
    db, meta = ingest.load_business(root, bid, canon=True)
    try:
        Nt = RD.resolve_tables(meta, written["files"])
        N = RD.relations(Nt)
        cols = {k: sorted({c for t in Nt[k] for c in SH._cols(db, t)}) for k in Nt}
        vs = RD.view_sql(written["roles"], N, cols)
        ds = RD.debt_sqls(written["roles"], N)
    finally:
        db.close()
    return {"canon_names": True, "views_sql": vs, "debt_sqls": list(ds) if ds else None}


def replay_shop(job) -> dict:
    bid, cands, qmeta, pkg, tmp, source_hash, loader_hash = job
    t0 = time.time()
    info, rec = SH.load_shop(WORK, bid)
    L = info["layers"]
    base_layer = {"canon_names": True, "views_sql": L["view_sql"] if L["view_ok"] else None, "debt_sqls": L["debt_sqls"] if L["debt_ok"] else None}
    gold = {g["id"]: g for g in map(json.loads, (pkg / "data_root" / "answers.jsonl").read_text(encoding="utf-8").splitlines())}
    work = tmp / bid
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    dbs, muts = {}, []
    try:
        dbs["base"] = build_db(pkg / "data_root", bid, base_layer, work / "base.duckdb")
    except Exception as e:  # noqa: BLE001
        dbs["base"] = e
    specs = [qmeta[c["id"]]["spec"] for c in cands]
    plan = info["plan"]
    for name, fn in list(WB.GENERIC) + [("m_target", lambda r: WB.m_target(r, specs))]:
        mrec = fn(rec)
        root = work / name
        folder = root / "data" / bid
        (root / "data").mkdir(parents=True)
        shutil.copy(WORK / "shops_root" / "calendar.csv", root / "calendar.csv")
        try:
            written = RD.write_shop(mrec, plan["family"], plan["style"], folder, SH.title_of(plan, mrec))
            dbs[name] = build_db(root, bid, mutant_layer(root, bid, written), work / f"{name}.duckdb")
        except Exception as e:  # noqa: BLE001
            dbs[name] = e
        muts.append((name, mrec))
    out = []
    for c in cands:
        q = qmeta[c["id"]]
        kind, sql = scorer.parse_completion(c["completion"])
        r = {"candidate_id": candidate_id(c), "question_id": c["id"], "business_id": bid, "completion_sha": F.sha(c["completion"]),
             "sql_sha": F.sha(sql), "source_hash": source_hash, "loader_closure_hash": loader_hash, "oracle_version": WB.ORACLE_VERSION,
             "scorer": scorer.VERSION, "exec_guard": exec_guard.VERSION, "mutants": {}}
        if isinstance(dbs["base"], Exception):
            r["base"] = {"status": "error", "detail": f"rebuild: {dbs['base']}"[:300]}
        elif not sql or not kind:
            r["base"] = {"status": "fail", "detail": "no TYPE/SQL"}
        else:
            st, det = judge(dbs["base"], kind, sql, gold[c["id"]])
            r["base"] = {"status": st, **({"detail": det} if det else {})}
        n_chk = n_chg = n_chg_ok = 0
        first = None if r["base"]["status"] == "ok" else "base"
        for name, mrec in muts:
            db = dbs.get(name)
            if isinstance(db, Exception) or db is None:
                r["mutants"][name] = {"status": "unavailable", "detail": str(db)[:200]}
                continue
            try:
                mg = WB.gold(mrec, q["spec"])
            except (WB.Degenerate, ValueError, KeyError, StopIteration, ZeroDivisionError):
                r["mutants"][name] = {"status": "not_checkable"}
                continue
            al = PK.alias_of(mrec, mg) if mg["type"] != "unanswerable" else None
            mgold = {**mg, **({"alias": al} if al else {})}
            changed = {k: v for k, v in mg.items() if k != "alias"} != {k: v for k, v in q["gold"].items() if k != "alias"}
            st, det = judge(db, kind, sql, mgold) if sql and kind else ("fail", "no TYPE/SQL")
            n_chk += 1
            n_chg += changed
            n_chg_ok += changed and st == "ok"
            r["mutants"][name] = {"status": st, "gold_changed": changed, **({"detail": det} if det else {})}
            if st != "ok" and first is None:
                first = name
        r.update(n_checked=n_chk, n_changed=n_chg, n_changed_ok=n_chg_ok, first_failure=first,
                 **{"pass": r["base"]["status"] == "ok" and first is None and n_chk >= 2})
        out.append(r)
    for db in dbs.values():
        if not isinstance(db, Exception):
            db.close()
    shutil.rmtree(work, ignore_errors=True)
    return {"bid": bid, "rows": out, "secs": round(time.time() - t0, 1),
            "mutants_unavailable": sorted(k for k, v in dbs.items() if isinstance(v, Exception))}


def merge_subset(subset):
    """mutation_report.<subset>.jsonl (same snapshot) merged into mutation_report.jsonl by candidate id, sorted, summary updated."""
    base = OUT / "mutation_report.jsonl"
    th = OUT / f"mutation_report.{subset}.jsonl"
    rows = {x["candidate_id"]: x for x in map(json.loads, base.read_text(encoding="utf-8").splitlines())}
    trows = [json.loads(x) for x in th.read_text(encoding="utf-8").splitlines()]
    src = {r["source_hash"] for r in list(rows.values())[:50] + trows}
    if len(src) != 1:
        raise SystemExit(f"reports from different snapshots: {src}")
    rows.update({r["candidate_id"]: r for r in trows})
    out = sorted(rows.values(), key=lambda x: x["candidate_id"])
    base.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    summ = json.loads((OUT / "mutation_summary.json").read_text(encoding="utf-8"))
    summ["candidates"] = len(out)
    summ["pass"] = sum(r["pass"] for r in out)
    summ[f"{subset}_merged"] = {"candidates": len(trows), "pass": sum(r["pass"] for r in trows),
                               "pass_with_changed_gold": sum(r["pass"] and r["n_changed_ok"] > 0 for r in trows)}
    summ["report_sha256"] = F.file_hash(base)
    (OUT / "mutation_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ[f"{subset}_merged"]), summ["candidates"], summ["pass"])


def main(a):
    if a.merge:
        return merge_subset(a.merge)
    snap = a.snapshot.resolve()
    assert Path(ingest.__file__).resolve().is_relative_to(snap), ingest.__file__
    source_hash = E.source_hash(snap)
    loader_hash = F.sha(E.import_closure(snap))
    cands = load_candidates(a.pkg, ("thinking.jsonl",) if a.subset == "thinking" else ("solutions.jsonl", "thinking.jsonl"))
    if a.subset == "retemplated":
        cands = [c for c in cands if c["metadata"].get("retemplated")]
    sel = {x["id"]: x for x in map(json.loads, (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines()) if x.get("id")}
    by = collections.defaultdict(list)
    for c in cands:
        by[c["business_id"]].append(c)
    shops = a.shops or sorted(by)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, logs = [], []
    jobs = [(b, by[b], {c["id"]: sel[c["id"]] for c in by[b]}, a.pkg, a.tmp, source_hash, loader_hash) for b in shops]
    with cf.ProcessPoolExecutor(a.procs) as ex:
        for i, r in enumerate(ex.map(replay_shop, jobs), 1):
            rows += r["rows"]
            logs.append({k: v for k, v in r.items() if k != "rows"})
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} shops, {len(rows)} candidates, pass {sum(x['pass'] for x in rows)}", flush=True)
    rows.sort(key=lambda x: x["candidate_id"])
    path = OUT / (f"mutation_report.{a.subset}.jsonl" if a.subset else "mutation_report.jsonl" if not a.shops else "mutation_report.partial.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = collections.Counter()
    for r in rows:
        st["pass" if r["pass"] else "fail:" + (r["first_failure"] or ("too_few_mutants" if r["n_checked"] < 2 else "?"))] += 1
    summ = {"snapshot": str(snap), "source_hash": source_hash, "loader_closure_hash": loader_hash, "oracle_version": WB.ORACLE_VERSION,
            "scorer": scorer.VERSION, "exec_guard": exec_guard.VERSION, "candidates": len(rows), "outcome": dict(st),
            "pass": sum(r["pass"] for r in rows), "base_ok": sum(r["base"]["status"] == "ok" for r in rows),
            "pass_with_changed_gold": sum(r["pass"] and r["n_changed_ok"] > 0 for r in rows),
            "n_checked_hist": dict(collections.Counter(r["n_checked"] for r in rows)),
            "mutant_status": {m: dict(collections.Counter(r["mutants"].get(m, {}).get("status") for r in rows)) for m in [n for n, _ in WB.GENERIC] + ["m_target"]},
            "shops": len(logs), "shops_with_unavailable_mutants": {x["bid"]: x["mutants_unavailable"] for x in logs if x["mutants_unavailable"]},
            "report_file": str(path), "report_sha256": F.file_hash(path)}
    (OUT / (f"mutation_summary.{a.subset}.json" if a.subset else "mutation_summary.json" if not a.shops else "mutation_summary.partial.json")).write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "shops_with_unavailable_mutants"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(A)
