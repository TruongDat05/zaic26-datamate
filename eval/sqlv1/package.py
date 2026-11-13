"""Chọn mẫu (bám MINING) và dựng gói nguồn trên nhánh upstream: build/synthetic_sql_v1 (train + dev) và build/synthetic_sql_v1_final.

    python eval/sqlv1/package.py [--target-rows 6600] [--out <controller>/build]

Gói nguồn KHÔNG có sft/ (exporter SFT v2 của pipeline tạo; chưa có trên upstream). Lời giải đã kiểm nằm ở candidates/solutions.jsonl
để exporter chạy lại. data_hash = null + pending_content_hash = true (content_hash() chưa có trên upstream).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

from sqlv1.up import UPSTREAM, upstream_commit  # noqa: E402
from sqlv1 import shops as SH, world as WB  # noqa: E402
from sqlv1.questions import WORK, fold  # noqa: E402
import synth_corpus as C  # noqa: E402
import synth_rubric as RB  # noqa: E402

ROOT = SH.SYNTH_ROOT
SKILL = dict(C.SKILL_FAMILY, revenue_payment="nhom_kenh_ca", n_orders_item="don", n_returned="don", top_item="so_luong_xep_hang",
             top_customer_orders="khach", rev_change="doanh_thu", share_channel="nhom_kenh_ca", shift_orders="nhom_kenh_ca",
             debt_older="cong_no", debt_new="cong_no")
WBINS = ((0, 8, "<=8"), (9, 12, "9-12"), (13, 16, "13-16"), (17, 20, "17-20"), (21, 999, ">20"))
PRIO = ("thang_cu_the", "quy", "nam", "ngay_cu_the", "den_het", "hom_nay", "hom_qua", "thang_nay", "thang_truoc", "tuan_nay", "tuan_truoc", "n_ngay", "cung_ky", "tet")
DIM_W = {"industry": 1.0, "skill": 1.5, "type": 1.5, "period": 1.5, "words": 0.7}
ROUTE_RANK = {"macro": 0, "view": 1, "raw": 2}
PROV_RANK = {"gptsol": 0, "deepseek": 1, "haiku": 2, "qwen": 3, "gemini": 4, "template": 9}


def wbin(q):
    n = len(q.split())
    return next(lab for lo, hi, lab in WBINS if lo <= n <= hi)


def ptag(q):
    t = set(RB.period_tags(q))
    return next((x for x in PRIO if x in t), "none")


def features(x):
    return {"industry": x["industry"] if x["industry"] in WB.CORE else "khac", "skill": SKILL.get(x["spec"]["slot"], "khac"), "type": x["gold"]["type"],
            "period": ptag(x["question"]), "words": wbin(x["question"])}


def mining_targets():
    rows = [json.loads(x) for x in (ROOT / "data/synth_source/seeds/questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    m = [r for r in rows if r["role"] == "mining" and r.get("skill_family")]
    out = {}
    for d in DIM_W:
        c = collections.Counter({"industry": lambda r: r["industry"] if r["industry"] in WB.CORE else "khac", "skill": lambda r: r["skill_family"],
                                 "type": lambda r: (r.get("gold") or {}).get("type"), "period": lambda r: ptag(r["question"]),
                                 "words": lambda r: wbin(r["question"])}[d](r) for r in m)
        out[d] = {k: v / len(m) for k, v in c.items()}
    return out


def l1(c: collections.Counter, n, tgt):
    keys = set(tgt) | set(c)
    return sum(abs(c.get(k, 0) / max(1, n) - tgt.get(k, 0)) for k in keys)


def load_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def best_solutions(sols: list[dict]) -> list[dict]:
    """≤ 2 lời giải đạt, khác khung; model trước SQL mẫu; macro > view > bảng thô; rồi provider, SQL ngắn."""
    from sqlv1 import verify as V
    ok = [s for s in sols if s.get("ok")]
    for s in ok:
        s.setdefault("route", V.route(s["sql"]))
        s.setdefault("skeleton", V.skeleton(s["sql"]))
    model = [s for s in ok if s["provider"] != "template"]
    pool = model or [s for s in ok if s["provider"] == "template"]
    pool.sort(key=lambda s: (ROUTE_RANK.get(s.get("route") or "raw", 2), PROV_RANK.get(s["provider"], 5), len(s["sql"])))
    top_route = pool[0].get("route") if pool else None
    out, seen = [], set()
    for s in pool:
        if s.get("route") != top_route:
            continue
        if s["skeleton"] in seen:
            continue
        seen.add(s["skeleton"])
        out.append(s)
        if len(out) == 2:
            break
    return out


def select_train(cands: list[dict], target_rows: int, tgt: dict) -> list[dict]:
    """Tham lam theo ô 5 chiều: mỗi bước thêm câu làm giảm nhiều nhất tổng L1 có trọng số; giữ ≥ 90% dòng ngành lõi, ≤ 15% dòng SQL mẫu."""
    rng = random.Random(2026)
    F = {x["qid"]: features(x) for x in cands}
    cells = collections.defaultdict(list)
    for x in sorted(cands, key=lambda x: x["qid"]):
        cells[tuple(F[x["qid"]][d] for d in DIM_W)].append(x)
    for v in cells.values():
        rng.shuffle(v)
        v.sort(key=lambda x: (x["_template"], -x["_rows"]))  # câu có lời giải model, 2 lời giải trước
    counts = {d: collections.Counter() for d in DIM_W}
    n_q, rows, core_rows, tpl_rows, shop_n = 0, 0, 0, 0, collections.Counter()
    chosen = []
    dims = list(DIM_W)
    texts = set()
    while rows < target_rows and any(cells.values()):
        for v in cells.values():  # câu trùng y hệt câu đã chọn (shop khác): bỏ
            while v and fold(v[0]["question"]) in texts:
                v.pop(0)
        gain_of = {}
        for d in dims:
            base = l1(counts[d], n_q, tgt[d])
            for k in {key[dims.index(d)] for key, v in cells.items() if v}:
                c2 = counts[d].copy()
                c2[k] += 1
                gain_of[(d, k)] = DIM_W[d] * (base - l1(c2, n_q + 1, tgt[d]))
        best, gain_best = None, None
        for key, v in cells.items():
            if not v:
                continue
            x = v[0]
            if x["_template"] and tpl_rows + x["_rows"] > 0.14 * max(target_rows, rows + x["_rows"]):
                continue
            if F[x["qid"]]["industry"] == "khac" and (rows + x["_rows"] - core_rows) > 0.08 * max(target_rows, rows + x["_rows"]):
                continue
            g = sum(gain_of[(d, key[i])] for i, d in enumerate(dims)) - 0.002 * shop_n[x["business_id"]]
            if gain_best is None or g > gain_best:
                best, gain_best = key, g
        if best is None:
            break
        x = cells[best].pop(0)
        texts.add(fold(x["question"]))
        chosen.append(x)
        n_q += 1
        rows += x["_rows"]
        core_rows += x["_rows"] if x["industry"] in WB.CORE else 0
        tpl_rows += x["_rows"] if x["_template"] else 0
        shop_n[x["business_id"]] += 1
        for d, k in zip(DIM_W, best):
            counts[d][k] += 1
    return chosen


def alias_of(rec, gold):
    """Mã hàng / mã khách là cách viết được chấp nhận của tên (scoring: alias)."""
    names = {}
    for it in rec["items"]:
        names.setdefault(fold(it["name"]), it["item"])
    if rec["opts"].get("codes_on_orders"):
        for c in rec["customers"]:
            names.setdefault(fold(c["name"]), c["cust"])
    t, v = gold["type"], gold.get("answer")
    if t == "string" and fold(v) in names:
        return [names[fold(v)]]
    if t == "list" and v and all(isinstance(x, str) for x in v):
        al = [[names[fold(x)]] if fold(x) in names else [] for x in v]
        return al if any(al) else None
    if t == "table" and v["rows"]:
        al = [[names[fold(r[0])]] if isinstance(r[0], str) and fold(r[0]) in names else [] for r in v["rows"]]
        return al if any(al) else None
    return None


def write_pkg(out: Path, role: str, questions: list[dict], plan: dict, sols: dict | None, dataset_id: str, generator_commit: str):
    if out.exists():
        shutil.rmtree(out)
    root = out / "data_root"
    (root / "data").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "train" / "rulebook.md", root / "rulebook.md")  # bản sao rulebook BTC: chỉ cục bộ, không gửi API
    shutil.copy(WORK / "shops_root" / "calendar.csv", root / "calendar.csv")
    bids = sorted({q["business_id"] for q in questions})
    businesses, lineages = {}, {}
    recs = {}
    for b in bids:
        shutil.copytree(WORK / "shops_root" / "data" / b, root / "data" / b)
        info, rec = SH.load_shop(WORK, b)
        recs[b] = rec
        s = plan[b]
        split = s["split"]
        businesses[b] = {"generator": "zaic26-datamate-synth eval/sqlv1 (synth_world.simulate + world.to_btc + render)", "generator_commit": generator_commit,
                         "seed": s["seed"], "oracle_version": WB.ORACLE_VERSION, "source_hash": info["source_hash"], "parent_example_id": s["parent_example_id"],
                         "schema_family": s["family"], "industry": s["industry"], "split": split, "origin": s["origin"], "old_name": s.get("old_name"),
                         "dataset_role": {"train": "synthetic_train", "dev": "synthetic_dev", "final": "synthetic_benchmark"}[split],
                         "data_start": s["start"], "today": s["today"], "complex": bool(s.get("complex")), "style": {k: v for k, v in s["style"].items() if v},
                         "view": info["layers"]["view_ok"], "debt_layer": info["layers"]["debt_ok"]}
        lineages[b] = {"parent_example_id": s["parent_example_id"], "schema_family": s["family"], "split": split}
    qrows, arows, cand = [], [], []
    for q in questions:
        qrows.append({"id": q["id"], "business_id": q["business_id"], "question": q["question"], "as_of_date": q["as_of_date"]})
        g = q["gold"]
        a = {"id": q["id"], "type": g["type"]} if g["type"] == "unanswerable" else {"id": q["id"], "type": g["type"], "answer": g["answer"]}
        if g["type"] != "unanswerable":
            al = alias_of(recs[q["business_id"]], g)
            if al:
                a["alias"] = al
        arows.append(a)
        for s in (sols or {}).get(q["qid"], []):
            cand.append({"id": q["id"], "business_id": q["business_id"], "split": plan[q["business_id"]]["split"], "completion": s["completion"],
                         "thinking_mode": "off", "metadata": {"solution_source": "template" if s["provider"] == "template" else "model",
                                                              "provider": s["provider"], "attempt": s.get("attempt"), "route": s.get("route"),
                                                              "skeleton_sha": hashlib.sha256(s["skeleton"].encode()).hexdigest()[:16],
                                                              "checks": {"exec_twice_same": True, "score": 1, "mutants": s.get("mutants"),
                                                                         "n_mutants_checked": s.get("n_mutants_checked"), "n_mutants_changed": s.get("n_mutants_changed"),
                                                                         "static": s.get("static") or []},
                                                              "slot": q["spec"]["slot"], "origin": q["origin"], "question_key": q["qid"]}})
    for name, rows in (("questions.jsonl", qrows), ("answers.jsonl", arows)):
        with open(root / name, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    man = {"dataset_id": dataset_id, "dataset_role": role, "training_eligible": role != "synthetic_benchmark", "schema_version": 1,
           "created": dt.date.today().isoformat(), "rules": "btc_v1 (eval/sqlv1/rules_spec.md); rulebook.md = bản sao rulebook BTC train (chỉ cục bộ)",
           "oracle_version": WB.ORACLE_VERSION, "upstream_commit": upstream_commit(), "generator_commit": generator_commit,
           "n_questions": len(qrows), "n_businesses": len(bids), "splits": dict(collections.Counter(plan[b]["split"] for b in bids)), "businesses": businesses}
    (root / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.path.append(str(UPSTREAM / "training"))
    from make_synthetic_fixture import content_hash  # hàm hash của exporter SFT v2 (upstream feat/synthetic-sql-v1-sft-v2)
    sm = {"kind": "synthetic_sft_source", "schema_version": 1, "lineage_by": "business", "dataset_id": dataset_id, "dataset_role": role,
          "training_eligible": role != "synthetic_benchmark", "data_root": "data_root", "data_hash": content_hash(root), "lineages": lineages}
    (out / "source_manifest.json").write_text(json.dumps(sm, ensure_ascii=False, indent=1), encoding="utf-8")
    if sols is not None:
        (out / "candidates").mkdir()
        with open(out / "candidates" / "solutions.jsonl", "w", encoding="utf-8") as f:
            for x in cand:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return {"questions": len(qrows), "businesses": len(bids), "candidates": len(cand)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rows", type=int, default=6600)
    ap.add_argument("--out", type=Path, default=controller_root() / "build")
    a = ap.parse_args()
    plan = {s["business_id"]: s for s in json.loads((WORK / "plan.json").read_text(encoding="utf-8"))}
    sols = collections.defaultdict(list)
    for f in sorted((WORK / "solutions").glob("pool_*.jsonl")):
        for x in load_jsonl(f):
            sols[x["qid"]].append(x)
    tr = load_jsonl(WORK / "questions" / "pool_train.jsonl") + load_jsonl(WORK / "questions" / "pool_train2.jsonl")
    dv = load_jsonl(WORK / "questions" / "pool_dev.jsonl")
    fi = load_jsonl(WORK / "questions" / "final_set.jsonl")
    from sqlv1 import verify as V
    qmap = {x["qid"]: x for x in tr + dv}
    n_restatic = 0
    for qid, ss in sols.items():  # luật ngày neo được nới sau khi một phần lời giải đã chấm: chỉ gỡ lỗi tĩnh thuần về ngày
        for x in ss:
            dateish = lambda e: e.startswith("date_not_anchored") or (e.startswith("literal_not_visible:") and V.DATETIME_LIT.fullmatch(e.split(":", 1)[1].strip()))  # noqa: E731
            if not x.get("ok") and (x.get("stage") or "").startswith("static:") and x.get("static") \
                    and all(dateish(e) for e in x["static"]) and qid in qmap and not V.restatic(x, qmap[qid]):
                x.update(ok=True, stage="ok", static=[], restatic=True)
                n_restatic += 1
    print("restatic (date rule relaxed):", n_restatic)
    # kiểm ngữ nghĩa tĩnh theo slot (thêm sau audit bởi LLM độc lập vòng 1): áp cho MỌI lời giải đạt, và loại câu hỏi tên mà đòi số
    sem = collections.Counter()
    roles_of = {}
    for qid, ss in sols.items():
        q = qmap.get(qid)
        if not q:
            continue
        b = q["business_id"]
        if b not in roles_of:
            info = json.loads((WORK / "shops" / b / "shop.json").read_text(encoding="utf-8"))
            roles_of[b] = (info["roles"], bool(info["plan"]["opts"].get("codes_on_orders")))
        for x in ss:
            if x.get("ok"):
                notes = [ln[2:] for ln in x["completion"].split("TYPE:")[0].splitlines() if ln.startswith("- ")]
                e = V.semantic_check(x["sql"], notes, q, *roles_of[b])
                if e:
                    x.update(ok=False, stage="semantic:" + e[0], semantic=e)
                    sem[e[0] + ":" + x["provider"]] += 1
    print("semantic rejections:", dict(sem))
    badq = {x["qid"]: V.question_ok(x) for x in tr + dv + fi}
    badq = {k: v for k, v in badq.items() if v}
    print("questions dropped (name question asks amount):", len(badq))
    tr = [x for x in tr if x["qid"] not in badq]
    dv = [x for x in dv if x["qid"] not in badq]
    fi = [x for x in fi if x["qid"] not in badq]
    chosen_sols, rejected, rej_rows = {}, collections.Counter(), []
    for x in tr + dv:
        b = best_solutions(sols.get(x["qid"], []))
        x["_rows"], x["_template"] = len(b), bool(b) and b[0]["provider"] == "template"
        if b:
            chosen_sols[x["qid"]] = b
        else:
            why = "not_attempted" if not sols.get(x["qid"]) else "all_failed"
            rejected["no_valid_solution:" + why] += 1
            stages = collections.Counter((s.get("provider"), (s.get("stage") or "?").split(":")[0]) for s in sols.get(x["qid"], []))
            rej_rows.append({"question_key": x["qid"], "business_id": x["business_id"], "split": x["split"], "slot": x["spec"]["slot"],
                             "reason": "no_valid_solution:" + why, "attempts": {f"{p}:{st}": n for (p, st), n in stages.items()}})
    for x in load_jsonl(WORK / "questions" / "gen_train.jsonl") + load_jsonl(WORK / "questions" / "gen_train_month.jsonl") + load_jsonl(WORK / "questions" / "gen_dev.jsonl"):
        if not x["verify"]["ok"]:
            rej_rows.append({"question_key": x["key"], "business_id": x["business_id"], "split": x["split"], "slot": x["spec"]["slot"],
                             "reason": "verifier_mismatch:" + x["verify"]["why"]})
    tgt = mining_targets()
    train = select_train([x for x in tr if x["_rows"]], a.target_rows, tgt)
    dev = [x for x in dv if x["_rows"]]
    rng = random.Random(11)
    allq = train + dev
    rng.shuffle(allq)
    for i, x in enumerate(allq, 1):
        x["id"] = f"syn_sales_v1:q{i:06d}"
    for i, x in enumerate(sorted(fi, key=lambda x: x["qid"]), 1):
        x["id"] = f"syn_sales_v1:q{900000 + i:06d}"
    gc = __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    r1 = write_pkg(a.out / "synthetic_sql_v1", "synthetic_train", allq, plan, chosen_sols, "syn_sales_sql_v1", gc)
    r2 = write_pkg(a.out / "synthetic_sql_v1_final", "synthetic_benchmark", fi, plan, None, "syn_sales_sql_v1_final", gc)
    notsel = [x for x in tr if x["_rows"] and x["qid"] not in {y["qid"] for y in train}]
    for x in notsel:
        rej_rows.append({"question_key": x["qid"], "business_id": x["business_id"], "split": x["split"], "slot": x["spec"]["slot"],
                         "reason": "not_selected:distribution_or_quota"})
    with open(a.out / "synthetic_sql_v1" / "candidates" / "rejected_questions.jsonl", "w", encoding="utf-8") as f:
        for x in rej_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    sel = {"train_questions": len(train), "train_rows": sum(x["_rows"] for x in train), "dev_questions": len(dev), "dev_rows": sum(x["_rows"] for x in dev),
           "final_questions": len(fi), "pkg": r1, "final_pkg": r2, "rejected_pool": dict(rejected),
           "train_template_rows": sum(x["_rows"] for x in train if x["_template"]), "train_core_rows": sum(x["_rows"] for x in train if x["industry"] in WB.CORE),
           "ids": {x["id"]: x["qid"] for x in allq + fi}}
    (WORK / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(WORK / "selected.jsonl", "w", encoding="utf-8") as f:
        for x in allq + fi:
            f.write(json.dumps({k: v for k, v in x.items() if k not in ("verify",)}, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: v for k, v in sel.items() if k != "ids"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
