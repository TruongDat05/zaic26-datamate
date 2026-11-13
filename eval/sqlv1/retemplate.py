"""Sinh lại SQL mẫu của các slot đọc bảng thô theo mã hàng (giá ngày, % đổi giá, giá nhập bình quân, tồn) sau khi sqlgen tra mã theo TÊN
hàng (sqlgen._item_in): SQL cũ ghi cứng mã ('NT003') mà prompt không nối mã với tên (VALUES MATCHING chỉ liệt kê tên; mã chỉ nằm trong
danh sách giá trị của schema) → model học đoán mã. Không gọi API.

    DATAMATE_UPSTREAM=<upstream> python eval/sqlv1/retemplate.py [--procs 8]
      → outputs/synth_sql_v1/solutions/pool_retemplate.template.jsonl (dòng sự kiện, như solve.py template)
      → <pkg>/candidates/solutions.jsonl (ứng viên mẫu bị ảnh hưởng thay bằng SQL mới; bản cũ: solutions.before_retemplate.jsonl)
      → outputs/synth_sql_v1/sft_q35/retemplate_summary.json

Mỗi SQL mới qua đúng judge() của solve.py (chạy 2 lần, khớp gold oracle, đột biến, kiểm tĩnh). SQL mới trượt → ứng viên bị bỏ (ghi lý do).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

from sqlv1 import render as RD, shops as SH, solve as SV, sqlgen as G, verify as V  # noqa: E402

WORK = SH.SYNTH_ROOT / "outputs" / "synth_sql_v1"
PKG = controller_root() / "build/synthetic_sql_v1"
SLOTS = {"price_on_day", "price_change_pct", "avg_cost", "stock_qty"}


def shop_job(args):
    bid, items, specs = args
    env = V.ShopEnv(WORK, bid, specs)
    raw = RD.raw_info(env.info["roles"], RD.relations(env.layers["N"]))
    out = []
    try:
        for c, q in items:
            try:
                kind, sql, notes = G.for_spec(q["spec"], env.rec, raw)
            except G.Unsupported as e:
                out.append({"candidate": c, "line": {"qid": q["qid"], "provider": "template", "attempt": "template", "ok": False, "stage": f"unsupported:{e}"}})
                continue
            it = {"type": kind, "sql": sql, "notes": notes}
            hits, periods = env.hints(q)
            res = SV.judge(env, q, it, env.layers["ctx"] + "\n" + "\n".join(hits + periods))
            out.append({"candidate": c, "line": {"qid": q["qid"], "provider": "template", "attempt": "template", "completion": SV.completion(it), "sql": sql,
                                                 "type": kind, **res}})
    finally:
        env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    src = PKG / "candidates" / "solutions.jsonl"
    backup = PKG / "candidates" / "solutions.before_retemplate.jsonl"
    if not backup.exists():
        shutil.copy2(src, backup)
    cands = [json.loads(x) for x in backup.read_text(encoding="utf-8").splitlines() if x.strip()]
    sel = {x["id"]: x for x in map(json.loads, (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines()) if x.get("id")}
    specs = collections.defaultdict(list)
    for x in sel.values():
        specs[x["business_id"]].append(x["spec"])
    todo = collections.defaultdict(list)
    for i, c in enumerate(cands):
        m = c["metadata"]
        if m["solution_source"] == "template" and m.get("slot") in SLOTS:
            todo[c["business_id"]].append((i, sel[c["id"]]))
    results = []
    with cf.ProcessPoolExecutor(a.procs) as ex:
        for r in ex.map(shop_job, [(b, todo[b], specs[b]) for b in sorted(todo)]):
            results += r
    lines = WORK / "solutions" / "pool_retemplate.template.jsonl"
    with open(lines, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r["line"], ensure_ascii=False, default=str) + "\n")
    new, dropped, st = list(cands), [], collections.Counter()
    for r in results:
        i, ln = r["candidate"], r["line"]
        old = cands[i]
        if not ln.get("ok"):
            dropped.append({"id": old["id"], "slot": old["metadata"]["slot"], "stage": ln.get("stage"), "error": str(ln.get("error"))[:200]})
            new[i] = None
            st["dropped:" + str(ln.get("stage")).split(":")[0]] += 1
            continue
        m = dict(old["metadata"])
        m["retemplated"] = {"reason": "item looked up by its name (sqlgen._item_in), no code literal", "old_skeleton_sha": m["skeleton_sha"],
                            "old_completion_sha": hashlib.sha256(old["completion"].encode()).hexdigest()}
        m["skeleton_sha"] = hashlib.sha256(V.skeleton(ln["sql"]).encode()).hexdigest()[:16]
        m["route"] = ln.get("route") or m.get("route")
        m["checks"] = {"exec_twice_same": True, "score": 1, "mutants": ln.get("mutants"), "n_mutants_checked": ln.get("n_mutants_checked"),
                       "n_mutants_changed": ln.get("n_mutants_changed"), "static": ln.get("static")}
        new[i] = {**old, "completion": ln["completion"], "metadata": m}
        st["retemplated"] += 1
    src.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in new if c is not None), encoding="utf-8")
    summ = {"affected": len(results), **st, "by_slot": dict(collections.Counter(cands[r["candidate"]]["metadata"]["slot"] for r in results)),
            "dropped": dropped, "events": str(lines), "candidates_file": str(src), "backup": str(backup),
            "sqlgen_sha256": "sha256:" + hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest()}
    (WORK / "sft_q35").mkdir(parents=True, exist_ok=True)
    (WORK / "sft_q35" / "retemplate_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "dropped"}, ensure_ascii=False), "dropped", len(dropped))


if __name__ == "__main__":
    main()
