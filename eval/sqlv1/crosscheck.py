"""Đối chiếu oracle Python (world.gold) với SQL mẫu (sqlgen) chạy bằng sqlexec trên shop đã nạp qua ingest upstream.

Lệch nào cũng là lỗi ở oracle, bộ dựng file, SELECT ánh xạ hoặc SQL mẫu. Dùng làm test hồi quy và để đo độ phủ SQL mẫu.
    python eval/sqlv1/crosscheck.py --work outputs/synth_sql_v1 --shops syn_shop_0001,... [--per-slot 2]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.up import scoring, sqlexec  # noqa: E402
from sqlv1 import render as RD, shops as SH, specs as SP, sqlgen as G, world as WB  # noqa: E402


def check_shop(work: Path, bid: str, per_slot: int = 2, seed=0) -> list[dict]:
    info, rec = SH.load_shop(work, bid)
    db, meta, layers = SH.open_db(work, bid, info)
    raw = RD.raw_info(info["roles"], RD.relations(layers["N"]))
    out = []
    fam = info["plan"]["family"]
    for slot in SP.SLOT_W:
        for sp in SP.sample(rec, fam, per_slot, f"x{seed}:{slot}", force_slots=[slot]):
            try:
                kind, sql, notes = G.for_spec(sp, rec, raw)
            except G.Unsupported:
                out.append({"bid": bid, "slot": slot, "res": "unsupported"})
                continue
            text = G.completion(kind, sql, notes)
            try:
                pred = sqlexec.execute(db, text, "")
                ok = scoring.score(pred, sp["gold"]) == 1
                out.append({"bid": bid, "slot": slot, "res": "ok" if ok else "wrong", "gold": sp["gold"], "pred": pred, "spec": {k: sp[k] for k in ("period", "base_period", "params", "as_of")},
                            "sql": sql if not ok else None})
            except Exception as e:  # noqa: BLE001
                out.append({"bid": bid, "slot": slot, "res": "error", "err": str(e)[:200], "gold": sp["gold"], "spec": {k: sp[k] for k in ("period", "base_period", "params", "as_of")}, "sql": sql})
    db.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=SH.SYNTH_ROOT / "outputs" / "synth_sql_v1")
    ap.add_argument("--shops", required=True)
    ap.add_argument("--per-slot", type=int, default=2)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    res = []
    import concurrent.futures as cf
    shops = a.shops.split(",")
    with cf.ProcessPoolExecutor(a.procs) as ex:
        for b, r in zip(shops, ex.map(check_shop, [a.work] * len(shops), shops, [a.per_slot] * len(shops))):
            res += r
            c = collections.Counter(x["res"] for x in r)
            print(b, dict(c), flush=True)
    by = collections.defaultdict(collections.Counter)
    for x in res:
        by[x["slot"]][x["res"]] += 1
    bad = {k: dict(v) for k, v in by.items() if v["wrong"] or v["error"]}
    print("slots with problems:", json.dumps(bad, ensure_ascii=False))
    tot = collections.Counter(x["res"] for x in res)
    print("total", dict(tot))
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=0, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
