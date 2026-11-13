"""Bước 1.3 (phần 1, chạy bằng venv sinh cũ, pandas 2.2.3): lấy THAM SỐ của các câu train thuộc dạng câu phổ biến.

Chạy lớp template cũ (datamate/templates.py của repo sinh, đã khớp gold) và ghi lại lời gọi cấp ngoài cùng của Shop
(revenue/n_orders/counted/...): kỳ [đầu, cuối], mặt hàng, nhóm. KHÔNG ghi câu hỏi, chỉ ghi id + tham số; không in câu hỏi.

    python eval/sqlv1/btc_capture.py OUT.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

FORMS = ("revenue", "n_orders", "cancel_rate", "n_cancelled", "avg_order", "qty", "best_day", "top_items", "top_item",
         "top_customer", "top_customers", "share_group", "rev_change_pct", "rev_change", "days_over", "orders_over", "best_month",
         "busiest_day", "g_rev")


def main(out: Path):
    from datamate import shop as S
    from synth_tools import Kit
    calls, depth = [], [0]

    def ser(v):
        if hasattr(v, "start") and hasattr(v, "end"):
            return {"start": v.start.isoformat(), "end": v.end.isoformat()}
        if isinstance(v, (dt.date,)):
            return v.isoformat()
        if isinstance(v, (str, int, float, type(None))):
            return v
        return "<df>"

    for name in ("revenue", "n_orders", "n_cancelled", "cancel_rate", "avg_order", "qty", "share_group", "rank_items", "customers", "counted", "in_period", "by_shift"):
        orig = getattr(S.Shop, name)

        def wrap(self, *a, _o=orig, _n=name, **k):
            if depth[0] == 0:
                calls.append([_n, [ser(x) for x in a], {kk: ser(vv) for kk, vv in k.items()}])
            depth[0] += 1
            try:
                return _o(self, *a, **k)
            finally:
                depth[0] -= 1
        setattr(S.Shop, name, wrap)

    kit = Kit()
    seeds = [json.loads(line) for line in open(ROOT / "data/synth_source/seeds/questions.jsonl", encoding="utf-8")]
    gold = {x["id"]: x for x in map(json.loads, open(ROOT / "data/train/answers.jsonl", encoding="utf-8"))}
    res, n_form = {}, 0
    for s in seeds:
        tpl = (s.get("route") or {}).get("template")
        if tpl not in FORMS:
            continue
        n_form += 1
        calls.clear()
        a = kit.oracle(s["business_id"], s["question"], s["as_of_date"])
        if a is None:
            res[s["id"]] = {"business_id": s["business_id"], "template": tpl, "status": "no_template"}
            continue
        import re
        th = re.search(r"\d{1,3}(?:[.,]\d{3})+", s["question"]) if a["template"] in ("days_over", "orders_over") else None
        res[s["id"]] = {"business_id": s["business_id"], "as_of": s["as_of_date"], "template": a["template"], "calls": list(calls),
                        "threshold": int(re.sub(r"[.,]", "", th.group(0))) if th else None,
                        "old_type": a["type"], "old_answer": a["answer"], "old_ok": a["type"] == gold[s["id"]]["type"] and a["answer"] == gold[s["id"]].get("answer")}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding="utf-8")
    ok = sum(1 for r in res.values() if r.get("old_ok"))
    print(f"dạng câu áp dụng: {n_form}; bắt được tham số: {sum('calls' in r for r in res.values())}; lớp cũ khớp gold: {ok}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
