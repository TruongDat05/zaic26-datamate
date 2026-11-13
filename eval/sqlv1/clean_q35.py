"""Bộ SFT sạch nhất cho Qwen3.5-9B: danh sách ứng viên cần loại (mức ứng viên, có lý do) từ một build đã xuất, không sinh lại gì.

    python eval/sqlv1/clean_q35.py --build <build>/sft --out outputs/synth_sql_v1/sft_q35/clean_exclusions.json
      → rồi: bundle.py --exclude-candidates <out> ... ; build_sft.py ... (exporter cấp manifest mới cho bộ sạch)

Luật (đo trên dòng của build, ghi số từng luật):
- mutation_not_discriminating: câu trả lời được (không phải unanswerable) mà không bản đột biến nào làm đổi đáp án: SQL khớp gold nhưng
  chưa có bằng chứng mạnh rằng nó không khớp do tình cờ.
- price_not_catalog: % đổi giá / giá ngày D mà giá danh mục (bảng giá, lịch sử giá của thế giới sinh) không đổi / khác: chênh lệch chỉ do
  bán theo gói có chiết khấu, câu hỏi hiểu được theo hai cách.
- final_text_twin: câu chữ trùng một câu FINAL (khác shop) — để FINAL sạch khi dùng đánh giá.
- money_round_2dp: đáp án tiền làm tròn 2 chữ số lẻ (luật: tiền làm tròn đồng; khớp gold chỉ vì số tròn).
- thinking_not_audit_passed: dòng thinking không có phán quyết ĐẠT của audit độc lập (mọi dòng thinking của build đã được audit).
- audit_failed: mọi dòng có phán quyết KHÔNG ĐẠT ở bất kỳ vòng audit nào (dòng nào còn lại).
- list_order_not_guaranteed: đáp án list (scorer so theo thứ tự) mà truy vấn ngoài cùng không có ORDER BY: đúng nhờ kế hoạch thực thi.
- repetitive_trace: vết thinking lặp (một dòng ≥ 12 ký tự lặp ≥ 3 lần, hoặc một cụm 8 từ lặp ≥ 5 lần).
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

WORK = Path(__file__).resolve().parents[2] / "outputs" / "synth_sql_v1"
FINAL = controller_root() / "build/synthetic_sql_v1_final/data_root/questions.jsonl"
MONEY = {"revenue", "revenue_item", "revenue_channel", "revenue_payment", "avg_order", "avg_cost", "avg_daily_rev", "rev_change", "staff_rev",
         "shift_rev", "purchase_money", "profit_gross", "price_on_day", "debt_total", "debt_cust", "debt_new", "debt_collected", "debt_older", "rx_rev"}
ROUND2 = re.compile(r"(?is)round\((?:[^()]|\([^()]*\))*,\s*2\s*\)")


def outer_has_order(sql: str) -> bool:
    """ORDER BY at parenthesis depth 0 (string literals blanked)."""
    s, depth, out = re.sub(r"'[^']*'", "''", sql), 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return bool(re.search(r"(?i)\border\s+by\b", "".join(out)))


def repetitive(trace: str) -> bool:
    lines = collections.Counter(x.strip() for x in trace.split("\n") if len(x.strip()) >= 12)
    w = re.findall(r"\w+", trace.lower())
    grams = collections.Counter(" ".join(w[i:i + 8]) for i in range(len(w) - 7))
    return bool(lines and lines.most_common(1)[0][1] >= 3) or bool(grams and grams.most_common(1)[0][1] >= 5)


def fold(s: str) -> str:
    """The gates' own normalisation (sqlv1.questions.fold: accents, case, punctuation), so 'twin' means the same here and in gates.json."""
    from sqlv1.questions import fold as qfold
    return qfold(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from synth_world import resolve
    rows = [json.loads(x) for s in ("train", "dev") for x in (a.build / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    sel = {x["id"]: x for x in map(json.loads, (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines()) if x.get("id")}
    final_texts = {fold(json.loads(x)["question"]) for x in FINAL.read_text(encoding="utf-8").splitlines() if x.strip()}
    verdicts = {}
    for f in sorted(glob.glob(str(WORK / "audit_q35" / "verdicts" / "*.jsonl"))):
        for x in map(json.loads, Path(f).read_text(encoding="utf-8").splitlines()):
            verdicts[x["row"]] = (bool(x.get("pass")), x.get("ly_do") or "", Path(f).stem)
    recs = {}

    def rec(b):
        if b not in recs:
            recs[b] = pickle.load(gzip.open(WORK / "shops" / b / "rec.pkl.gz"))
        return recs[b]

    out, by = [], collections.Counter()
    for r in rows:
        m = r["metadata"]
        s = sel[m["question_id"]]
        spec, slot = s["spec"], s["spec"].get("slot")
        final = r["completion"].split("</think>")[-1]
        sql = final.split("```sql")[-1]
        why = []
        if m["answer_type"] != "unanswerable" and not m["mutation"]["discriminating"]:
            why.append(("mutation_not_discriminating", "no mutant changed the answer"))
        if fold(m["question"]) in final_texts:
            why.append(("final_text_twin", "same text as a FINAL question (other shop)"))
        if m["answer_type"] == "number" and slot in MONEY and ROUND2.search(sql):
            why.append(("money_round_2dp", "money answer rounded to 2 decimals"))
        if slot in ("price_change_pct", "price_on_day"):
            R = rec(r["business_id"])
            it = spec["params"]["item"]
            base = next(i for i in R["items"] if i["item"] == it)
            hist = sorted((h for h in (R.get("price_history") or []) if h["item"] == it), key=lambda h: h["from"])

            def cat(d):
                p = base["price"]
                for h in hist:
                    if h["from"] <= d:
                        p = h["price"]
                return p
            gold = s["gold"]["answer"]
            if slot == "price_change_pct":
                c1, c2 = cat(spec["period"]["d1"]), cat(spec["period"]["d2"])
                if c1 == c2 or abs(gold - round(100 * (c2 - c1) / c1, 2)) > 0.011:
                    why.append(("price_not_catalog", f"catalog {c1} -> {c2}, gold {gold}%"))
            else:
                d = str(resolve(spec["period"], spec["as_of"])[0])
                if abs(float(gold) - cat(d)) > 0.5:
                    why.append(("price_not_catalog", f"catalog {cat(d)} vs gold {gold}"))
        if m["answer_type"] == "list" and not outer_has_order(sql.split("```")[0]):
            why.append(("list_order_not_guaranteed", "list answer, no ORDER BY in the outermost query"))
        if r["thinking_mode"] and repetitive(r["completion"].split("</think>")[0]):
            why.append(("repetitive_trace", "a line or an 8-word span repeats in the reasoning"))
        v = verdicts.get(r["id"])
        if v and not v[0]:
            why.append(("audit_failed", f"{v[2]}: {v[1]}"[:200]))
        if r["thinking_mode"] and not (v and v[0]):
            why.append(("thinking_not_audit_passed", "no PASS verdict of the independent audit"))
        if why:
            by.update({w for w, _ in why})
            out.append({"candidate_id": m["sample_id"], "row": r["id"], "question_id": m["question_id"], "split": m["split"],
                        "thinking": r["thinking_mode"], "reasons": [w for w, _ in why], "reason": why[0][0], "detail": "; ".join(d for _, d in why)[:300]})
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    left = collections.Counter(r["metadata"]["split"] for r in rows if r["id"] not in {x["row"] for x in out})
    print(json.dumps({"rows_in": len(rows), "excluded": len(out), "by_rule": dict(by), "left": dict(left),
                      "thinking_left": sum(1 for r in rows if r["thinking_mode"] and r["id"] not in {x["row"] for x in out})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
