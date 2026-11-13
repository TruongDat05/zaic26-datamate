"""Gom code đã chấm đúng từ các lượt chạy trên train thành kho code mẫu cho few-shot.

Model cỡ 9–14B học cách áp luật (lọc đơn huỷ/trả, quy đổi đơn vị, xếp hạng bằng nhau…) từ một đoạn code mẫu tốt hơn
nhiều so với chỉ thấy đáp án. Chỉ lấy câu train (luật 6), chỉ lấy đáp án khớp gold, và chỉ lấy câu có đồng thuận.
Doanh nghiệp holdout vẫn bị FewShot loại khi chạy, nên điểm holdout không bị lạc quan.

    python eval/build_code_bank.py --runs OUT/dev_run OUT/train_bank --gold data/train/answers.jsonl \\
        --out data/code_bank.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from datamate import formatter  # noqa: E402
from datamate.llm import extract_code  # noqa: E402
from score import match  # noqa: E402


def winning_code(r: dict) -> str | None:
    """Code ngắn nhất trong các bước cho ra đúng đáp án cuối."""
    win = formatter.canonical(r["final"])
    codes = []
    for st in r.get("steps", []):
        res = st.get("result") or {}
        if not res.get("ok"):
            continue
        fin = formatter.finalize(r["id"], res["answer"])
        if formatter.canonical(fin) == win:
            code = extract_code(st.get("response") or "")
            if code:
                codes.append(code.strip())
    return min(codes, key=len) if codes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gold", default=str(ROOT / "data/train/answers.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/code_bank.jsonl"))
    a = ap.parse_args()
    gold = {}
    for l in Path(a.gold).read_text(encoding="utf-8").splitlines():
        if l.strip():
            g = json.loads(l)
            gold[g["id"]] = g
    bank = {}
    if Path(a.out).exists():  # gộp với kho cũ
        for l in Path(a.out).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                bank[r["id"]] = r
    n_logs = n_ok = 0
    for run in a.runs:
        for p in (Path(run) / "logs").glob("q_*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("id") not in gold or r.get("decision") != "consensus":
                continue
            n_logs += 1
            if match(r["final"], gold[r["id"]]) != 1:
                continue
            code = winning_code(r)
            if code and len(code) < 4000:
                n_ok += 1
                bank[r["id"]] = {"id": r["id"], "business_id": r["business_id"], "question": r["question"], "code": code}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in sorted(bank.values(), key=lambda r: r["id"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{n_logs} câu đồng thuận có gold, {n_ok} câu đúng có code → kho {len(bank)} câu: {a.out}")


if __name__ == "__main__":
    main()
