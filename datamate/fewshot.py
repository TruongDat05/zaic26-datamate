"""Tìm câu hỏi train giống câu đang hỏi để làm ví dụ về kiểu và format đáp án.

Chỉ dùng dữ liệu train (được phép theo luật 6). Không lấy ví dụ của cùng doanh nghiệp, để điểm
đo trên train/dev không bị lạc quan hơn lúc chạy test. Doanh nghiệp holdout (eval/holdout_businesses.txt)
bị loại hẳn khỏi nguồn ví dụ, để điểm holdout đo đúng trường hợp gặp doanh nghiệp lạ.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .clean import norm_key


def _grams(s: str, n: int = 3) -> Counter:
    s = f" {norm_key(s)} "
    c = Counter(s[i:i + n] for i in range(len(s) - n + 1))
    c.update(w for w in s.split() if len(w) > 1)
    return c


class FewShot:
    def __init__(self, questions: str | Path, answers: str | Path, exclude_businesses=(), code_bank=None):
        exclude = set(exclude_businesses)
        qs = [json.loads(l) for l in Path(questions).read_text(encoding="utf-8").splitlines() if l.strip()]
        qs = [q for q in qs if q["business_id"] not in exclude]
        ans = {json.loads(l)["id"]: json.loads(l) for l in Path(answers).read_text(encoding="utf-8").splitlines()
               if l.strip()}
        self.items = [(q, ans[q["id"]]) for q in qs if q["id"] in ans]
        self.vecs = [_grams(q["question"]) for q, _ in self.items]
        df = Counter()
        for v in self.vecs:
            df.update(v.keys())
        n = len(self.vecs)
        self.idf = {g: math.log((n + 1) / (c + 1)) + 1 for g, c in df.items()}
        # code đã chấm đúng (eval/build_code_bank.py), theo id câu train
        self.code = {}
        if code_bank and Path(code_bank).exists():
            for l in Path(code_bank).read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    self.code[r["id"]] = r["code"]
        self.norms = [math.sqrt(sum((w * self.idf[g]) ** 2 for g, w in v.items())) for v in self.vecs]

    def similar_code(self, question: str, business_id: str, k: int = 1) -> list[tuple[dict, str]]:
        """k câu giống nhất có code mẫu: [(câu hỏi, code)]."""
        if not self.code:
            return []
        return [(q, self.code[q["id"]]) for q, _ in self.similar(question, business_id, k, only_code=True)]

    def similar(self, question: str, business_id: str, k: int = 4, only_code: bool = False) -> list[tuple[dict, dict]]:
        qv = _grams(question)
        qn = math.sqrt(sum((w * self.idf.get(g, 1.0)) ** 2 for g, w in qv.items())) or 1.0
        scored = []
        for i, (v, (q, a)) in enumerate(zip(self.vecs, self.items)):
            if q["business_id"] == business_id or (only_code and q["id"] not in self.code):
                continue
            dot = sum(w * v[g] * self.idf[g] ** 2 for g, w in qv.items() if g in v)
            scored.append((dot / (qn * self.norms[i]), i))
        scored.sort(reverse=True)
        out, seen = [], set()
        for _, i in scored:
            q, a = self.items[i]
            key = norm_key(q["question"])
            if key in seen:
                continue
            seen.add(key)
            out.append((q, a))
            if len(out) == k:
                break
        return out
