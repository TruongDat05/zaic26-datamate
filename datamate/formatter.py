"""Đưa kết quả về một trong 9 kiểu đáp án:
number, percent, string, date, boolean, list, table, unanswerable, abstain.

`finalize` kiểm tra đúng format của luật chấm; sai format thì trả abstain (0 điểm) thay vì nộp
một đáp án chắc chắn bị chấm sai (−1).
"""
from __future__ import annotations

import math
import re

TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}
_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


class FormatError(ValueError):
    pass


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _num(v):
    if not _is_num(v):
        raise FormatError(f"không phải số: {v!r}")
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return round(v, 2) if isinstance(v, float) else v


def _cell(v):
    if _is_num(v):
        return _num(v)
    if isinstance(v, str) and v.strip():
        return v.strip()
    if isinstance(v, bool):
        return v
    raise FormatError(f"ô không hợp lệ: {v!r}")


def finalize(qid: str, ans: dict | None) -> dict:
    """Answer (dict có 'type') → một dòng submission.jsonl hợp lệ."""
    if not ans or ans.get("type") not in TYPES:
        return {"id": qid, "type": "abstain"}
    t = ans["type"]
    try:
        if t == "abstain":
            return {"id": qid, "type": "abstain"}
        if t == "unanswerable":
            out = {"id": qid, "type": "unanswerable"}
            if isinstance(ans.get("reason"), str) and ans["reason"]:
                out["reason"] = ans["reason"][:300]
            return out
        v = ans.get("answer")
        if t in ("number", "percent"):
            v = _num(v)
        elif t == "string":
            if not isinstance(v, str) or not v.strip():
                raise FormatError("string rỗng")
            v = v.strip()
        elif t == "date":
            if not isinstance(v, str) or not _DATE.match(v):
                raise FormatError(f"date sai format: {v!r}")
        elif t == "boolean":
            if not isinstance(v, bool):
                raise FormatError("boolean phải là true/false")
        elif t == "list":
            if not isinstance(v, list) or not v:
                raise FormatError("list rỗng hoặc không phải mảng")
            v = [_cell(x) for x in v]
        elif t == "table":
            if not isinstance(v, dict) or not isinstance(v.get("columns"), list) or not isinstance(v.get("rows"), list):
                raise FormatError("table phải có columns và rows")
            cols = [str(c) for c in v["columns"]]
            rows = [[_cell(x) for x in r] for r in v["rows"]]
            if not rows or any(len(r) != len(cols) for r in rows):
                raise FormatError("số ô mỗi dòng phải bằng số cột")
            v = {"columns": cols, "rows": rows}
        return {"id": qid, "type": t, "answer": v}
    except FormatError:
        return {"id": qid, "type": "abstain"}


def canonical(ans: dict) -> str:
    """Khoá so sánh hai đáp án (dùng khi bỏ phiếu giữa nhiều lần sinh)."""
    import json

    from .clean import norm_text

    t = ans.get("type")
    if t in ("unanswerable", "abstain"):
        return t
    v = ans.get("answer")
    if t == "string":
        v = norm_text(v)
    elif t == "list":
        v = [norm_text(x) if isinstance(x, str) else x for x in v]
    elif t == "table":
        v = {"rows": sorted(json.dumps([norm_text(x) if isinstance(x, str) else x for x in r], ensure_ascii=False)
                            for r in v["rows"])}
    return t + ":" + json.dumps(v, ensure_ascii=False, sort_keys=True)
