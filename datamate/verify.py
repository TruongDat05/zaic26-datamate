"""Verifier quy tắc: chặn đáp án sai kiểu so với cách hỏi, trước khi tính phiếu.

Chỉ dùng các quy tắc đã đo trên 3.000 câu train không chặn nhầm đáp án chuẩn nào
(tests/test_verify.py chạy lại phép đo đó). Đáp án unanswerable / abstain không bị kiểm tra.
Lỗi trả về kèm lời nhắn để LLM sửa code (agent gửi lại như một lượt sửa lỗi).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from . import clean
from .clean import norm_key, strip_accents


@dataclass
class Issue:
    code: str
    message: str


_PERCENT = re.compile(r"phan tram|\bty le\b|\bti le\b|ty trong|ti trong")
# "%" đứng một mình hay nằm trong tên hàng ("Otrivin 0,1%"), nên chỉ tính khi đi sau từ hỏi lượng
_PERCENT_SIGN = re.compile(r"(bao nhieu|bao nhiu|bnhiu|bn|may|hon|kem|tang|giam|chiem)\s*%")
_MONTH = re.compile(r"\bthang nao\b")
_DAY = re.compile(r"\bngay nao\b")
_BOOL = re.compile(r"\b(co|da|con)\b.*\b(khong|ko)\s*[?.!]*\s*$|\b(phai khong|dung khong|hay khong)\b")
_MONEY = re.compile(r"\b(bao nhieu|bn|bnhiu|bao nhiu|may) tien\b")
_TOPN = re.compile(r"\btop ?(\d+)\b")


def _is_int(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer()


def check(question: str, ans: dict) -> list[Issue]:
    """ans: đáp án đã qua formatter.finalize. Trả về danh sách lỗi (rỗng = đạt)."""
    t = ans.get("type")
    if t in ("unanswerable", "abstain", None):
        return []
    v = ans.get("answer")
    q = norm_key(question)
    issues: list[Issue] = []
    if (_PERCENT.search(q) or _PERCENT_SIGN.search(strip_accents(question.lower()))) and t != "percent":
        issues.append(Issue("TYPE_MISMATCH", "Câu hỏi hỏi tỉ lệ / phần trăm: đáp án phải là "
                                             "dm.percent / dm.share / dm.pct_change."))
    if _MONTH.search(q) and not (t == "date" and len(v) == 7):
        issues.append(Issue("TYPE_MISMATCH", "Câu hỏi 'tháng nào': đáp án phải là dm.month_answer(...) dạng YYYY-MM."))
    if _DAY.search(q) and not (t == "date" and len(v) == 10):
        issues.append(Issue("TYPE_MISMATCH", "Câu hỏi 'ngày nào': đáp án phải là dm.date(...) dạng YYYY-MM-DD."))
    if _BOOL.search(q) and t != "boolean":
        issues.append(Issue("TYPE_MISMATCH", "Câu hỏi có / không: đáp án phải là dm.boolean(...)."))
    if _MONEY.search(q) and not ((t == "number" and _is_int(v)) or (t == "list" and all(_is_int(x) for x in v))):
        issues.append(Issue("TYPE_MISMATCH", "Câu hỏi bao nhiêu tiền: đáp án là tiền VND nguyên, dùng dm.money(...) "
                                             "(nhiều số thì dm.list_ các số tiền)."))
    m = _TOPN.search(q)
    if m:
        k = int(m.group(1))
        n = len(v) if t == "list" else len(v["rows"]) if t == "table" else None
        if n is None:
            issues.append(Issue("TYPE_MISMATCH", f"Câu hỏi top {k}: đáp án phải là dm.list_ (hoặc dm.table nếu hỏi kèm số đo)."))
        elif n > k:
            issues.append(Issue("TOPK_TOO_LONG", f"Câu hỏi top {k} nhưng đáp án có {n} phần tử; chỉ lấy {k} phần tử đầu."))
    return issues


_MONEY_WORD = re.compile(r"\b(doanh thu|dthu|dt|tien|gia|gia tri|chi phi|von|lai|no|thu no|tra no|luong|phi|hoa hong)\b")


def round_money(question: str, ans: dict) -> dict:
    """Tiền ra số lẻ làm tròn về đồng, kể cả trung bình của một khoản tiền (rulebook §9).

    Câu sai 70873.95 so với 70874: model tính đúng nhưng quên dm.money. Đã đo trên train: 0/1884 đáp án số là số lẻ."""
    v = ans.get("answer")
    if ans.get("type") == "number" and isinstance(v, float) and not v.is_integer() and _MONEY_WORD.search(norm_key(question)):
        return {**ans, "answer": int(Decimal(repr(v)).quantize(Decimal(1), rounding=ROUND_HALF_UP))}
    return ans


_ZERO_OK = re.compile(r"\b(e|khong ban|chua ban|ton|het hang)\b")


def zero_suspect(question: str, ans: dict) -> bool:
    """Đáp án 0 gần như luôn do lọc rỗng: gold train chỉ có 2/1884 số bằng 0, và cả hai đều là câu 'mặt hàng ế'.

    Trên public v2.5 có 55 đáp án 0 (lọc tên không khớp sổ ghi tắt, dòng phụ thiếu ngày)."""
    if ans.get("type") == "number":
        zero = ans.get("answer") == 0
    elif ans.get("type") == "list":
        v = ans.get("answer") or []
        zero = bool(v) and all(x == 0 for x in v)
    else:
        return False
    return zero and not _ZERO_OK.search(clean.norm_key(question))
