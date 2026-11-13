"""Tìm các dòng rulebook nhắc tới đúng khái niệm trong câu hỏi, để đặt ngay cạnh câu hỏi.

Rulebook public dài ~24 KB, có cả luật của ngành khác; model cỡ 9–14B hay bỏ sót luật định nghĩa riêng
("khách sộp nhất" chỉ xét khách từ 3 đơn, "giờ cao điểm" trả số giờ…) dù luật đã nằm trong system prompt.
Trích lại vài dòng khớp tên luật với câu hỏi giúp model áp đúng định nghĩa. Chỉ thêm, không bớt rulebook.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .clean import norm_key

# từ quá chung, khớp với gần như câu nào
_STOP = {"cua", "la", "va", "trong", "theo", "mot", "cac", "nhung", "co", "bao", "nhieu", "ky", "thi",
         "nhat", "x", "y", "n", "m", "d", "q", "h", "so", "tien", "don", "hang", "mat", "ngay", "thang", "nam"}

# viết tắt hay gặp trong câu hỏi (prompt hệ thống có liệt kê; ở đây chỉ để ghép tên luật)
_SLANG = {"dthu": "doanh thu", "dt": "doanh thu", "sp": "san pham", "kh": "khach", "nv": "nhan vien",
          "sl": "so luong", "tb": "trung binh", "ncc": "nha cung cap"}


@dataclass
class Rule:
    section: str
    name: str
    text: str
    aliases: list  # mỗi alias là tuple từ khoá (không dấu)


def _clean_cell(s: str) -> str:
    return re.sub(r"[*`]", "", s).strip()


def parse(rulebook: str) -> list[Rule]:
    rules, section = [], ""
    for line in rulebook.splitlines():
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in ("Luật", "Cách nói", "Cách ghi", "Từ", "type"):
            continue
        name = _clean_cell(cells[0])
        aliases = []
        for part in re.split(r"[,/;]| hoặc ", name):
            toks = tuple(t for t in norm_key(re.sub(r"<[^>]*>|\([^)]*\)", " ", part)).split() if t not in _STOP)
            if toks:
                aliases.append(toks)
        if aliases:
            rules.append(Rule(section, name, " | ".join(cells), aliases))
    return rules


def relevant(rules: list[Rule], question: str, k: int = 5) -> list[Rule]:
    """Dòng luật có một tên gọi mà mọi từ khoá đều có trong câu hỏi; tên càng nhiều từ càng ưu tiên."""
    words = set(norm_key(question).split())
    for short, full in _SLANG.items():
        if short in words:
            words.update(full.split())
    scored = []
    for i, r in enumerate(rules):
        best = 0
        for toks in r.aliases:
            if all(t in words for t in toks) and (len(toks) >= 2 or len(toks[0]) >= 4):
                best = max(best, len(toks))
        if best:
            scored.append((-best, i, r))
    return [r for _, _, r in sorted(scored)[:k]]


# Lỗi model lặp lại trên holdout: code chạy được, các phiếu đồng ý với nhau, nhưng sai cùng một kiểu. Tính lại từ dữ liệu ra
# đúng đáp án: đếm dòng thay vì đếm đơn (92 → 60), lấy cả tháng trước thay vì cùng số ngày (−12,96 → 19,85).
_CONCEPTS = [
    (re.compile(r"\b(ti le|ty le)\b.*\bhuy\b|\bhuy\b.*(%|phan tram|\b(ti le|ty le)\b)"),
     "Tỉ lệ đơn huỷ tính theo ĐƠN, không theo dòng: `dm.cancel_rate(df_ky, cot_ma_don, cot_trang_thai)`, df_ky gồm mọi trạng thái."),
    (re.compile(r"\b(may|bao nhieu|bao nhiu|bnhiu|bn|so|dem)\s+(don|hoa don|phieu|bill)\b|\bhuy\s+(may|bao nhieu|bn|bnhiu)\b"),
     "Đếm ĐƠN, không đếm dòng (một đơn có nhiều dòng hàng): `dm.n_orders(df, cot_ma_don)`, không dùng `len(df)`."),
    (re.compile(r"\b(dot|luot|chuyen)\b.*\b(giao|cho|cho hang)\b|\b(giao|cho)\b.*\b(dot|luot|chuyen)\b"),
     "Mỗi đợt của mỗi đơn tính một đợt: `dm.n_distinct(df, [cot_ma_don, cot_luot])`, không dùng `len(df)` (một đợt có nhiều dòng hàng)."),
    (re.compile(r"\bcung ky\b.*\b(thang truoc|th truoc)\b|\b(dau thang|thang nay)\b.*\b(thang truoc|th truoc|cung ky)\b"),
     "Kỳ đang tính là `dm.this_month(TODAY)`, kỳ so là `dm.prev_month_same_days(TODAY)` (cùng số ngày), KHÔNG lấy cả tháng trước."),
    (re.compile(r"\b(binh quan|trung binh|tb)\b.*\b(tien|gia|doanh thu|dthu)\b|\b(tien|gia|doanh thu|dthu)\b.*\b(binh quan|trung binh|tb)\b"),
     "Trung bình của một khoản tiền cũng là tiền: `dm.money(...)`, làm tròn về đồng."),
]


def concept_block(question: str) -> str:
    q = norm_key(question)
    hits = [text for pat, text in _CONCEPTS if pat.search(q)]
    if "%" in question and re.search(r"\bhuy\b", q) and _CONCEPTS[0][1] not in hits:
        hits.insert(0, _CONCEPTS[0][1])
    return "## Lưu ý tránh lỗi hay gặp\n" + "\n".join(f"- {h}" for h in hits) + "\n\n" if hits else ""


def block(rules: list[Rule], question: str, k: int = 5) -> str:
    hits = relevant(rules, question, k)
    if not hits:
        return ""
    lines = ["## Luật rulebook nhắc tới khái niệm trong câu hỏi (áp dụng đúng định nghĩa này)"]
    lines += [f"- [{r.section}] {r.text}" for r in hits]
    return "\n".join(lines) + "\n\n"
