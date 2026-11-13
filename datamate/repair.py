"""Chẩn đoán loại lỗi rồi mới yêu cầu sửa: model cỡ nhỏ sửa tốt hơn khi được nói rõ lỗi thuộc loại nào và cần xem lại gì."""
from __future__ import annotations

import difflib
import re


def _columns(biz) -> list:
    return [c for df in biz.tables.values() for c in df.columns if not str(c).startswith("_")] if biz is not None else []


def classify(error: str, biz=None) -> tuple[str, str]:
    """(loại lỗi, gợi ý sửa)."""
    e = error or ""
    m = re.search(r"KeyError: ['\"](.+?)['\"]", e)
    if m:
        near = difflib.get_close_matches(m[1], _columns(biz), n=5, cutoff=0.4)
        return "SAI_TEN_COT", f"Không có cột hoặc bảng `{m[1]}`. Tên gần nhất: {near}. Chỉ dùng tên trong phần DỮ LIỆU, chép nguyên văn."
    if re.search(r"Cannot compare|Invalid comparison|not supported between instances of .*(date|Timestamp)", e):
        return "SO_SANH_NGAY", "So sánh cột ngày với kiểu khác. Dùng p.mask(cot_ngay) hoặc pd.Timestamp(TODAY)."
    if re.search(r"could not convert|invalid literal|unsupported operand type.*str|can't multiply", e):
        return "SAI_KIEU_SO", "Cột đang là chữ. Dùng dm.to_number(cot) hoặc cột __so trước khi tính."
    if re.search(r"division by zero|ZeroDivisionError", e):
        return "CHIA_0", "Mẫu số bằng 0: kỳ hoặc bộ lọc không còn dòng nào. Kiểm tra lại kỳ và tên; dữ liệu không có thì dm.unanswerable(...)."
    if re.search(r"(?:index|positional indexer).*out of|iloc|IndexError|single positional", e):
        return "BANG_RONG", "Bộ lọc ra bảng rỗng. Kiểm tra tên bằng dm.match_one, kiểm tra kỳ; dữ liệu không có thì dm.unanswerable(...)."
    if re.search(r"module 'dm'|dm has no attribute|has no attribute", e):
        return "SAI_HAM", "Hàm hoặc thuộc tính không tồn tại. Chỉ dùng các hàm dm được liệt kê, cột dùng df['tên']."
    if "ModuleNotFoundError" in e or "ImportError" in e:
        return "IMPORT", "Không import thêm: pd, np, dm, dt, re đã có sẵn."
    if "SyntaxError" in e or "IndentationError" in e:
        return "CU_PHAP", "Lỗi cú pháp: viết lại khối code ngắn hơn, ít comment."
    if "merge" in e.lower() or "You are trying to merge" in e:
        return "NOI_BANG", "Hai cột nối khác kiểu: ép cùng kiểu chuỗi bằng .astype(str) trước khi merge."
    if "ANSWER" in e:
        return "CHUA_GAN", "Gán ANSWER ở cấp ngoài cùng bằng hàm dm."
    return "KHAC", "Đọc traceback, sửa đúng dòng lỗi, giữ nguyên các phần đã đúng."
