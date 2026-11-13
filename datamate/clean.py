"""Hàm làm sạch giá trị thô: tiền, số lượng, ngày giờ, chuỗi.

Dùng ở hai nơi: loader (tự chuẩn hoá cột khi chắc chắn) và code do LLM sinh ra (qua `dm`).
"""
from __future__ import annotations

import datetime as dt
import functools
import math
import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------- chuỗi

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


@functools.lru_cache(maxsize=500_000)
def _norm_text_str(s: str) -> str:
    s = nfc(s).lower()
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return " ".join(s.split())


def norm_text(s) -> str:
    """Chuẩn hoá để so sánh: NFC, chữ thường, ký tự lạ → khoảng trắng, gộp khoảng trắng.

    Giữ dấu tiếng Việt (giống luật chấm `string`)."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return _norm_text_str(str(s))


def norm_key(s) -> str:
    """Chuẩn hoá mạnh hơn để ghép tên: bỏ cả dấu tiếng Việt."""
    return _strip_cached(norm_text(s))


@functools.lru_cache(maxsize=500_000)
def _strip_cached(s: str) -> str:
    return strip_accents(s)


def norm_col(name) -> str:
    """Tên cột → dạng so khớp: không dấu, chữ thường, '_' thay khoảng trắng."""
    return norm_key(name).replace(" ", "_")


# ---------------------------------------------------------------- số

_NUM_RE = re.compile(r"^[+-]?[\d.,]+$")


def parse_number(x):
    """'38,000' / '45,000.0000' / '1.234.567' / '12,5' / '38.000đ' / 1200 → float. Không đọc được → NaN."""
    if x is None:
        return math.nan
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return _parse_number_str(str(x))


@functools.lru_cache(maxsize=500_000)
def _parse_number_str(s: str) -> float:
    s = nfc(s).strip().lower()
    if not s or s in {"-", "nan", "none", "null"}:
        return math.nan
    s = re.sub(r"(vnd|vnđ|đồng|đ|d)$", "", s).strip()
    s = s.replace(" ", "").replace(" ", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    short = _money_shorthand(s)
    if short is not None:
        return -short if neg else short
    if not _NUM_RE.match(s):
        return math.nan
    if "," in s and "." in s:
        # dấu xuất hiện sau cùng là dấu thập phân
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        # '38,000' hoặc '1,234,567' → phân cách nghìn; '12,5' → thập phân
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and parts[0] not in {"0", "-0"}):
            s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return math.nan
    return -v if neg else v


_QTY_RE = re.compile(r"^\s*([+-]?[\d.,]+)\s*(.*)$")
_TR_RE = re.compile(r"^(\d+)tr(\d{1,3})$")  # '1tr270' = 1.270.000
_UNIT_RE = re.compile(r"^(\d+(?:[.,]\d+)?)(k|nghìn|nghin|ngàn|ngan|tr|triệu|trieu)$")
_UNIT = {"k": 1e3, "nghìn": 1e3, "nghin": 1e3, "ngàn": 1e3, "ngan": 1e3, "tr": 1e6, "triệu": 1e6, "trieu": 1e6}


def _money_shorthand(s: str):
    """Tiền ghi tắt trong sổ tay: '45k' → 45000, '1tr270' → 1270000, '1,5tr' → 1500000. Không khớp → None."""
    m = _TR_RE.match(s)
    if m:
        return float(int(m.group(1)) * 1_000_000 + int(m.group(2).ljust(3, "0")) * 1_000)
    m = _UNIT_RE.match(s)
    if m:
        return round(float(m.group(1).replace(",", ".")) * _UNIT[m.group(2)], 2)
    return None


def parse_qty(x):
    """'5 chai' → (5.0, 'chai'); '1 bao 25kg' → (1.0, 'bao 25kg'); 3 → (3.0, ''). Không đọc được → (NaN, '')."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return (math.nan, "")
    if isinstance(x, (int, float)):
        return (float(x), "")
    return _parse_qty_str(str(x))


@functools.lru_cache(maxsize=200_000)
def _parse_qty_str(s: str):
    m = _QTY_RE.match(nfc(s))
    if not m:
        return (math.nan, "")
    return (parse_number(m.group(1)), m.group(2).strip())


def to_number(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype("float64")


# ---------------------------------------------------------------- ngày giờ

_EXCEL_EPOCH = dt.datetime(1899, 12, 30)
_VI_MONTH = re.compile(r"^\s*(\d{1,2})\s*(?:thg|tháng|th)\s*(\d{1,2})\s*,?\s*(\d{4})(.*)$", re.I)
_DMY = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})(?:[ T,]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\s*$")
_YMD = re.compile(r"^\s*(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?:[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?)?\s*$")
_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")


def parse_datetime(x, year: int | None = None):
    """Đọc ngày giờ theo kiểu Việt Nam (ngày trước tháng). Trả pd.Timestamp hoặc NaT.

    Nhận: datetime, số serial Excel, '01/03/2025', '1/3/2025 14:05', '2025-03-01',
    '1 thg 3 2025', '1 tháng 3, 2025'. `year`: năm dùng khi chuỗi chỉ có ngày/tháng."""
    if x is None:
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, dt.datetime):
        return pd.Timestamp(x)
    if isinstance(x, dt.date):
        return pd.Timestamp(x)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if math.isnan(x):
            return pd.NaT
        if 20000 < x < 80000:  # serial Excel (1954–2119)
            return pd.Timestamp(_EXCEL_EPOCH + dt.timedelta(days=float(x)))
        return pd.NaT
    return _parse_dt_str(str(x), year)


@functools.lru_cache(maxsize=500_000)
def _parse_dt_str(s: str, year):
    s = nfc(s).strip()
    if not s:
        return pd.NaT
    m = _DMY.match(s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        hh, mm, ss = (int(m.group(i) or 0) for i in (4, 5, 6))
        return _mk(y, mo, d, hh, mm, ss)
    m = _YMD.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh, mm, ss = (int(m.group(i) or 0) for i in (4, 5, 6))
        return _mk(y, mo, d, hh, mm, ss)
    m = _VI_MONTH.match(s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        rest = _TIME.match(m.group(4).strip(" ,")) if m.group(4).strip(" ,") else None
        hh, mm, ss = (int(rest.group(i) or 0) for i in (1, 2, 3)) if rest else (0, 0, 0)
        return _mk(y, mo, d, hh, mm, ss)
    m = re.match(r"^\s*(\d{1,2})[/\-.](\d{1,2})\s*$", s)
    if m and year:
        return _mk(year, int(m.group(2)), int(m.group(1)))
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        return parse_datetime(float(s))
    return pd.NaT


def _mk(y, mo, d, hh=0, mm=0, ss=0):
    try:
        return pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mm, second=ss)
    except ValueError:
        return pd.NaT


def to_datetime(series: pd.Series, year: int | None = None) -> pd.Series:
    return pd.to_datetime(series.map(lambda v: parse_datetime(v, year)), errors="coerce")


def parse_time(x):
    """'14:05', datetime, time, số thập phân Excel → dt.time hoặc None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, dt.time):
        return x
    if isinstance(x, (dt.datetime, pd.Timestamp)):
        return x.time()
    if isinstance(x, float) and 0 <= x < 1:
        secs = round(x * 86400)
        return dt.time(secs // 3600 % 24, secs // 60 % 60, secs % 60)
    s = str(x).strip()
    m = _TIME.match(s)
    if m:
        return dt.time(int(m.group(1)) % 24, int(m.group(2)), int(m.group(3) or 0))
    ts = parse_datetime(s)
    return None if pd.isna(ts) else ts.time()


# ---------------------------------------------------------------- nhận dạng giá trị đặc biệt

_WALKIN = {"kl", "khach le", "le", "khach vang lai", "vang lai", "khach", "khachle", "kh le", "guest", "walk in"}


def is_walkin(name) -> bool:
    """Khách lẻ / vãng lai / để trống."""
    k = norm_key(name)
    return k == "" or k in _WALKIN or k.startswith("khach le")


_STATUS = {
    "huy": ["huy", "da huy", "cancel", "cancelled", "canceled", "bi huy", "khach huy", "void", "huy don",
            "shop huy", "khong giao duoc", "giao that bai", "giao khong thanh cong", "that bai", "failed",
            "khach bom hang", "bom hang"],
    "tra_hang": ["tra hang", "da tra hang", "tra", "hoan tra", "return", "returned", "refund", "khach tra", "tra lai",
                 "hang tra ve", "tra ve", "khach tra hang", "hoan hang", "doi tra", "returns"],
    "dang_giao": ["dang giao", "dang van chuyen", "shipping", "delivering", "cho giao", "dang ship", "in transit",
                  "dang giao hang", "dang chuyen", "dang xu ly", "cho xu ly", "processing", "pending",
                  "cho xac nhan", "cho lay hang", "dang dong goi", "dang giao dich", "dang gui"],
    "hoan_thanh": ["hoan thanh", "da hoan thanh", "xong", "ok", "done", "completed", "complete", "thanh cong",
                   "da giao", "da thanh toan", "paid", "success", "finished", "da xong", "da giao hang",
                   "giao thanh cong", "hoan tat", "da nhan", "delivered", "da lay", "khach da lay"],
}


def norm_status(x) -> str:
    """Trạng thái đơn → một trong: hoan_thanh, huy, tra_hang, dang_giao, '' (không rõ)."""
    k = norm_key(x)
    if not k:
        return ""
    for code, words in _STATUS.items():
        if k in words:
            return code
    for code in ("huy", "tra_hang", "dang_giao", "hoan_thanh"):
        if any(w in k for w in _STATUS[code] if len(w) > 3):
            return code
    return ""
