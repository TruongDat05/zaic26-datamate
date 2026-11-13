"""Thư viện `dm` cho code do LLM sinh ra: kỳ thời gian, làm sạch, ghép tên, xếp hạng, tạo đáp án.

Code sinh ra chạy với các biến có sẵn: `T` (dict tên bảng → DataFrame), `TODAY` (datetime.date),
`pd`, `np`, `dm`. Kết quả gán vào `ANSWER = dm.<kiểu>(...)`.
"""
from __future__ import annotations

import datetime as dt
import difflib
import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

import functools

from . import clean as _clean
from .clean import norm_col, parse_datetime, parse_qty, parse_time, strip_accents, to_datetime, to_number  # noqa: F401
from .timeutil import (Period, date_range, day, last_month, last_n_days, last_week, month, month_no_year,  # noqa: F401
                       months_between, months_ending, prev_month_same_days, previous, quarter, quarter_no_year,
                       same_period_last_year, shift_mask, shift_of, this_month, this_quarter, this_week, until,
                       week_of, year, year_to_date, yesterday)

cal = None  # LunarCalendar, sandbox gán khi chạy


def _series_aware(f):
    """Hàm một giá trị nhận luôn cả Series / Index (model hay viết `~dm.is_walkin(df['Khách'])`)."""
    @functools.wraps(f)
    def g(x, *a, **k):
        if isinstance(x, (pd.Series, pd.Index)):
            return x.map(lambda v: f(v, *a, **k))
        return f(x, *a, **k)
    return g


is_walkin = _series_aware(_clean.is_walkin)
norm_status = _series_aware(_clean.norm_status)
norm_text = _series_aware(_clean.norm_text)
norm_key = _series_aware(_clean.norm_key)
parse_number = _series_aware(_clean.parse_number)


def tet(y: int) -> Period:
    return cal.tet(y)


def before_tet(y: int) -> Period:
    return cal.before_tet(y)


# ---------------------------------------------------------------- làm tròn

def round_half_up(x, nd: int = 0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise ValueError("giá trị rỗng (NaN)")
    q = Decimal(1).scaleb(-nd)
    v = Decimal(repr(float(x))).quantize(Decimal("1e-9"))  # bỏ sai số dấu phẩy động trước khi làm tròn
    v = v.quantize(q, rounding=ROUND_HALF_UP)
    return int(v) if nd == 0 else float(v)


# ---------------------------------------------------------------- ghép tên

def match(candidates, query: str, n: int = 1, cutoff: float = 0.6):
    """Tìm tên trong `candidates` khớp `query` (không phân biệt dấu, hoa thường, viết tắt nhẹ).

    Trả list tối đa n tên gốc, tốt nhất trước. Dùng để tìm mặt hàng / khách / nhóm câu hỏi nhắc tới."""
    cands = [c for c in dict.fromkeys(pd.Series(list(candidates)).dropna().astype(str))]
    qk = norm_key(query)
    if not qk:
        return []
    keyed = [(c, norm_key(c)) for c in cands]
    exact = [c for c, k in keyed if k == qk]
    if exact:
        return exact[:n]
    qt = set(qk.split())
    scored = []
    for c, k in keyed:
        kt = set(k.split())
        if not kt:
            continue
        overlap = len(qt & kt) / len(qt | kt)
        contain = 1.0 if (qk in k or k in qk) else 0.0
        ratio = difflib.SequenceMatcher(None, qk, k).ratio()
        scored.append((max(ratio, 0.5 * overlap + 0.5 * contain, overlap), c))
    scored.sort(key=lambda t: -t[0])
    return [c for s, c in scored if s >= cutoff][:n]


def match_one(candidates, query: str, cutoff: float = 0.6):
    r = match(candidates, query, 1, cutoff)
    if not r:
        near = match(candidates, query, 5, 0.3)
        raise LookupError(f"không tìm thấy {query!r} trong dữ liệu"
                          + (f"; tên gần nhất: {near}" if near else "; có thể dữ liệu không có thứ này (unanswerable)"))
    return r[0]


def eq(series: pd.Series, value) -> pd.Series:
    """So khớp cột chữ với một giá trị, bỏ qua hoa thường / dấu câu / khoảng trắng."""
    v = norm_text(value)
    return series.map(norm_text) == v


# ---------------------------------------------------------------- xếp hạng

def rank(df: pd.DataFrame, by: str, ascending: bool = False, code: str | None = None, k: int | None = None):
    """Sắp xếp theo `by` (mặc định giảm dần), bằng nhau thì theo `code` tăng dần (không có mã thì theo cột đầu)."""
    tie = code or df.columns[0]
    out = df.sort_values([by, tie], ascending=[ascending, True], kind="mergesort")
    return out.head(k) if k else out


# ---------------------------------------------------------------- đáp án

class Answer(dict):
    pass


def _num(x):
    if isinstance(x, Answer) and x.get("type") in ("number", "percent"):
        return x["answer"]  # dm.money(dm.money(...)) hay số đã bọc đáp án
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (pd.Series, pd.DataFrame, list, tuple, dict)):
        raise TypeError(f"cần một số, nhận {type(x).__name__}")
    return x


def money(x) -> Answer:
    """Số tiền (VND): làm tròn về đồng, nửa đồng làm tròn lên."""
    return Answer(type="number", answer=round_half_up(_num(x), 0))


def count(x) -> Answer:
    """Số đếm (đơn, khách, ly, bao…)."""
    v = _num(x)
    if abs(v - round(v)) > 1e-6:
        return Answer(type="number", answer=round_half_up(v, 2))
    return Answer(type="number", answer=int(round(v)))


def number(x, nd: int | None = None) -> Answer:
    """Số khác. nd=None: số nguyên giữ nguyên, số lẻ làm tròn 2 chữ số."""
    v = _num(x)
    if nd is not None:
        return Answer(type="number", answer=round_half_up(v, nd))
    if abs(v - round(v)) < 1e-9:
        return Answer(type="number", answer=int(round(v)))
    return Answer(type="number", answer=round_half_up(v, 2))


def percent(x) -> Answer:
    """x là SỐ PHẦN TRĂM (12.4 nghĩa là 12,4 %), làm tròn 2 chữ số."""
    return Answer(type="percent", answer=round_half_up(_num(x), 2))


def pct_change(new, old) -> Answer:
    """Phần trăm thay đổi so với kỳ so (old). Giảm ra số âm."""
    new, old = _num(new), _num(old)
    if old == 0:
        raise ZeroDivisionError("kỳ so bằng 0, không tính được phần trăm thay đổi")
    return percent((new - old) / old * 100)


def share(part, total) -> Answer:
    """Tỉ trọng / tỉ lệ phần trăm part / total."""
    part, total = _num(part), _num(total)
    if total == 0:
        raise ZeroDivisionError("mẫu số bằng 0")
    return percent(part / total * 100)


def date(d, *month_day):
    """Đáp án là một ngày 'YYYY-MM-DD'. Gọi kiểu dm.date(y, m, d) thì trả datetime.date (model hay dùng như hàm tạo ngày)."""
    if month_day:
        return dt.date(d, *month_day)
    return Answer(type="date", answer=pd.Timestamp(d).strftime("%Y-%m-%d"))


def month_answer(d) -> Answer:
    """Đáp án là một tháng: 'YYYY-MM'."""
    if isinstance(d, Period):
        d = d.start
    return Answer(type="date", answer=pd.Timestamp(d).strftime("%Y-%m"))


def boolean(x) -> Answer:
    return Answer(type="boolean", answer=bool(x))


def string(x) -> Answer:
    if isinstance(x, (list, tuple, pd.Series)):
        raise TypeError("string cần một giá trị, không phải danh sách")
    return Answer(type="string", answer=str(x))


def list_(items) -> Answer:
    """Danh sách có thứ tự (tên hoặc số). Một phần tử vẫn là danh sách."""
    out = []
    for v in list(items):
        v = _num(v)
        if isinstance(v, float) and abs(v - round(v)) < 1e-9:
            v = int(round(v))
        elif isinstance(v, float):
            v = round_half_up(v, 2)
        out.append(v)
    return Answer(type="list", answer=out)


def table(columns, rows) -> Answer:
    """Bảng: cột đầu là tên đối tượng, các cột sau là số đo theo đúng thứ tự câu hỏi."""
    clean_rows = []
    for r in rows:
        rr = []
        for v in list(r):
            v = _num(v)
            if isinstance(v, float) and abs(v - round(v)) < 1e-9:
                v = int(round(v))
            elif isinstance(v, float):
                v = round_half_up(v, 2)
            rr.append(v)
        clean_rows.append(rr)
    return Answer(type="table", answer={"columns": list(columns), "rows": clean_rows})


def table_from_df(df: pd.DataFrame) -> Answer:
    return table(list(df.columns), df.values.tolist())


def n_orders(df: pd.DataFrame, id_col: str) -> int:
    """Số đơn: mỗi mã đơn tính một lần, dù đơn có nhiều dòng hàng. Không dùng len(df)."""
    return int(df[id_col].nunique())


def n_distinct(df: pd.DataFrame, cols) -> int:
    """Số tổ hợp khác nhau, vd đợt giao: dm.n_distinct(df, ['Mã đơn', 'Lượt giao'])."""
    return int(df[list(cols)].dropna().drop_duplicates().shape[0])


def cancel_rate(df: pd.DataFrame, id_col: str, status_col: str) -> Answer:
    """Tỉ lệ đơn huỷ: df là mọi dòng của các đơn đặt trong kỳ (mọi trạng thái); đếm theo đơn, không theo dòng."""
    total = n_orders(df, id_col)
    if total == 0:
        raise ZeroDivisionError("kỳ này không có đơn nào")
    return share(n_orders(df[norm_status(df[status_col]) == "huy"], id_col), total)


def unanswerable(reason: str = "") -> Answer:
    return Answer(type="unanswerable", reason=str(reason))


def abstain(reason: str = "") -> Answer:
    return Answer(type="abstain", reason=str(reason))


# ---------------------------------------------------------------- tiện ích dữ liệu

def data_span(*date_series) -> Period:
    """Khoảng ngày có dữ liệu (từ ngày sớm nhất đến muộn nhất của các cột ngày đưa vào)."""
    s = pd.concat([pd.to_datetime(x, errors="coerce") for x in date_series]).dropna()
    return Period(s.min().date(), s.max().date(), "range")


def no_data(period: Period, *date_series) -> bool:
    """True khi kỳ nằm HOÀN TOÀN ngoài khoảng ngày có dữ liệu → câu hỏi không trả lời được (rulebook §7)."""
    span = data_span(*date_series)
    return period.end < span.start or period.start > span.end
