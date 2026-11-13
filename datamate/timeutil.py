"""Kỳ thời gian theo rulebook §2. Mọi kỳ là khoảng ngày [start, end] gồm cả hai đầu.

Không sửa luật ở đây khi rulebook của bộ test khác: các hàm chỉ là cách nói chung
(tuần bắt đầu thứ Hai, tháng này tính đến hôm nay…) mà rulebook các bộ đang dùng chung.
"""
from __future__ import annotations

import calendar as _cal
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _d(x) -> dt.date:
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return pd.Timestamp(x).date()


@dataclass(frozen=True)
class Period:
    start: dt.date
    end: dt.date
    kind: str = "range"  # day | week | week_to_date | month | month_to_date | quarter | year | ytd | last_n | range | until | tet

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def mask(self, s: pd.Series) -> pd.Series:
        """Dòng có ngày (datetime) thuộc kỳ. Giờ trong ngày không ảnh hưởng."""
        s = pd.to_datetime(s, errors="coerce")
        lo = pd.Timestamp(self.start)
        hi = pd.Timestamp(self.end) + pd.Timedelta(days=1)
        return (s >= lo) & (s < hi)

    def __contains__(self, d) -> bool:
        return self.start <= _d(d) <= self.end

    def dates(self) -> list[dt.date]:
        return [self.start + dt.timedelta(days=i) for i in range(self.days)]

    # lặp được như danh sách ngày: `cot.dt.date.isin(p)`, `for d in p` (model hay viết vậy)
    def __iter__(self):
        return iter(self.dates())

    def __len__(self) -> int:
        return self.days

    def __repr__(self) -> str:
        return f"Period({self.start}..{self.end}, {self.kind})"


# ---------------------------------------------------------------- kỳ cơ bản

def day(d) -> Period:
    d = _d(d)
    return Period(d, d, "day")


def yesterday(today) -> Period:
    return day(_d(today) - dt.timedelta(days=1))


def date_range(d1, d2) -> Period:
    return Period(_d(d1), _d(d2), "range")


def until(d) -> Period:
    """Từ đầu dữ liệu đến hết ngày d (số còn lại: còn nợ, còn tồn…)."""
    return Period(dt.date(1900, 1, 1), _d(d), "until")


def last_n_days(n: int, today) -> Period:
    t = _d(today)
    return Period(t - dt.timedelta(days=n - 1), t, "last_n")


def this_week(today) -> Period:
    t = _d(today)
    return Period(t - dt.timedelta(days=t.weekday()), t, "week_to_date")


def week_of(d) -> Period:
    d = _d(d)
    mon = d - dt.timedelta(days=d.weekday())
    return Period(mon, mon + dt.timedelta(days=6), "week")


def last_week(today) -> Period:
    """Tuần trước / tuần rồi: cả tuần liền trước, thứ Hai đến Chủ nhật."""
    t = _d(today)
    mon = t - dt.timedelta(days=t.weekday() + 7)
    return Period(mon, mon + dt.timedelta(days=6), "week")


def month(year: int, m: int) -> Period:
    last = _cal.monthrange(year, m)[1]
    return Period(dt.date(year, m, 1), dt.date(year, m, last), "month")


def month_no_year(m: int, today) -> Period:
    """Tháng M không ghi năm: năm nay nếu tháng đó không sau hôm nay, ngược lại năm trước."""
    t = _d(today)
    y = t.year if dt.date(t.year, m, 1) <= t else t.year - 1
    return month(y, m)


def this_month(today) -> Period:
    t = _d(today)
    return Period(dt.date(t.year, t.month, 1), t, "month_to_date")


def last_month(today) -> Period:
    """Tháng trước (cả tháng)."""
    t = _d(today)
    y, m = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
    return month(y, m)


def prev_month_same_days(today) -> Period:
    """Kỳ so sánh của 'tháng này': tháng trước từ ngày 1 đến ngày trùng số với hôm nay
    (tháng trước không có ngày đó thì đến hết tháng trước)."""
    t = _d(today)
    y, m = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
    last = _cal.monthrange(y, m)[1]
    return Period(dt.date(y, m, 1), dt.date(y, m, min(t.day, last)), "month_to_date")


def quarter(year: int, q: int) -> Period:
    m0 = 3 * (q - 1) + 1
    return Period(dt.date(year, m0, 1), month(year, m0 + 2).end, "quarter")


def quarter_no_year(q: int, today) -> Period:
    t = _d(today)
    y = t.year if dt.date(t.year, 3 * (q - 1) + 1, 1) <= t else t.year - 1
    return quarter(y, q)


def this_quarter(today) -> Period:
    t = _d(today)
    q = (t.month - 1) // 3 + 1
    return Period(quarter(t.year, q).start, t, "quarter_to_date")


def year(y: int) -> Period:
    return Period(dt.date(y, 1, 1), dt.date(y, 12, 31), "year")


def year_to_date(today) -> Period:
    t = _d(today)
    return Period(dt.date(t.year, 1, 1), t, "ytd")


def months_ending(y: int, m: int, n: int) -> list[Period]:
    """n tháng liền nhau kết thúc ở tháng m/y, từ cũ đến mới."""
    out = []
    for _ in range(n):
        out.append(month(y, m))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return out[::-1]


def months_between(y1: int, m1: int, y2: int, m2: int) -> list[Period]:
    out = []
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        out.append(month(y, m))
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    return out


# ---------------------------------------------------------------- kỳ trước, cùng kỳ

def previous(p: Period) -> Period:
    """Kỳ liền trước dùng cho 'so với kỳ trước' (rulebook §1)."""
    if p.kind == "month":
        return last_month(p.start)
    if p.kind == "month_to_date":
        return prev_month_same_days(p.end)
    if p.kind == "quarter":
        y, q = p.start.year, (p.start.month - 1) // 3 + 1
        return quarter(y, q - 1) if q > 1 else quarter(y - 1, 4)
    if p.kind == "year":
        return year(p.start.year - 1)
    if p.kind == "ytd":
        return Period(dt.date(p.start.year - 1, 1, 1), _same_day_prev_year(p.end), "ytd")
    if p.kind in ("week", "week_to_date"):
        return Period(p.start - dt.timedelta(days=7), p.end - dt.timedelta(days=7), p.kind)
    return Period(p.start - dt.timedelta(days=p.days), p.start - dt.timedelta(days=1), p.kind)


def same_period_last_year(p: Period) -> Period:
    if p.kind == "month":
        return month(p.start.year - 1, p.start.month)
    if p.kind == "quarter":
        return quarter(p.start.year - 1, (p.start.month - 1) // 3 + 1)
    if p.kind == "year":
        return year(p.start.year - 1)
    return Period(_same_day_prev_year(p.start), _same_day_prev_year(p.end), p.kind)


def _same_day_prev_year(d: dt.date) -> dt.date:
    try:
        return d.replace(year=d.year - 1)
    except ValueError:  # 29/02
        return d.replace(year=d.year - 1, day=28)


# ---------------------------------------------------------------- ca làm việc

def shift_of(t) -> str | None:
    """'sang' (06:00–13:59), 'chieu' (14:00–21:59) hoặc None."""
    if t is None or (isinstance(t, float) and pd.isna(t)) or t is pd.NaT:
        return None
    h = t.hour
    if 6 <= h <= 13:
        return "sang"
    if 14 <= h <= 21:
        return "chieu"
    return None


def shift_mask(s: pd.Series, shift: str) -> pd.Series:
    """s: cột datetime (có giờ) hoặc cột giờ. shift: 'sang' | 'chieu'."""
    if pd.api.types.is_datetime64_any_dtype(s):
        h = s.dt.hour
    else:
        h = s.map(lambda v: getattr(v, "hour", None)).astype("float")
    if shift == "sang":
        return (h >= 6) & (h <= 13)
    if shift == "chieu":
        return (h >= 14) & (h <= 21)
    raise ValueError("shift phải là 'sang' hoặc 'chieu'")


# ---------------------------------------------------------------- lịch âm, Tết

class LunarCalendar:
    def __init__(self, path: str | Path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["lunar_leap"] = df["lunar_leap"].astype(str).str.lower().isin(["true", "1"])
        self.df = df

    def solar(self, lunar_year: int, lunar_month: int, lunar_day: int, leap: bool = False) -> dt.date:
        r = self.df[(self.df.lunar_year == lunar_year) & (self.df.lunar_month == lunar_month)
                    & (self.df.lunar_day == lunar_day) & (self.df.lunar_leap == leap)]
        if r.empty:
            raise ValueError(f"không có ngày âm {lunar_day}/{lunar_month}/{lunar_year} trong calendar.csv")
        return r.iloc[0]["date"]

    def lunar(self, d) -> tuple[int, int, int, bool]:
        r = self.df[self.df.date == _d(d)]
        if r.empty:
            raise ValueError(f"không có ngày {d} trong calendar.csv")
        r = r.iloc[0]
        return int(r.lunar_year), int(r.lunar_month), int(r.lunar_day), bool(r.lunar_leap)

    def tet(self, y: int) -> Period:
        """Tết năm Y: 23 tháng Chạp (năm âm Y−1) đến hết mùng 10 tháng Giêng (năm âm Y)."""
        return Period(self.solar(y - 1, 12, 23), self.solar(y, 1, 10), "tet")

    def before_tet(self, y: int) -> Period:
        """Cùng số ngày ngay trước 23 tháng Chạp."""
        t = self.tet(y)
        return Period(t.start - dt.timedelta(days=t.days), t.start - dt.timedelta(days=1), "range")

    def holidays(self) -> pd.DataFrame:
        return self.df[self.df.holiday.notna() & (self.df.holiday.astype(str) != "")][["date", "holiday"]]
