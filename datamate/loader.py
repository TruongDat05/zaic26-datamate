"""Đọc và làm sạch dữ liệu của một doanh nghiệp (data/<split>/data/<business_id>/).

Mỗi file / sheet / bảng sqlite thành một DataFrame. Những việc làm tự động, đều theo rulebook §8
hoặc là lỗi xuất file hiển nhiên:
- tìm dòng tiêu đề, bỏ dòng tên báo cáo và tiêu đề lặp lại giữa sheet, bỏ dòng tổng cộng;
- gộp các sheet / file cùng bộ cột (tách theo tháng, theo năm) thành một bảng, giữ cột `_nguon`;
- bỏ khoảng trắng thừa, ô rỗng → NaN;
- cột số ghi dạng chuỗi ('38,000', '45,000.0000') → số; cột ngày → datetime (ngày trước tháng);
- cột số lượng kiểu '5 chai' → thêm cột `<tên>__so` (số) và `<tên>__dv` (đơn vị);
- ô trống ở các dòng sau của cùng một đơn → lấy theo dòng đầu đơn;
- hai dòng giống hệt nhau của cùng một đơn → giữ một;
- cột tên (hàng, khách, nhóm): các cách viết của cùng một tên → một cách viết.
Mọi thay đổi được ghi vào `notes` để đưa vào prompt.
"""
from __future__ import annotations

import csv
import io
import math
import pickle
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean

ID_HINTS = ("ma_don", "ma_hoa_don", "ma_hd", "so_hd", "don_so", "so_don", "order", "hoa_don", "ma_dh",
            "so_hoa_don", "invoice", "mahoadon", "madonhang", "ma_phieu", "so_phieu", "madon", "mahd", "bill", "so_ct", "so_chung_tu",
            "ma_giao_dich", "giao_dich", "ma_gd")
# tên cột mã đơn quá ngắn, chỉ nhận khi trùng nguyên văn (không tìm chuỗi con): sổ tay ghi 'HĐ', 'STT'
ID_EXACT = ("hd", "stt", "so_tt", "don")
LINE_HINTS = ("ten_hang", "ten_san_pham", "mat_hang", "san_pham", "noi_dung", "item", "mon", "ma_hang",
              "hang_hoa", "ten_sp", "mahanghoa", "ten_thuoc", "product")
DATE_HINTS = ("ngay", "date", "thoi_gian", "han", "luc", "time", "sinh_nhat", "dang_ky", "thg")
TEXT_HINTS = ("sdt", "so_dien_thoai", "dien_thoai", "phone", "ma_", "cccd", "mst", "barcode", "ma_vach", "ean",
              "bien_so", "so_lo", "ma lo", "id")


@dataclass
class Business:
    business_id: str
    path: Path
    readme: str
    tables: dict[str, pd.DataFrame]
    notes: dict[str, list[str]] = field(default_factory=dict)

    def profile(self, max_examples: int = 4, max_levels: int = 25) -> str:
        return profile_tables(self.tables, self.notes, max_examples, max_levels)


# ---------------------------------------------------------------- đọc file thô → list[list]

def _read_csv_rows(path: Path) -> list[list]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
        delim = dialect.delimiter
    except csv.Error:
        delim = ","
    return [row for row in csv.reader(io.StringIO(text), delimiter=delim)]


def _read_excel_sheets(path: Path) -> list[tuple[str, list[list]]]:
    """calamine (Rust) nhanh hơn openpyxl hàng chục lần; lỗi thì quay về openpyxl / xlrd."""
    try:
        from python_calamine import CalamineWorkbook

        wb = CalamineWorkbook.from_path(str(path))
        return [(n, wb.get_sheet_by_name(n).to_python()) for n in wb.sheet_names]
    except Exception:  # noqa: BLE001
        if path.suffix.lower() == ".xls":
            return _read_xls_sheets(path)
        return _read_xlsx_sheets(path)


def _read_xlsx_sheets(path: Path) -> list[tuple[str, list[list]]]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append((ws.title, [list(r) for r in ws.iter_rows(values_only=True)]))
    wb.close()
    return out


def _read_xls_sheets(path: Path) -> list[tuple[str, list[list]]]:
    import xlrd

    book = xlrd.open_workbook(path)
    out = []
    for sh in book.sheets():
        rows = []
        for i in range(sh.nrows):
            row = []
            for c in sh.row(i):
                if c.ctype == xlrd.XL_CELL_DATE:
                    row.append(xlrd.xldate.xldate_as_datetime(c.value, book.datemode))
                elif c.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                    row.append(None)
                else:
                    row.append(c.value)
            rows.append(row)
        out.append((sh.name, rows))
    return out


# ---------------------------------------------------------------- rows → DataFrame

def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def _row_stats(rows: list[list]) -> tuple[list[int], list, int, bool]:
    """Một lượt quét: số ô không rỗng, giá trị không rỗng đầu tiên, độ rộng, và mọi dòng cùng độ rộng."""
    counts, firsts = [], []
    width, even, seen = 0, True, False
    for r in rows:
        lr = len(r)
        if not seen:
            seen = True
            width = lr
        elif lr != width:
            even = False
            if lr > width:
                width = lr
        n, first = 0, None
        for v in r:
            if v is None:
                continue
            if isinstance(v, str):
                if v.strip():
                    n += 1
                    if first is None:
                        first = v
            elif isinstance(v, float):
                if not math.isnan(v):
                    n += 1
                    if first is None:
                        first = v
            else:
                n += 1
                if first is None:
                    first = v
        counts.append(n)
        firsts.append(first)
    return counts, firsts, width, even


def _is_header(row, width) -> bool:
    vals = [v for v in row if not _empty(v)]
    if len(vals) < max(2, math.ceil(0.6 * width)):
        return False
    strs = [v for v in vals if isinstance(v, str)]
    if len(strs) < 0.8 * len(vals):
        return False
    # tiêu đề không chứa giá trị kiểu số tiền hay ngày
    return sum(not math.isnan(clean.parse_number(v)) for v in strs) <= 1


def _rows_to_frame(rows: list[list], notes: list[str]) -> pd.DataFrame | None:
    counts, firsts, width, even = _row_stats(rows)
    end = len(rows)
    while end and counts[end - 1] == 0:
        end -= 1
    if end != len(rows):
        rows = rows[:end]
        counts = counts[:end]
        firsts = firsts[:end]
    if not rows:
        return None
    # độ rộng thực = số cột có dữ liệu
    if even:
        used = [j for j in range(width) if any(not _empty(r[j]) for r in rows)]
    else:
        used = [j for j in range(width) if any(j < len(r) and not _empty(r[j]) for r in rows)]
    if not used:
        return None
    if not even:
        rows = [[r[j] if j < len(r) else None for j in used] for r in rows]
    elif len(used) != width:
        rows = [[r[j] for j in used] for r in rows]
    width = len(used)

    hi = next((i for i, r in enumerate(rows[:30]) if _is_header(r, width)), None)
    if hi is None:
        return None
    if hi > 0:
        notes.append(f"bỏ {hi} dòng tên báo cáo ở đầu sheet")
    header = [str(v).strip() if not _empty(v) else f"cot_{j}" for j, v in enumerate(rows[hi])]
    header = _dedupe_names(header)
    hkey = [clean.norm_key(v) for v in rows[hi]]

    body, dropped_hdr, dropped_title, dropped_total = [], 0, 0, 0
    fk_cache: dict[str, str] = {}
    for r, n, first in zip(rows[hi + 1:], counts[hi + 1:], firsts[hi + 1:]):
        if n == 0:
            continue
        if clean.norm_key(r[0]) == hkey[0] and [clean.norm_key(v) for v in r] == hkey:
            dropped_hdr += 1
            continue
        fk = ""
        if isinstance(first, str):
            fk = fk_cache.get(first)
            if fk is None:
                fk = fk_cache[first] = clean.norm_key(first)
        # 'Tổng tháng', 'Tổng ngày', 'Cộng quý'…: ô đầu bắt đầu bằng 'tổng'/'cộng' và dòng gần như trống
        short_total = n <= 3 and fk.split(" ", 1)[0] in ("tong", "cong")
        if fk.startswith(("tong cong", "tong so", "cong ", "total", "grand total")) or fk in {"tong", "cong"} or short_total:
            dropped_total += 1
            continue
        if n <= 2 and width >= 5 and all(isinstance(v, str) for v in r if not _empty(v)) and (
                ":" in " ".join(str(v) for v in r if not _empty(v))
                or (isinstance(first, str) and first.strip().isupper() and len(first.strip()) > 6)):
            dropped_title += 1
            continue
        body.append(r)
    if dropped_hdr:
        notes.append(f"bỏ {dropped_hdr} dòng tiêu đề lặp lại giữa sheet")
    if dropped_title:
        notes.append(f"bỏ {dropped_title} dòng tên báo cáo / kỳ báo cáo giữa sheet")
    if dropped_total:
        notes.append(f"bỏ {dropped_total} dòng tổng cộng")
    df = pd.DataFrame(body, columns=header)
    return df


def _dedupe_names(names):
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


# ---------------------------------------------------------------- làm sạch cột

_NO_STRING_KINDS = frozenset({
    "integer", "floating", "decimal", "date", "datetime", "datetime64", "time",
    "timedelta", "timedelta64", "boolean", "bytes", "empty",
})
_NUMERIC_KINDS = frozenset({"integer", "floating", "decimal", "boolean"})


def _strip_cell(v):
    if isinstance(v, str):
        v = clean.nfc(v).strip()
        return None if v == "" else v
    return v


def _strip_str(v):
    """NFC + strip, one pass; return `v` untouched when it is already clean.

    ASCII has no canonical compositions, so NFC is the identity there and a
    byte-range boundary check replaces `strip`. For non-ASCII the whole
    normalize+strip is only paid when `is_normalized` or a boundary space says
    the cell would change.
    """
    if not isinstance(v, str) or not v:
        return None
    if v.isascii():
        if v[0] > " " and v[-1] > " ":
            return v
        return v.strip() or None
    if unicodedata.is_normalized("NFC", v) and not v[0].isspace() and not v[-1].isspace():
        return v
    return unicodedata.normalize("NFC", v).strip() or None


def _strip_cells(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for col in df.columns:
        s = df[col]
        kind = pd.api.types.infer_dtype(s, skipna=True)
        if kind == "string":
            vals = [_strip_str(v) for v in s.tolist()]
            cols[col] = pd.Series(vals, index=s.index, dtype=object)
        elif kind in _NO_STRING_KINDS:
            cols[col] = s
        else:
            cols[col] = s.map(_strip_cell)
    return pd.DataFrame(cols, index=df.index).astype(object).where(df.notna(), None)


def _is_texty(col_key: str) -> bool:
    return any(col_key == h.rstrip("_") or col_key.startswith(h) or col_key.endswith("_" + h.rstrip("_"))
               for h in TEXT_HINTS)


def _leading_zero_skip(strs: pd.Series, n_all: int) -> bool:
    """Phần lớn giá trị là mã có số 0 đầu, dạng '0123456789' (bỏ hết dấu cách) hoặc '0012' (sau strip).

    `str.isdecimal()` nhận đúng bộ ký tự số Unicode mà `\\d` khớp, nên hai kiểm tra này tương đương
    `fullmatch(r"0\\d{7,}")` và `fullmatch(r"0\\d+")`.
    """
    long_hits = short_hits = 0
    for v in strs:
        t = v.strip()
        if len(t) > 1 and t[0] == "0" and t.isdecimal():
            short_hits += 1
        u = v.replace(" ", "") if " " in v else v
        if len(u) > 7 and u[0] == "0" and u.isdecimal():
            long_hits += 1
    return long_hits * 2 > len(strs) or short_hits * 2 > n_all


def _infer_columns(df: pd.DataFrame, year_hint: int | None, notes: list[str], dates: bool = True,
                   others: bool = True) -> pd.DataFrame:
    """dates: chỉ đổi cột ngày (cần năm của sheet); others: cột số và số lượng kèm đơn vị."""
    df = df.copy()
    for col in list(df.columns):
        s = df[col]
        nn = s.dropna()
        if nn.empty or col.startswith("_"):
            continue
        key = clean.norm_col(col)
        kind = pd.api.types.infer_dtype(nn, skipna=True)
        if kind == "string":
            is_dt_obj = 0.0
            strs = nn
        elif kind in _NO_STRING_KINDS:
            is_dt_obj = 1.0 if kind in ("date", "datetime", "datetime64") else 0.0
            strs = nn.iloc[0:0]
        else:
            is_dt_obj = nn.map(lambda v: hasattr(v, "year") and hasattr(v, "month")).mean()
            strs = nn[nn.map(lambda v: isinstance(v, str))]

        # --- ngày giờ
        date_named = any(h in key for h in DATE_HINTS)
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        is_date = is_dt_obj > 0.95
        ratio = None
        if not is_date and date_named:
            ratio = _date_ratio(nn, year_hint)
            is_date = ratio > 0.9
        if not is_date and len(strs) == len(nn) and _looks_date_str(strs):
            if ratio is None:
                ratio = _date_ratio(nn, year_hint)
            is_date = ratio > 0.97
        if is_date:
            if _all_times(nn, kind) or not dates:
                continue
            df[col] = clean.to_datetime(s, year_hint)
            continue
        if not others:
            continue

        # --- số
        if _is_texty(key) and len(strs) > 0:
            continue
        if kind in _NUMERIC_KINDS:
            # đã là số (calamine trả int/float): ép kiểu một lượt, khỏi gọi parse_number từng ô
            nums = nn.astype("float64")
        elif kind == "string":
            # chuỗi lặp nhiều: tra bảng theo giá trị khác nhau thay vì gọi hàm từng ô
            lut = {v: clean.parse_number(v) for v in pd.unique(nn)}
            nums = nn.map(lut)
        else:
            nums = nn.map(clean.parse_number)
        ok = nums.notna().mean()
        # Mã có số 0 đầu ('0012') ở cột tên không gợi ý là mã: đổi sang số sẽ mất số 0. Chỉ cần
        # kiểm tra khi cột thực sự đọc được thành số, nếu không thì kết quả gate không đổi hành vi.
        if len(strs) and ok > 0.2 and _leading_zero_skip(strs, len(nn)):
            continue
        if ok > 0.97:
            vals = nums.reindex(s.index).astype("float64")
            df[col] = vals
            if (vals.dropna() % 1 == 0).all():
                df[col] = vals.astype("Int64")
            continue

        # --- số lượng kèm đơn vị: '5 chai'
        if len(strs) and ok > 0.2:
            q = nn.map(clean.parse_qty)
            if q.map(lambda t: not math.isnan(t[0])).mean() > 0.97:
                parsed = s.map(clean.parse_qty)
                df[col + "__so"] = parsed.map(lambda t: t[0]).astype("float64")
                df[col + "__dv"] = parsed.map(lambda t: t[1] or None)
                notes.append(f"cột `{col}` có dạng '5 chai': đã tách thành `{col}__so` (số) và `{col}__dv` "
                             f"(đơn vị ghi kèm, trống nếu chỉ ghi số)")
    return df


def _date_ratio(nn: pd.Series, year_hint) -> float:
    sample = nn.head(400)
    return sample.map(lambda v: not pd.isna(clean.parse_datetime(v, year_hint))).mean()


def _looks_date_str(strs: pd.Series) -> bool:
    return strs.head(200).str.contains(r"\d{1,4}[/\-.]\d{1,2}|thg|tháng", regex=True, na=False).mean() > 0.95


def _all_times(nn: pd.Series, kind: str) -> bool:
    if kind == "time":
        return True
    if kind in _NO_STRING_KINDS:
        return False
    import datetime as dt
    return nn.map(lambda v: isinstance(v, dt.time)).mean() > 0.95


def _find_col(cols, hints):
    for c in cols:
        k = clean.norm_col(c)
        if any(h in k for h in hints):
            return c
    return None


def _id_column(df: pd.DataFrame):
    cols = [c for c in df.columns if not c.startswith("_")]
    idc = _find_col(cols, ID_HINTS) or next((c for c in cols if clean.norm_col(c) in ID_EXACT), None)
    if idc is not None:
        return idc
    # sổ phụ không có cột mã, nối về sổ bán bằng ghi chú 'STT 34' ở mọi dòng
    for c in cols:
        v = df[c].dropna()
        if len(v) > 10 and v.astype(str).str.fullmatch(r"(?i)stt\s*\d+").mean() > 0.9 and v.nunique() < len(v):
            return c
    return None


def _is_line_value(col: pd.Series) -> bool:
    """Cột số mỗi dòng một ý nghĩa (tiền, số lượng): điền xuống sẽ làm cộng dồn sai."""
    return pd.api.types.is_numeric_dtype(col)


_TOTAL_CELL = re.compile(r"^\s*(tong|cong)\b")


def _drop_summary_rows(df: pd.DataFrame, idc: str, notes: list[str]) -> pd.DataFrame:
    """Dòng cộng theo ngày / tháng giữa sổ: không có mã đơn, không có ngày, không có hàng, chỉ có số tiền.

    Có shop ghi nhãn ở cột bất kỳ ('Cộng ngày 01/01/2025' ở cột khách), có shop không ghi nhãn. Giữ lại thì bị điền
    ngày và mã của đơn ngay trên, doanh thu kỳ gấp đôi."""
    blank = df[idc].isna()
    if not blank.any():
        return df
    cols = [c for c in df.columns if not c.startswith("_")]
    dates = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    linec = _find_col([c for c in cols if c != idc], LINE_HINTS) or next(
        (c for c in cols if clean.norm_col(c) in ("hang", "sp", "mon", "ten", "ten_mon")), None)
    no_date = df[dates].isna().all(axis=1) if dates else pd.Series(True, index=df.index)
    no_line = df[linec].isna() if linec is not None else pd.Series(True, index=df.index)
    labelled = pd.Series(False, index=df.index)
    for c in cols:
        if c != idc and not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
            labelled |= df[c].map(lambda v: isinstance(v, str) and bool(_TOTAL_CELL.match(clean.norm_key(v))))
    drop = blank & ((no_date & no_line) | labelled)
    if drop.any():
        notes.append(f"bỏ {int(drop.sum())} dòng cộng theo ngày/tháng giữa sổ (không có mã đơn, không có hàng)")
        df = df[~drop].reset_index(drop=True)
    return df


def _group_fill_and_dedupe(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    idc = _id_column(df)
    if idc is None:
        return df
    df = df.copy()
    df = _drop_summary_rows(df, idc, notes)
    blank = df[idc].isna()
    if 0 < blank.mean() < 0.95 and not blank.iloc[0]:
        cont = df[blank]
        order_cols = [c for c in df.columns if not c.startswith("_") and cont[c].isna().mean() >= 0.9
                      and (c == idc or not _is_line_value(df[c]))]
        if idc not in order_cols:
            order_cols.append(idc)
        # lấy đúng dòng đầu của đơn: ffill cả cột sẽ kéo ghi chú 'khách trả lại' của đơn trước sang đơn sau không có ghi chú
        group = (~blank).cumsum()
        filled = df.loc[~blank, order_cols].set_axis(group[~blank]).reindex(group.values)
        filled.index = df.index
        n = int(blank.sum())
        df.loc[blank, order_cols] = filled.loc[blank, order_cols]
        notes.append(f"{n} dòng sau của cùng một đơn để trống cột {', '.join('`'+c+'`' for c in order_cols)}: "
                     f"đã điền theo dòng đầu đơn (rulebook §8)")
    # có mã ở mọi dòng nhưng dòng phụ (Voucher, Phí ship, kỳ góp…) để trống ngày, khách: lấy theo dòng khác của cùng đơn
    if df[idc].notna().any() and df[idc].nunique() < len(df):
        first = df.groupby(idc, sort=False)
        fill_cols = []
        for c in df.columns:
            if c.startswith("_") or c == idc or _is_line_value(df[c]) or not 0 < df[c].isna().mean() < 1:
                continue
            nun = first[c].nunique()
            # chỉ cột cấp đơn: mỗi đơn đúng một giá trị (hàng, đơn vị… khác nhau trong đơn thì không điền)
            if (nun > 1).mean() < 0.005 and (nun == 1).mean() > 0.3:
                fill_cols.append(c)
        if fill_cols:
            n = int(df[fill_cols].isna().any(axis=1).sum())
            df[fill_cols] = first[fill_cols].transform(lambda x: x.ffill().bfill())
            notes.append(f"{n} dòng phụ của đơn để trống {', '.join('`'+c+'`' for c in fill_cols)}: "
                         f"đã điền theo dòng khác cùng đơn (rulebook §8)")
    # không đòi có cột tên hàng: 'Chi tiết hàng' của biz_124 lọt qua nên 1.188 dòng trùng làm doanh thu lệch so với đáp án
    data_cols = [c for c in df.columns if not c.startswith("_")]
    dup = df.duplicated(subset=data_cols, keep="first")
    if dup.any():
        notes.append(f"bỏ {int(dup.sum())} dòng giống hệt dòng khác của cùng đơn (lỗi xuất file, rulebook §8)")
        df = df[~dup].reset_index(drop=True)
    return df


# ---------------------------------------------------------------- gom sheet/file cùng bộ cột

_PERIOD_SHEET = re.compile(r"^((T|Thang|Tháng|Th|Thg|Q|Quy|Quý)[ _\-.]?\d{1,2}([ _\-/.]\d{2,4})?|\d{1,2}[_\-/.]\d{4}|\d{4}"
                           r"|Sheet\s?\d*|Trang\s?\d+|Page\s?\d+|Data|Du lieu|Dữ liệu)$", re.I)


def _base_name(stem: str) -> str:
    s = re.sub(r"[_\- ]?(T|Thang|Th|thg)?\d{1,2}[_\-.]\d{4}$", "", stem, flags=re.I)
    s = re.sub(r"[_\- ]?\d{4}$", "", s)
    s = re.sub(r"[_\- ]?\d{1,2}$", "", s)
    return s or stem


def _year_hint(*names: str) -> int | None:
    for n in names:
        m = re.search(r"(20\d{2})", n or "")
        if m:
            return int(m.group(1))
    return None


def load_business(path: str | Path) -> Business:
    path = Path(path)
    readme = (path / "README.md").read_text(encoding="utf-8") if (path / "README.md").exists() else ""
    pieces: list[tuple[str, str, str, pd.DataFrame, list[str]]] = []  # (file, sheet, base, df, notes)

    for f in sorted(path.iterdir()):
        suf = f.suffix.lower()
        try:
            if suf == ".csv":
                sheets = [("", _read_csv_rows(f))]
            elif suf in (".xlsx", ".xlsm", ".xls"):
                sheets = _read_excel_sheets(f)
            elif suf in (".sqlite", ".db", ".sqlite3"):
                con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
                names = [r[0] for r in con.execute("select name from sqlite_master where type in ('table','view')")]
                for n in names:
                    df = pd.read_sql_query(f'select * from "{n}"', con)
                    pieces.append((f.name, n, n, df, []))
                con.close()
                continue
            else:
                continue
        except Exception as e:  # noqa: BLE001 — một file hỏng không làm hỏng cả shop
            pieces.append((f.name, "", f.stem, pd.DataFrame(), [f"không đọc được: {e}"]))
            continue
        for sheet, rows in sheets:
            notes: list[str] = []
            df = _rows_to_frame(rows, notes)
            if df is None or df.empty:
                continue
            base = _base_name(f.stem) if not sheet or _PERIOD_SHEET.match(sheet.strip()) else _base_name(sheet)
            pieces.append((f.name, sheet, base, df, notes))

    # gom theo tên gốc; trong cùng tên gốc, các mảnh có bộ cột gần giống nhau (≥ 70%) là một bảng
    groups: list[tuple[str, set, list]] = []
    for fname, sheet, base, df, notes in pieces:
        cols = {clean.norm_col(c) for c in df.columns}
        for gbase, gcols, items in groups:
            if gbase == base and len(cols & gcols) >= 0.7 * len(cols | gcols):
                items.append((fname, sheet, df, notes))
                gcols |= cols
                break
        else:
            groups.append((base, set(cols), [(fname, sheet, df, notes)]))

    tables, all_notes = {}, {}
    for base, _, items in groups:
        frames, notes = [], []
        for fname, sheet, df, n in items:
            df = _strip_cells(df) if not df.empty else df
            df = _infer_columns(df, _year_hint(sheet, fname), n, dates=True, others=False)
            if len(items) > 1:
                df.insert(0, "_nguon", fname + (f"[{sheet}]" if sheet else ""))
            frames.append(df)
            for x in n:
                if x not in notes:
                    notes.append(x)
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        if len(frames) > 1:
            # cột ngày ở mảnh này đã đổi, ở mảnh khác thì chưa (vd toàn trống) → đổi nốt
            for c in df.columns:
                if any(pd.api.types.is_datetime64_any_dtype(f[c]) for f in frames if c in f):
                    df[c] = pd.to_datetime(df[c].map(lambda v: clean.parse_datetime(v)), errors="coerce")
        df = _infer_columns(df, None, notes, dates=True, others=True)
        df = _group_fill_and_dedupe(df, notes)
        df = _merge_name_variants(df, notes)
        name = _unique(re.sub(r"\s+", "_", clean.strip_accents(base)).strip("_"), tables)
        if len(items) > 1:
            notes.insert(0, f"gộp {len(items)} sheet/file cùng loại: "
                            + ", ".join(sorted({a + (f'[{b}]' if b else '') for a, b, _, _ in items}))
                            + " (cột `_nguon` ghi file/sheet gốc)")
        else:
            fname, sheet = items[0][0], items[0][1]
            notes.insert(0, f"từ {fname}" + (f" [sheet {sheet}]" if sheet else ""))
        tables[name] = df
        all_notes[name] = notes
    _derive_alias_columns(tables, all_notes)
    return Business(path.name, path, readme, tables, all_notes)


_HONORIFIC = {"a", "anh", "chi", "c", "em", "e", "chu", "co", "thay", "bac", "ong", "ba"}
_PLATE = re.compile(r"\b\d{2}[A-Za-z]\d?[\s-]*[\d.]{4,7}\b")
_PHONE = re.compile(r"[\s(\-–]*(0\d[\d .]{7,12}\d)\)?\s*$")


_NAME_COL = re.compile(r"(^|_)(hang|hang_hoa|hang_ban|mat_hang|san_pham|sp|mon|thuoc|item|product|khach|khach_hang|nhom|"
                       r"nhom_hang|loai|phan_loai|doi_tac|ncc|nha_cung_cap)($|_)")
_NOT_NAME_COL = re.compile(r"(^|_)(ma|id|so|sl|luong|gia|tien|don_vi|dvt|sdt|ghi_chu|trang_thai|ngay|nguon)($|_)")


def _merge_name_variants(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Gộp các cách viết của cùng một tên về cách viết nhiều dòng nhất: 'SINH TỐ XOÀI (L)', 'Bia  Heineken', 'A54 128 cu'.

    Lọc `==` theo một cách viết sẽ thiếu dòng mà không báo lỗi. Chỉ gộp khi khác hoa thường, khoảng trắng, dấu câu, hoặc
    một bên không gõ dấu; hai tên khác dấu thật ('Sting dâu', 'Sting dầu') vẫn giữ riêng."""
    df = df.copy()
    for c in df.columns:
        key = clean.norm_col(c)
        if c.startswith("_") or "__" in c or not _NAME_COL.search(key) or _NOT_NAME_COL.search(key):
            continue
        s = df[c].dropna()
        if s.empty or not s.map(lambda v: isinstance(v, str)).all():
            continue
        counts = s.value_counts()
        groups: dict[str, list] = {}
        for v in counts.index:
            groups.setdefault(clean.norm_key(v), []).append(v)
        repl = {}
        for vs in groups.values():
            if len(vs) < 2:
                continue
            # đại diện: cách viết có dấu nhiều dòng nhất (value_counts đã giảm dần); toàn không dấu thì lấy cái đầu
            best = next((v for v in vs if clean.strip_accents(v) != v), vs[0])
            for v in vs:
                if v != best and (clean.norm_text(v) == clean.norm_text(best) or clean.strip_accents(v) == v):
                    repl[v] = best
        if repl:
            n = int(df[c].isin(list(repl)).sum())
            df[c] = df[c].map(lambda v: repl.get(v, v))
            notes.append(f"cột `{c}`: {n} dòng viết khác (hoa thường, thiếu dấu, thừa khoảng trắng) đã gộp về một cách viết")
    return df


def _derive_alias_columns(tables: dict, notes: dict) -> None:
    """Thêm cột chuẩn hoá cho sổ tay ghi tắt: tên hàng đầy đủ, họ tên thợ, biển số, tên khách không kèm SĐT.

    Câu hỏi nói tên đầy đủ, sổ ghi 'Sạc 25W SS' / 'A.Đức': so thẳng thì không khớp và ra 0."""
    abbr, names = {}, {}
    for t, df in tables.items():
        cols = {clean.norm_col(c): c for c in df.columns}
        if "ghi_tat" in cols and "ten_hang" in cols:
            for _, r in df.iterrows():
                loai = r[cols["loai"]] if "loai" in cols else None
                abbr[clean.norm_key(r[cols["ghi_tat"]])] = (r[cols["ten_hang"]], loai)
                abbr[clean.norm_key(r[cols["ten_hang"]])] = (r[cols["ten_hang"]], loai)
        if "ho_ten" in cols and len(df) < 100:
            last = {}
            for v in df[cols["ho_ten"]].dropna():
                last.setdefault(clean.norm_key(v).split()[-1], []).append(v)
            names = {k: v[0] for k, v in last.items() if len(v) == 1}
    for t, df in tables.items():
        added = []
        for c in list(df.columns):
            k = clean.norm_col(c)
            if abbr and k in ("hang", "may", "ten_hang", "mat_hang") and "ghi_tat" not in {clean.norm_col(x) for x in df.columns}:
                hit = df[c].map(lambda v: abbr.get(clean.norm_key(v)))
                if hit.notna().mean() > 0.5:
                    df[f"{c}__ten"] = hit.map(lambda h: h[0] if h else None).where(hit.notna(), df[c])
                    df[f"{c}__loai"] = hit.map(lambda h: h[1] if h else None)
                    added += [f"{c}__ten", f"{c}__loai"]
            if names and k in ("tho", "nhan_vien", "nv"):
                def full(v):
                    w = [x for x in clean.norm_key(v).split() if x not in _HONORIFIC]
                    return names.get(w[-1]) if w else None
                hit = df[c].map(full)
                if hit.notna().mean() > 0.5:
                    df[f"{c}__ho_ten"] = hit
                    added.append(f"{c}__ho_ten")
            if k == "xe" and df[c].dtype == object:
                plate = df[c].map(lambda v: (_PLATE.search(v).group(0) if isinstance(v, str) and _PLATE.search(v) else None))
                if plate.notna().mean() > 0.5:
                    df[f"{c}__bien_so"] = plate
                    df[f"{c}__dong"] = df[c].map(lambda v: clean.norm_text(_PLATE.sub("", v)) if isinstance(v, str) else None)
                    added += [f"{c}__bien_so", f"{c}__dong"]
            if k in ("khach", "khach_hang", "ten_khach") and df[c].dtype == object:
                has = df[c].map(lambda v: isinstance(v, str) and bool(_PHONE.search(v)))
                if has.mean() > 0.01:
                    df[f"{c}__ten"] = df[c].map(lambda v: _PHONE.sub("", v).strip(" -–(") if isinstance(v, str) else v)
                    added.append(f"{c}__ten")
        if added:
            notes[t].append("cột thêm từ sổ ghi tắt: " + ", ".join(f"`{a}`" for a in added)
                            + " (tên đầy đủ, loại hàng, họ tên, biển số…; lọc theo tên trong câu hỏi thì dùng các cột này)")


def _unique(name, existing):
    n, i = name, 2
    while n in existing:
        n, i = f"{name}_{i}", i + 1
    return n


# ---------------------------------------------------------------- mô tả bảng cho prompt

def profile_tables(tables: dict[str, pd.DataFrame], notes: dict[str, list[str]],
                   max_examples: int = 4, max_levels: int = 25) -> str:
    out = []
    for name, df in tables.items():
        out.append(f"### T[{name!r}]  ({len(df)} dòng)")
        for n in notes.get(name, []):
            out.append(f"- ghi chú: {n}")
        for col in df.columns:
            s = df[col]
            nn = s.dropna()
            nnull = len(s) - len(nn)
            dtype = _dtype_label(s)
            line = f"- `{col}` [{dtype}]"
            if nnull:
                line += f" trống {nnull}"
            if nn.empty:
                out.append(line + " (toàn trống)")
                continue
            if dtype in ("số nguyên", "số thực"):
                line += f" min={_fmt(nn.min())} max={_fmt(nn.max())}"
            elif dtype == "ngày giờ":
                has_time = (nn.dt.hour != 0).any() or (nn.dt.minute != 0).any()
                line += f" từ {nn.min():%Y-%m-%d} đến {nn.max():%Y-%m-%d}" + (" (có giờ)" if has_time else "")
            vc = nn.astype(str).value_counts()
            if col == "_nguon":
                out.append(line + f" ({len(vc)} file/sheet)")
                continue
            if dtype == "chữ" and len(vc) <= max_levels:
                line += " giá trị: " + ", ".join(f"{_q(v)}×{c}" for v, c in vc.items())
            elif dtype not in ("ngày giờ",):
                ex = list(dict.fromkeys(nn.astype(str).head(200)))[:max_examples]
                line += f" ({len(vc)} giá trị khác nhau) ví dụ: " + ", ".join(_q(v) for v in ex)
            out.append(line)
        out.append("")
    return "\n".join(out)


def _q(v, n=60):
    v = str(v)
    return repr(v if len(v) <= n else v[:n] + "…")


def _fmt(x):
    try:
        return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"
    except Exception:  # noqa: BLE001
        return str(x)


def _dtype_label(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "ngày giờ"
    if pd.api.types.is_integer_dtype(s):
        return "số nguyên"
    if pd.api.types.is_float_dtype(s):
        return "số thực"
    if pd.api.types.is_bool_dtype(s):
        return "đúng/sai"
    return "chữ"


# ---------------------------------------------------------------- cache

def _code_hash() -> str:
    """Băm mã nguồn loader + clean + phiên bản pandas/numpy: sửa cách làm sạch hoặc đổi thư viện
    thì cache cũ tự hết hiệu lực (pickle của pandas khác phiên bản có thể đọc ra dữ liệu khác)."""
    import hashlib
    here = Path(__file__).parent
    src = b"".join((here / f).read_bytes() for f in ("loader.py", "clean.py"))
    src += f"pandas={pd.__version__};numpy={np.__version__}".encode()
    return hashlib.sha256(src).hexdigest()[:10]


def cache_file(path: str | Path, cache_dir: str | Path) -> Path:
    path = Path(path)
    return Path(cache_dir) / f"{path.parent.parent.name}__{path.name}__{_code_hash()}.pkl"


def load_cached(path: str | Path, cache_dir: str | Path) -> Business:
    path = Path(path)
    cache = cache_file(path, cache_dir)
    mtime = max((f.stat().st_mtime for f in path.iterdir()), default=0)
    if cache.exists() and cache.stat().st_mtime >= mtime:
        try:
            with cache.open("rb") as fh:
                return pickle.load(fh)
        except Exception:  # noqa: BLE001
            pass
    biz = load_business(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump(biz, fh)
    return biz
