"""Load one business (CSV/XLSX/SQLite) into DuckDB with typed, named tables and a schema prompt."""
import csv
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd


def norm(x):
    x = unicodedata.normalize("NFKD", str(x).replace("đ", "d").replace("Đ", "D"))
    return re.sub(r"[^a-z0-9]+", "_", x.encode("ascii", "ignore").decode().lower()).strip("_")


def ckey(x):
    """Column-name key: like norm, but camelCase headers (MaHoaDon) split into words."""
    return norm(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(x)))


def quote(x):
    return '"' + str(x).replace('"', '""') + '"'


HEADER_KEYS = ("stt", "ma", "ten", "ngay", "thoi_gian", "gio", "so_luong", "sl", "don_gia", "gia", "thanh_tien",
               "tien", "so_hoa_don", "trang_thai", "khach", "khach_hang", "hoa_don", "hang", "hang_hoa", "san_pham",
               "mon", "don_vi", "dvt", "tong_tien", "tong", "nhom", "loai", "ghi_chu", "nhan_vien", "tho", "xe",
               "lam_gi", "so", "size", "hang_sx", "qty", "price", "amount", "date", "status", "id", "name")


def header_score(row):
    vals = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
    if len(vals) < 2:
        return -1
    hits = sum(any(ckey(v) == k or ckey(v).startswith(k + "_") for k in HEADER_KEYS) for v in vals)
    nums = sum(bool(re.fullmatch(r"[\d,. /:-]+", v)) for v in vals)
    return hits * 5 + min(len(set(vals)), 12) - nums * 2 if hits else -1


def fix_header(rows):
    scores = [header_score(r) for r in rows[:25]]
    idx = max(range(len(scores)), key=scores.__getitem__) if scores else 0
    if not scores or scores[idx] < 7:
        idx = 0
    names, counts = [], defaultdict(int)
    for i, v in enumerate(rows[idx] if rows else []):
        col = str(v).strip() if pd.notna(v) and str(v).strip() else f"col_{i + 1}"
        counts[col] += 1
        names.append(f"{col}_{counts[col]}" if counts[col] > 1 else col)
    width = max([len(names)] + [len(r) for r in rows[idx + 1:]])
    names += [f"col_{i + 1}" for i in range(len(names), width)]
    body = [list(r) + [None] * (width - len(r)) for r in rows[idx + 1:]]
    return pd.DataFrame(body, columns=names, dtype=object)


DMY = r"^(?P<day>\d{1,2})[/.-](?P<month>\d{1,2})[/.-](?P<year>\d{4})"
YMD = r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
THG = r"^(?P<day>\d{1,2}) thg (?P<month>\d{1,2}),? (?P<year>\d{4})"
DM = r"^(?P<day>\d{1,2})/(?P<month>\d{1,2})"
HMS = r"(?:[ T,]+(?P<hour>\d{1,2})[:h](?P<minute>\d{2})(?::(?P<second>\d{2}))?(?:\.\d+)?)?$"


def to_dates(t, year=None):
    """Vectorized dd/mm/yyyy, yyyy-mm-dd, 'd thg m yyyy', d/m (+ sheet/file year), Excel serial, else NaT."""
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    for pat in (DMY + HMS, YMD + HMS, THG + HMS) + ((DM + HMS,) if year else ()):
        p = t.str.extract(pat, flags=re.IGNORECASE)
        if "year" not in p:
            p["year"] = year
        p = p.dropna(subset=["day", "month"])
        if len(p):
            p = p[["year", "month", "day", "hour", "minute", "second"]].fillna(0).astype(int)
            out.loc[p.index] = out.loc[p.index].fillna(pd.to_datetime(p, errors="coerce"))
    serial = t.str.fullmatch(r"[3-6]\d{4}(\.0+)?").fillna(False)
    if serial.any():
        out.loc[serial] = pd.Timestamp("1899-12-30") + pd.to_timedelta(t[serial].astype(float).astype(int), unit="D")
    return out


def to_number(t):
    """'1.234.000', '1,234,000', '198 000', '55.000đ', '25k', '8tr600', '1,27tr' -> float, else NaN."""
    clean = t.str.replace(r"\s*(?:₫|vnđ|vnd|đ|đồng)\s*$", "", regex=True, flags=re.IGNORECASE).str.replace(" ", " ")
    clean = clean.where(~clean.str.fullmatch(r"[+-]?\d{1,3}(?: \d{3})+").fillna(False), clean.str.replace(" ", "", regex=False))
    out = pd.Series(float("nan"), index=t.index)
    m = clean.str.extract(r"^([+-]?\d+(?:[.,]\d+)?)\s*(k|tr|triệu)(\d{1,3})?$", flags=re.IGNORECASE).dropna(subset=[0])
    if len(m):
        scale = m[1].str.lower().map({"k": 1e3, "tr": 1e6, "triệu": 1e6})
        frac = m[2].fillna("").str.ljust(3, "0").replace("000", "0").astype(float) * scale / 1e3
        frac = frac.where(~m[0].str.startswith("-"), -frac)  # the sign covers the tail too, including '-0tr600'
        out.loc[m.index] = (m[0].str.replace(",", ".", regex=False).astype(float) * scale + frac.where(m[2].notna(), 0)).round()
    us = clean.str.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?").fillna(False)
    eu = clean.str.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+(?:,\d+)?").fillna(False)
    clean.loc[us] = clean.loc[us].str.replace(",", "", regex=False)
    clean.loc[eu] = clean.loc[eu].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    dec = clean.str.fullmatch(r"[+-]?\d+,\d+").fillna(False)
    clean.loc[dec] = clean.loc[dec].str.replace(",", ".", regex=False)
    return out.fillna(pd.to_numeric(clean, errors="coerce"))


ORDER_ID = r"(^|_)(ma|so|id)_?(don|hd|hoa_don|phieu|bill|ct|chung_tu|giao_dich)|^(ma|so)_?(dh|hd)$|^(hd|hoa_don|don_hang|don_so|so_don|order_?id|bill|stt)$"
STATUS_COL = r"(^|_)(trang_?thai|tinh_?trang|status|xu_?ly|ghi_?chu)"
CANCELLED = r"(?:^|_)(?:huy|cancel\w*|khong_giao_duoc)(?:_|$)"
# (?<!kiem_): 'kiểm tra lại tuần sau' / 'kiểm tra hàng' is a check, not a return; without it such tickets left revenue as returned orders
RETURNED = r"(?:^|_)(?<!kiem_)(?:tra_hang|tra_lai|tra_ve|doi_tra|hoan_tra|hoan_hang|return\w*)(?:_|$)|khach_(?:mang_)?tra$"
NOTE_ROW = re.compile(r"(?i)^\s*(ghi chú|ghi chu|tổng|tong cong|cộng|total)\b")


def prepare(df, year=None):
    df = df.dropna(how="all").copy()
    if not df.empty and len(df.columns) > 1:
        text = df.astype("string").apply(lambda col: col.str.strip().str.normalize("NFC")).replace("", pd.NA)
        repeats = sum(text[c].eq(str(c)).fillna(False).astype(int) for c in df.columns)  # repeated header rows
        sparse = text.notna().sum(axis=1) <= max(2, len(df.columns) // 3)
        note = text.apply(lambda col: col.str.match(NOTE_ROW).fillna(False)).any(axis=1)  # footer/subtotal rows
        keys = [c for c in df.columns if re.search(ORDER_ID + r"|(^|_)(ngay|date)", ckey(c))]
        keyless = text[keys].isna().any(axis=1) | text[keys].apply(lambda col: col.str.match(NOTE_ROW).fillna(False)).any(axis=1) if keys else False
        df = df.loc[(repeats < max(2, len(df.columns) // 2)) & ~(note & (sparse | keyless))].copy()
    for c in list(df.columns):
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        native = s.map(lambda v: hasattr(v, "year") and hasattr(v, "month"))
        t = s.map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))).astype("string").str.strip().replace("", pd.NA)
        valid = t.dropna()
        if valid.empty:
            df[c] = t
            continue
        key = ckey(c)
        if native.any() or valid.str.contains(r"^\d{1,4}[/.-]\d|\d thg \d", regex=True).mean() > 0.8 or re.search(r"(^|_)(ngay|date|thoi_gian)", key):
            dt = to_dates(t, year)
            if native.any():
                dt.loc[native] = pd.to_datetime(s[native].map(pd.Timestamp))
            if dt.notna().sum() / len(valid) > 0.8:
                df[c] = dt
                continue
        is_id = bool(re.search(r"(^|_)(ma|id|code|stt|sdt|phone|dien_thoai|barcode|mst|bien_so|so_dt|imei|serial|so_seri|dia_chi)(_|$)", key))
        if is_id or valid.str.match(r"^0\d+$").any():
            df[c] = t
            continue
        num = to_number(t)
        if num.notna().sum() / len(valid) > 0.97:
            df[c] = num
            continue
        qty = t.str.extract(r"^([+-]?\d+(?:[.,]\d+)?)\s*([^\d\s.,=:/-][^=:/]{0,15})$")  # "2 gói", "1 ly"
        both = num.fillna(pd.to_numeric(qty[0].str.replace(",", ".", regex=False), errors="coerce"))
        if both.notna().sum() / len(valid) > 0.97 and qty[1].notna().any():
            df[c] = both
            df.insert(df.columns.get_loc(c) + 1, f"{c} (đơn vị ghi kèm)", qty[1].astype("string"))
            continue
        df[c] = t
    # Rulebook §8: continuation lines of a grouped order leave order-level cells blank -> copy from the first line.
    ids = sorted((c for c in df.columns if re.search(ORDER_ID, ckey(c))), key=lambda c: ckey(c) == "stt")  # STT last
    filled = []
    if ids and len(df) and df[ids[0]].iloc[0] is not pd.NA and pd.notna(df[ids[0]].iloc[0]):
        blank = df[ids[0]].isna()
        if blank.mean() > 0.05:
            group = [c for c in df.columns if df.loc[blank, c].isna().mean() > 0.95]
            # fill continuation lines from their own order's first line only. A plain ffill would carry
            # a sparse order-level cell (note "trả hàng", order discount) into later orders
            first = df[group].groupby((~blank).cumsum()).transform("first")
            df.loc[blank, group] = first.loc[blank]
            filled = [c for c in group if c != ids[0]]  # written once per order: an amount here now repeats on every line of it
    if ids:
        # Rulebook: blank cells on later lines of an order take the order's value (date, customer, note, status...).
        g = df.groupby(df[ids[0]], sort=False)
        for c in df.columns:
            if c != ids[0] and df[c].isna().any() and not pd.api.types.is_numeric_dtype(df[c]) and (g[c].nunique() <= 1).mean() > 0.99:
                df[c] = df[c].fillna(g[c].transform("first"))
        df = df.drop_duplicates()  # Rulebook §8: identical lines of the same order are an export error, count once
    df = df.dropna(how="all", subset=[c for c in df.columns if not str(c).startswith("_")] or None)
    df = split_toppings(df)  # after the dedup: a split topping line may equal a topping line of its own in the same order
    # Status classes as flags, so every sampled query excludes cancelled/returned orders the same way.
    for c in [c for c in df.columns if re.search(STATUS_COL, ckey(c)) and df[c].dtype == "string"]:
        k = df[c].map(lambda v: norm(v) if isinstance(v, str) else "")
        huy, tra = k.str.contains(CANCELLED), k.str.contains(RETURNED)
        if huy.any() or tra.any():
            df["_huy"], df["_tra"] = huy, tra
            if ids:
                order = df[ids[0]]
                df["_don_huy"] = df["_huy"].groupby(order).transform("any").fillna(df["_huy"]).astype(bool)
                df["_don_tra"] = df["_tra"].groupby(order).transform("any").fillna(df["_tra"]).astype(bool)
            break
    df.attrs["filled"] = [c for c in filled if c in df.columns]  # read by load_business (order_level_tables)
    return df


TOPPING_PART = re.compile(r"\s*(?:(\d{1,2})\s+)?(.*?[^\W\d_].*?)\s*\(([^()]*\d[^()]*)\)\s*")
TOPPING_CELL = r"\S \+ (?:\d{1,2}\s+)?[^+()]*[^\W\d_][^+()]*\([^()]*\d[^()]*\)\s*$"


def split_toppings(df):
    """Rulebook (Cà phê): a topping written in the item's cell, 'Trà chanh (M) + 2 Trân châu đen (14.000đ)', is an item of its own:
    its count and money go to the topping (the same item as a topping sold on its own line), not to the drink. Each such line becomes
    one line per item: the drink keeps the line's qty and price, its amount less the toppings; each topping gets its count, its money
    and the per-item cells (unit, code) of its own lines. The money columns are found by the line arithmetic, amount = price x qty
    + toppings on 80%+ of these lines; a table where no columns fit is left as it is."""
    num = [c for c in df.columns if not str(c).startswith("_") and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if len(num) < 3 or len(df) < 20:
        return df
    for col in [c for c in df.columns if df[c].dtype == "string" and not str(c).startswith("_")]:
        cand = df[col].str.contains(" + ", regex=False).fillna(False)  # cheap first: most columns have no ' + ' at all
        if cand.sum() < 5:
            continue
        cand &= df[col].str.contains(TOPPING_CELL).fillna(False)  # ' + [n] name (money)' at the end, before any per-cell work
        if cand.sum() < 5:
            continue
        cells = {i: v.split(" + ") for i, v in df.loc[cand, col].items()}
        found = {i: [TOPPING_PART.fullmatch(p) for p in parts[1:]] for i, parts in cells.items()}
        found = {i: ms for i, ms in found.items() if all(ms) and cells[i][0].strip()}
        money = to_number(pd.Series([m[3].strip() for ms in found.values() for m in ms], dtype="string")).tolist()
        parsed, k = {}, 0
        for i, ms in found.items():
            tops = [(int(m[1] or 1), m[2].strip(), money[k + j]) for j, m in enumerate(ms)]
            k += len(ms)
            if all(t[2] > 0 for t in tops):  # NaN fails too
                parsed[i] = (cells[i][0].strip(), tops)
        if len(parsed) < 5 or len(parsed) < 0.8 * cand.sum():
            continue
        idx = list(parsed)
        top_money = pd.Series([sum(t[2] for t in parsed[i][1]) for i in idx], index=idx)
        fits = [(((df.loc[idx, a] - df.loc[idx, p] * df.loc[idx, q] - top_money).abs() <= 1).mean(), a, p, q)
                for a in num for p in num for q in num if len({a, p, q}) == 3]
        best = max(fits, key=lambda f: f[0], default=None)
        if not best or best[0] < 0.8:
            continue
        _, amount, price, qty = best
        if df.loc[idx, qty].abs().median() > df.loc[idx, price].abs().median():  # price x qty is symmetric: qty is the small one
            price, qty = qty, price
        alone = df.loc[~cand]
        by_key = {}
        for name in alone[col].value_counts().index:  # commonest spelling first
            by_key.setdefault(spelling_key(name), name)
        unit_price = alone.groupby(col)[price].median()

        def own(name, each=None):
            """The way the book writes this item on a line of its own ('Trân châu đen' -> 'Topping Trân châu đen')."""
            k = spelling_key(name)
            if k in by_key:
                return by_key[k]
            near = [v for kk, v in by_key.items() if kk.endswith(" " + k) and each and v in unit_price and abs(unit_price[v] - each) <= 0.25 * each]
            return near[0] if len(near) == 1 else name
        per_item, mode_of = [], {}
        for t in [c for c in df.columns if df[c].dtype == "string" and c != col and not str(c).startswith("_")]:
            g = alone.groupby(col)[t]
            big = g.size()[lambda s: s >= 3].index
            share = g.agg(lambda x: x.value_counts(normalize=True, dropna=False).iloc[0] if len(x) else 0)
            if len(big) >= 2 and (share[big] >= 0.9).mean() >= 0.9 and alone[t].nunique() >= 2:
                per_item.append(t)
                mode_of[t] = g.agg(lambda x: x.dropna().mode().iloc[0] if x.notna().any() else pd.NA)
        pos = pd.Series(range(len(df)), index=df.index, dtype=float)
        rep = [(i, j) for i in idx for j in range(len(parsed[i][1]) + 1)]
        if any(df.at[i, amount] - top_money[i] <= 0 for i in idx):
            continue
        new = df.loc[[i for i, _ in rep]].copy()
        names, amounts, qtys, prices = [], [], [], []
        for i, j in rep:
            head, tops = parsed[i]
            if j == 0:
                names.append(own(head)); amounts.append(df.at[i, amount] - top_money[i]); qtys.append(df.at[i, qty]); prices.append(df.at[i, price])
            else:
                n, name, money = tops[j - 1]
                names.append(own(name, money / n)); amounts.append(money); qtys.append(float(n)); prices.append(money / n)
        new[col] = pd.array(names, dtype="string")
        new[amount], new[qty], new[price] = amounts, qtys, prices
        for t in per_item:
            vals = [new[t].iat[k] if j == 0 or pd.isna(new[t].iat[k]) else mode_of[t].get(names[k], new[t].iat[k]) for k, (_, j) in enumerate(rep)]
            new[t] = pd.array(vals, dtype=df[t].dtype)
        if (note := f"{qty} (đơn vị ghi kèm)") in new.columns and note not in per_item:  # the drink's '1 ly' says nothing of its topping
            new[note] = pd.array([v if j == 0 else pd.NA for v, (_, j) in zip(new[note], rep)], dtype=df[note].dtype)
        new.index = pd.Index([pos[i] + j / 100 for i, j in rep])
        rest = df.loc[~df.index.isin(idx)].copy()
        rest.index = pd.Index(pos[rest.index].tolist())
        attrs = df.attrs
        df = pd.concat([rest, new]).sort_index().reset_index(drop=True)
        df.attrs = attrs
    return df


DATE_COL = r"(^|_)(ngay|date|thoi_gian|gio|time|tao|created)(_|$)"
NOT_ORDER_DATE = r"(^|_)(han|hsd|nsx|sinh|het|expir\w*|tra|hoan|thu|nhap|giao(?!_dich))(_|$)"  # expiry, birth, return, payment, receipt, delivery dates
DAY = "CAST({} AS DATE)"


def order_dates(db, members, columns):
    """Rulebook §1: the order date decides an order's period, whatever its status. Some books date the return lines of an order on
    the day of the return (SEARCH: every order whose lines span several dates was a returned one, in two shops; no valid or cancelled
    order did), so a query counting the orders of a month by line date counts a returned order in two months. On the lines of a
    cancelled/returned order (status flags, over all tables of one union) whose dates differ, every order-date column takes the order's
    first value. Valid orders stay as recorded. members: the tables of one union (or one table); columns: their (name, type)."""
    names = [c for c, _ in columns]
    if "_huy" not in names or "_tra" not in names:
        return 0
    ids = sorted((c for c in names if re.search(ORDER_ID, ckey(c))), key=lambda c: ckey(c) == "stt")
    if not ids:
        return 0
    oid, changed = quote(ids[0]), 0
    for c, typ in columns:
        if not (typ.upper() == "DATE" or typ.upper().startswith("TIMESTAMP")) or not re.search(DATE_COL, ckey(c)) or re.search(NOT_ORDER_DATE, ckey(c)):
            continue
        src = " UNION ALL ".join(f'SELECT {oid} AS o, {quote(c)} AS d, ("_huy" OR "_tra") AS f FROM {quote(t)}' for t in members)
        db.execute("DROP TABLE IF EXISTS _od")
        db.execute(f"CREATE TEMP TABLE _od AS SELECT o, min(d) AS m FROM ({src}) s WHERE o IS NOT NULL AND d IS NOT NULL GROUP BY o "
                   f"HAVING max(CASE WHEN f THEN 1 ELSE 0 END) = 1 AND count(DISTINCT {DAY.format('d')}) > 1")
        for t in members:
            n = db.execute(f"SELECT count(*) FROM {quote(t)} x JOIN _od ON x.{oid} = _od.o WHERE x.{quote(c)} <> _od.m").fetchone()[0]
            if n:
                db.execute(f"UPDATE {quote(t)} SET {quote(c)} = _od.m FROM _od WHERE {quote(t)}.{oid} = _od.o AND {quote(t)}.{quote(c)} <> _od.m")
                changed += n
        db.execute("DROP TABLE _od")
    return changed


def union_key(name):
    """Monthly/yearly split tables share a base name once periods are removed."""
    return re.sub(r"_+", "_", re.sub(r"(?:^|_)(?:t|thang|nam)?_?(?:0?[1-9]|1[0-2])?_?20\d{2}(?=_|$)|(?:^|_)(?:t|thang)_?(?:0?[1-9]|1[0-2])(?=_|$)", "_", name)).strip("_") or name


EXCEL = (".xlsx", ".xlsm", ".xls")  # calamine reads all three; train and public ship only .xlsx, a private shop may not


def read_rows(f):
    """Yield (table_name, rows) for every sheet/table in a file."""
    ext = f.suffix.lower()
    if ext in EXCEL:
        from python_calamine import CalamineWorkbook
        book = CalamineWorkbook.from_path(str(f))
        try:
            for sheet in book.sheet_names:
                rows = book.get_sheet_by_name(sheet).to_python(skip_empty_area=False)
                yield f"{f.stem}_{sheet}" if len(book.sheet_names) > 1 else f.stem, sheet, rows
        finally:
            book.close()
    elif ext == ".csv":
        try:
            text = f.read_text(encoding="utf-8-sig")
        except UnicodeError:
            text = f.read_text(encoding="cp1258")
        try:
            dialect = csv.Sniffer().sniff(text[:16384], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        yield f.stem, None, list(csv.reader(text.splitlines(), dialect))
    elif ext in (".db", ".sqlite", ".sqlite3"):
        with sqlite3.connect(f"file:{f.resolve()}?mode=ro", uri=True) as src:
            for (table,) in src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
                cur = src.execute(f"SELECT * FROM {quote(table)}")
                yield table, None, [[d[0] for d in cur.description]] + [list(r) for r in cur.fetchall()]


def profile(db, n, source):
    """Row count and per-column type + top values (text) or range (others) of one table/view, for the schema prompt."""
    prof = []
    for col, typ in [(r[0], r[1]) for r in db.execute(f"DESCRIBE {quote(n)}").fetchall()]:
        if typ in ("VARCHAR",):
            d = db.execute(f"SELECT count(DISTINCT {quote(col)}) FROM {quote(n)}").fetchone()[0]
            vals = [r[0] for r in db.execute(f"SELECT {quote(col)} FROM {quote(n)} WHERE {quote(col)} IS NOT NULL GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {25 if d <= 25 else 4}").fetchall()]
            prof.append((col, typ, d, vals))
        else:
            lo, hi = db.execute(f"SELECT min({quote(col)}), max({quote(col)}) FROM {quote(n)}").fetchone()
            prof.append((col, typ, None, [str(lo), str(hi)]))
    return {"name": n, "source": source, "rows": db.execute(f"SELECT count(*) FROM {quote(n)}").fetchone()[0], "columns": prof}


def spelling_key(v):
    """A name without case, accents or extra spaces."""
    s = unicodedata.normalize("NFD", str(v).replace("đ", "d").replace("Đ", "D"))
    return " ".join("".join(c for c in s if unicodedata.category(c) != "Mn").lower().split())


def spelling_map(counts):
    """{value: count} -> {value: spelling to use}, for names that differ only in the way they are written (rulebook: such
    names are one customer, one item). Forms with different accents stay apart (Bò, Bơ). A form without accents joins only
    when one accented form shares its key, since otherwise it could be either."""
    groups = defaultdict(list)
    for v, n in counts.items():
        k = spelling_key(v)
        if k and len(k) <= 80 and any(c.isalpha() for c in k):
            groups[k].append((v, n))
    out = {}
    for k, vs in groups.items():
        if len(vs) < 2:
            continue
        forms = defaultdict(list)
        for v, n in vs:
            forms[" ".join(unicodedata.normalize("NFC", str(v)).lower().split())].append((v, n))
        accented = [f for f in forms if f != k]
        best = lambda m: max(m, key=lambda x: (not x[0].isupper(), x[1]))[0]
        if len(accented) == 1:
            forms = {accented[0]: vs}
        elif accented:
            forms.pop(k, None)
        for m in forms.values():
            use = best(m)
            out.update({v: use for v, _ in m if v != use})
    return out


def canon_spellings(db, tables):
    """Writes every name of a shop one way, across all its tables, so an item or a customer is one value in the raw tables,
    the catalog joins and the views alike. Returns the number of cells changed."""
    cols, counts = [], defaultdict(int)
    for t in tables:
        if t == "calendar":
            continue
        for col, typ, *_ in db.execute(f"DESCRIBE {quote(t)}").fetchall():
            if typ == "VARCHAR" and not col.startswith("_"):
                cols.append((t, col))
                for v, n in db.execute(f"SELECT {quote(col)}, count(*) FROM {quote(t)} WHERE {quote(col)} IS NOT NULL GROUP BY 1").fetchall():
                    counts[v] += n
    mapping = spelling_map(counts)
    if not mapping:
        return 0
    db.register("_m", pd.DataFrame({"v": list(mapping), "c": list(mapping.values())}))
    db.execute("CREATE TEMP TABLE _spell AS SELECT * FROM _m")
    db.unregister("_m")
    changed = 0
    for t, col in cols:
        n = db.execute(f"SELECT count(*) FROM {quote(t)} x JOIN _spell m ON x.{quote(col)} = m.v").fetchone()[0]
        if n:
            db.execute(f"UPDATE {quote(t)} SET {quote(col)} = m.c FROM _spell m WHERE {quote(t)}.{quote(col)} = m.v")
            changed += n
    db.execute("DROP TABLE _spell")
    return changed


# A column naming the person who sold / did the work (not the customer: the rulebook tells customers apart by spelling)
PERSON_COL = r"(^|_)(tho|nhan_vien|nv|staff|nguoi_ban|nguoi_lam|nguoi_sua|thu_ngan|ky_thuat|ktv|seller|cashier|employee)(_|$)"
# A form of address before a call name: 'A.Phúc', 'a Phúc', 'C. Lan', 'chị Lan'. Lower-case words only, so a given name such as 'Anh Thư' stays
TITLE = re.compile(r"^(?:[AaCcEe]\.\s*|(?:a|anh|c|chị|chi|cô|co|chú|chu|e|em|bác|bac|ông|ong|bà|ba)\s+)(?=\w)")


def people_map(counts, rosters):
    """{call name: name to write}: one person written several ways ('Phúc', 'a Phúc', 'A.Phúc') becomes one value. With a roster (a
    list of full names), each call name maps to the one full name it ends with (rulebook §8: names of one thing are one thing).
    Without one, the forms of address are dropped."""
    out = {}
    for full_names in rosters:
        index = {spelling_key(f): f for f in full_names}
        hit = {}
        for v in counts:
            for form in (v, TITLE.sub("", v)):
                k = spelling_key(form)
                found = [f for kf, f in index.items() if kf == k or kf.endswith(" " + k)] if k else []
                if len(found) == 1:
                    hit[v] = found[0]
                    break
        if sum(counts[v] for v in hit) >= 0.8 * sum(counts.values()):
            return {v: f for v, f in hit.items() if v != f}
    groups = defaultdict(list)
    for v in counts:
        groups[spelling_key(TITLE.sub("", v))].append(v)
    for vs in groups.values():
        if len(vs) > 1:
            bare = [v for v in vs if not TITLE.match(v)] or vs
            use = max(bare, key=lambda v: counts[v])
            out.update({v: use for v in vs if v != use})
    return out


def canon_people(db, tables):
    """Writes every staff / technician name one way in the columns naming the person who sold or did the work. Returns the cells changed."""
    described = {t: db.execute(f"DESCRIBE {quote(t)}").fetchall() for t in tables if t != "calendar"}
    people = [(t, col) for t, cols in described.items() for col, typ, *_ in cols
              if typ == "VARCHAR" and not col.startswith("_") and re.search(PERSON_COL, ckey(col))]
    if not people:
        return 0
    rosters = []  # (table, names) of the columns of small tables holding full names (2+ words): a staff list
    for t2, cols2 in described.items():
        if db.execute(f"SELECT count(*) FROM {quote(t2)}").fetchone()[0] > 500:
            continue
        for c2, ty2, *_ in cols2:
            if ty2 == "VARCHAR" and not c2.startswith("_"):
                names = [str(r[0]) for r in db.execute(f"SELECT DISTINCT {quote(c2)} FROM {quote(t2)} WHERE {quote(c2)} IS NOT NULL").fetchall()]
                if names and sum(len(x.split()) >= 2 for x in names) >= 0.8 * len(names):
                    rosters.append((t2, names))
    changed = 0
    for t, col in people:
        counts = dict(db.execute(f"SELECT {quote(col)}, count(*) FROM {quote(t)} WHERE {quote(col)} IS NOT NULL GROUP BY 1").fetchall())
        if not counts or len(counts) > 200:
            continue
        for v, use in people_map(counts, [names for t2, names in rosters if t2 != t]).items():
            db.execute(f"UPDATE {quote(t)} SET {quote(col)} = ? WHERE {quote(col)} = ?", [use, v])
            changed += counts[v]
    return changed


NUMERIC_TYPE = re.compile(r"(?i)^(double|float|real|decimal|numeric|bigint|integer|int|smallint|tinyint|hugeint|ubigint|uinteger)")
QTY_PRICE = r"(^|_)(sl|so_luong|qty|quantity|don_gia|gia|gia_ban|price)(_|$)"  # line quantities and unit prices are never order-level
ORDER_LEVEL_VAR = "dm_order_level"  # DuckDB variable: {line table: {"id", "cols", "table"}}, read by sqlexec.order_level_error


def order_level_tables(db, tables, hidden, filled):
    """Rulebook §8: order cells (customer, note, the amount the customer paid, the order discount, the trade-in) belong to the whole
    order. A book writes them once per order (the loader fills them down) or repeats them on every line of it, so SUM over the lines
    counts an order once per line (about twice the right total when orders have two lines). Per visible
    line table with an order id: its numeric columns holding one value per order (filled down, or the same on 99.9%+ of the orders of
    2+ lines), and a table "<name>_don" with one row per order. Returns {line table: {"id", "cols", "table"}}."""
    out = {}
    for n in [n for n in list(tables) if n not in hidden and n != "calendar"]:
        cols = [(r[0], r[1]) for r in db.execute(f"DESCRIBE {quote(n)}").fetchall()]
        names = [c for c, _ in cols]
        ids = sorted((c for c in names if re.search(ORDER_ID, ckey(c))), key=lambda c: ckey(c) == "stt")
        cand = [(c, ty) for c, ty in cols if ids and c != ids[0] and not c.startswith("_")]
        if not cand:
            continue
        oid = quote(ids[0])
        same = lambda c: f"(count({quote(c)}) = 0 OR (count({quote(c)}) = count(*) AND min({quote(c)}) = max({quote(c)})))"  # one value, or none, on every line
        row = db.execute("SELECT count(*), " + ", ".join(f"avg(CAST(k{i} AS DOUBLE))" for i in range(len(cand)))
                         + " FROM (SELECT " + ", ".join(f"{same(c)} AS k{i}" for i, (c, _) in enumerate(cand))
                         + f" FROM {quote(n)} WHERE {oid} IS NOT NULL GROUP BY {oid} HAVING count(*) >= 2)").fetchone()
        if row[0] < 30:
            continue
        const = {c: x or 0 for (c, _), x in zip(cand, row[1:])}
        if not any(ty == "VARCHAR" and const[c] <= 0.5 for c, ty in cand):
            continue  # no text cell (the item) differs between the lines of an order: rows are events (payments of one loan), not order lines
        money = [c for c, ty in cand if NUMERIC_TYPE.match(ty) and (
            (c in filled.get(n, ()) and const[c] >= 0.98)
            or (const[c] >= 0.999 and not re.search(QTY_PRICE, ckey(c))
                and db.execute(f"SELECT count(DISTINCT {quote(c)}) FROM {quote(n)}").fetchone()[0] > 1))]
        if not money:
            continue
        keep = [c for c, ty in cand if c in money or (not NUMERIC_TYPE.match(ty) and const[c] >= 0.99)]
        flags = [f for f in ("_huy", "_tra") if f in names]
        base, i = re.sub(r"_all$", "", n) + "_don", 2
        name = base
        while name in tables:
            name, i = f"{base}_{i}", i + 1
        db.execute(f"CREATE TABLE {quote(name)} AS SELECT {oid}, " + ", ".join(f"any_value({quote(c)}) AS {quote(c)}" for c in keep)
                   + "".join(f", bool_or({quote(f)}) AS {quote('_don' + f)}" for f in flags) + f" FROM {quote(n)} WHERE {oid} IS NOT NULL GROUP BY {oid}")
        tables[name] = f'one row per order of "{n}" (its order id and order-level columns)'
        out[n] = {"id": ids[0], "cols": money, "table": name}
    if out:
        db.execute(f"SET VARIABLE {ORDER_LEVEL_VAR} = " + "'" + json.dumps(out, ensure_ascii=False).replace("'", "''") + "'")
    return out


def load_business(root, business, canon=False):
    root = Path(root)
    folder = root / "data" / business
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    db = duckdb.connect(":memory:")
    tables = {}  # name -> source description
    filled = {}  # name -> columns the loader filled down from an order's first line
    for f in sorted(folder.rglob("*")):
        if not f.is_file() or f.name.startswith("~$"):
            continue
        for name, sheet, rows in read_rows(f):
            rows = [r for r in rows if any(v is not None and str(v).strip() for v in r)]
            if len(rows) < 2:
                continue
            year = re.search(r"20\d{2}", str(sheet or "")) or re.search(r"20\d{2}", f.stem)
            df = prepare(fix_header(rows), int(year.group()) if year else None)
            if df.empty:
                continue
            if sheet is not None:
                df["_sheet"] = sheet  # rulebook: a date without year takes the year of its sheet
            name = norm(name) or "table"
            base, i = name, 2
            while name in tables:
                name, i = f"{base}_{i}", i + 1
            db.register("_src", df)
            db.execute(f"CREATE TABLE {quote(name)} AS SELECT * FROM _src")
            db.unregister("_src")
            tables[name] = f.name + (f"#{sheet}" if sheet else "")
            filled[name] = set(df.attrs.get("filled", ()))
    if not tables:
        raise ValueError(f"No usable tables: {folder}")
    if canon:
        canon_spellings(db, list(tables))
        canon_people(db, list(tables))
    if (root / "calendar.csv").exists():
        db.execute("CREATE TABLE calendar AS SELECT * FROM read_csv_auto(?, header=true)", [str(root / "calendar.csv")])
        tables["calendar"] = "calendar.csv"

    def cols(n):
        return [(r[0], r[1]) for r in db.execute(f"DESCRIBE {quote(n)}").fetchall()]

    groups = defaultdict(list)
    for n in tables:
        groups[tuple(c for c, _ in cols(n) if c != "_sheet")].append(n)
    hidden = set()
    for names in groups.values():
        if len(names) < 2:
            continue
        view = union_key(names[0]) + "_all"
        parts = [f"SELECT *, '{tables[n]}' AS _source FROM {quote(n)}" for n in names]
        db.execute(f"CREATE VIEW {quote(view)} AS " + " UNION ALL BY NAME ".join(parts))
        tables[view] = f"UNION of {len(names)} tables: " + ", ".join(tables[n] for n in names)
        hidden.update(names)
        filled[view] = set().union(*(filled.get(n, set()) for n in names))
    for names in groups.values():  # a return line can sit in the next month's sheet: one pass over all tables of a union
        if "calendar" not in names:
            order_dates(db, names, cols(names[0]))
    orders = order_level_tables(db, tables, hidden, filled)

    info = [profile(db, n, tables[n]) for n in tables if n not in hidden]
    for t in info:
        if t["name"] in orders:
            t["order_level"] = orders[t["name"]]
        if any(t["name"] == o["table"] for o in orders.values()):
            t["orders_of"] = True
    readme = "\n\n".join(p.read_text(encoding="utf-8-sig") for p in sorted(folder.rglob("*.md")))
    return db, {"tables": info, "readme": readme}


def entity_hits(db, meta, question, limit=12):
    """Exact values of text columns whose words appear in the question (product/customer names, statuses)."""
    q = norm(question)
    qw = set(q.split("_"))
    hits = []
    for t in meta["tables"]:
        if t.get("orders_of"):  # an order table repeats its line table's values
            continue
        for col, typ, d, _ in t["columns"]:
            if typ != "VARCHAR" or not d or d > 20000:
                continue
            for (v,) in db.execute(f"SELECT DISTINCT {quote(col)} FROM {quote(t['name'])} WHERE {quote(col)} IS NOT NULL").fetchall():
                nv = norm(v)
                w = set(nv.split("_"))
                if len(nv) >= 3 and not nv.isdigit() and (("_" + nv + "_") in ("_" + q + "_") or (len(w) >= 2 and len(w & qw) / len(w) >= 0.75)):
                    hits.append((len(nv), t["name"], col, v))
    hits.sort(reverse=True)
    return [f"{quote(t)}.{quote(c)} = '{v}'" for _, t, c, v in hits[:limit]]


def schema_text(meta):
    out = []
    for t in meta["tables"]:
        out.append(f"TABLE {quote(t['name'])}  -- {t['rows']} rows, from {t['source']}")
        for col, typ, d, vals in t["columns"]:
            if d is None:
                out.append(f"  {quote(col)} {typ}  range [{vals[0]} .. {vals[1]}]")
            else:
                tag = "values" if d <= 25 else f"{d} distinct, e.g."
                out.append(f"  {quote(col)} {typ}  {tag} {json.dumps([str(v)[:60] for v in vals], ensure_ascii=False)}")
        if o := t.get("order_level"):
            out.append(f"  -- ORDER-LEVEL columns {', '.join(quote(c) for c in o['cols'])}: one value per order ({quote(o['id'])}), repeated on every line "
                       f"of it; SUM/AVG them once per order, from {quote(o['table'])} (one row per order), never over these lines")
    return "\n".join(out)


if __name__ == "__main__":
    import sys, time
    for root in sys.argv[1:]:
        for b in sorted((Path(root) / "data").iterdir()):
            t0 = time.time()
            db, m = load_business(root, b.name)
            print(f"{b.name}: {len(m['tables'])} tables, {len(schema_text(m))} chars, {time.time() - t0:.1f}s")
