"""Trích "đặc tả dữ liệu" từ 30 shop train để làm khuôn cho bộ sinh dữ liệu shop (giai đoạn 1: sinh dữ liệu, chưa sinh câu hỏi).

    python eval/build_data_spec.py            # ghi data/synth_source/data_spec/

Đầu ra:
- archetypes.json      kiểu sổ → shop, họ file, cách đặt tên sheet/file, số dòng tiêu đề, tiêu đề hai tầng
- column_names.json    vai trò cột → các tên cột thật đã gặp (khuôn đặt tên cột), cột thừa không mang vai trò
- value_formats.json   cách ghi tiền, ngày, giờ, số lượng; từ vựng trạng thái đơn; cách ghi khách lẻ (đếm trên file thô)
- quirks.json          các bẫy loader đã xử lý (dòng tên báo cáo, dòng tổng, dòng trùng, ô trống dòng sau…) và shop có bẫy
- readme_phrases.json  cách README mô tả từng vai trò cột, các mục của README
- calibration.json     thống kê theo shop và theo ngành: số đơn/ngày, hệ số thứ, tháng, Tết, giờ, dòng/đơn, giá trị đơn,
                       tỉ lệ trạng thái, khách lẻ, khách quay lại, độ tập trung mặt hàng, số mức giá, kênh, thanh toán
- shop_complexity.json độ phức tạp cấu trúc từng shop (số file/bảng/cột, thông tin có ghi, tầng) + kết quả log có đáp án
                       (errors/*/by_shop.json, chạy eval/ingest_real_logs.py trước) + tương quan số bảng ↔ PA
- canonical_schema.json, catalog_schema.json, render_recipes.json, new_industries.json, volume_plan.json, model_routing.json
                       thiết kế bộ sinh và kế hoạch (lấy từ eval/synth_spec.py)
README.md của thư mục này viết tay (kế hoạch giai đoạn 1), script không ghi đè.
Chỉ đọc data/train (luật 6). Cần pandas==2.2.3.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import io
import json
import re
import sqlite3
import statistics
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

import synth_spec as spec  # noqa: E402

DATA = ROOT / "data" / "train"
MONEY_ROLES = {"amount", "price", "cost"}
N_RAW = 400  # số dòng dữ liệu thô đọc mỗi bảng để đếm định dạng


# ---------------------------------------------------------------- đọc thô
def raw_tables(path: Path) -> list[tuple[str, list[list]]]:
    """[(tên nguồn, các dòng thô)] của một file; mỗi sheet/bảng là một nguồn. Chỉ đọc phần đầu."""
    out = []
    if path.suffix == ".csv":
        raw = path.read_bytes()[:400_000].decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(raw)))[: N_RAW + 20]
        out.append((path.name, rows))
    elif path.suffix in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sh in wb.sheetnames:
            rows = []
            for i, r in enumerate(wb[sh].iter_rows(values_only=True)):
                rows.append(list(r))
                if i >= N_RAW + 20:
                    break
            out.append((f"{path.name}[{sh}]", rows))
    elif path.suffix == ".sqlite":
        con = sqlite3.connect(path)
        for (t,) in con.execute("select name from sqlite_master where type='table'"):
            cur = con.execute(f'select * from "{t}" limit {N_RAW}')
            out.append((f"{path.name}[{t}]", [[d[0] for d in cur.description]] + [list(r) for r in cur.fetchall()]))
        con.close()
    return out


def header_index(rows: list[list], names: set[str]) -> int | None:
    from datamate.clean import norm_col
    want = {norm_col(n) for n in names}
    for i, r in enumerate(rows[:15]):
        cells = {norm_col(str(c)) for c in r if c not in (None, "")}
        if cells and len(cells & want) >= max(2, 0.6 * len(cells)):
            return i
    return None


# ---------------------------------------------------------------- phân loại giá trị
MONEY_PATTERNS = [("so_nguyen", r"^-?\d+$"), ("so_thuc", r"^-?\d+\.\d+$"), ("phay_nghin", r"^-?\d{1,3}(,\d{3})+$"),
                  ("phay_nghin_.0000", r"^-?\d{1,3}(,\d{3})*\.\d{2,4}$"), ("cham_nghin_d", r"^-?\d{1,3}(\.\d{3})+ ?(đ|d|vnđ|vnd)$"),
                  ("cham_nghin", r"^-?\d{1,3}(\.\d{3})+$"), ("k", r"^-?\d+([.,]\d+)?k$"), ("tr_lien", r"^-?\d+tr\d+$"),
                  ("tr_thap_phan", r"^-?\d+[.,]\d+tr$"), ("tr", r"^-?\d+tr$"), ("so_d", r"^-?\d+ ?(đ|d)$")]
DATE_PATTERNS = [("excel_serial", r"^\d{5}(\.0)?$"), ("dd/mm/yyyy", r"^\d{2}/\d{2}/\d{4}$"), ("d/m/yyyy", r"^\d{1,2}/\d{1,2}/\d{4}$"),
                 ("d/m", r"^\d{1,2}/\d{1,2}$"), ("d thg m yyyy", r"^\d{1,2} thg \d{1,2} \d{4}$"), ("datetime", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?$"),
                 ("yyyy-mm-dd", r"^\d{4}-\d{2}-\d{2}$"), ("dd-mm-yyyy", r"^\d{1,2}-\d{1,2}-\d{4}$"), ("dd.mm.yyyy", r"^\d{1,2}\.\d{1,2}\.\d{4}$")]
QTY_PATTERNS = [("so", r"^-?\d+(\.0)?$"), ("so_kem_don_vi", r"^-?\d+([.,]\d+)? ?[^\d\s].*$"), ("so_thuc", r"^-?\d+[.,]\d+$")]
TIME_PATTERNS = [("HH:MM", r"^\d{1,2}:\d{2}$"), ("HH:MM:SS", r"^\d{1,2}:\d{2}:\d{2}$"), ("datetime", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?$")]


def kind(v, patterns) -> str | None:
    if v is None or v == "":
        return None
    if isinstance(v, dt.datetime):
        return "datetime" if (v.hour or v.minute or v.second) else "excel_date"
    if isinstance(v, dt.date):
        return "excel_date"
    if isinstance(v, dt.time):
        return "excel_time"
    s = str(v).strip().lower()
    if isinstance(v, float) and v.is_integer():
        s = str(int(v))
    for name, p in patterns:
        if re.match(p, s):
            return name
    return "khac"


# ---------------------------------------------------------------- thống kê hiệu chỉnh
def calibration(biz, s) -> dict:
    import pandas as pd
    from datamate import dm
    L = s.lines.copy()
    L = L[L["date"].notna()]
    agg = {"date": "min", "status": "first", "amount": "sum"}
    for c in ("cust", "time", "channel", "pay", "staff"):
        if c in L:
            agg[c] = "first"
    O = L.groupby("order").agg(agg)
    O["n_lines"] = L.groupby("order").size()
    days = pd.date_range(L["date"].min().normalize(), L["date"].max().normalize())
    per_day = O.groupby(O["date"].dt.normalize()).size().reindex(days, fill_value=0)
    mean = float(per_day.mean())
    wd = per_day.groupby(per_day.index.weekday).mean() / mean
    mo = per_day.groupby(per_day.index.month).mean() / mean
    tet = []
    for y in sorted({d.year for d in days}):
        try:
            p = dm.tet(y)
        except Exception:  # noqa: BLE001
            continue
        win = per_day[(per_day.index >= pd.Timestamp(p.start)) & (per_day.index <= pd.Timestamp(p.end))]
        n = (p.end - p.start).days + 1
        before = per_day[(per_day.index < pd.Timestamp(p.start)) & (per_day.index >= pd.Timestamp(p.start) - pd.Timedelta(days=n))]
        if len(win) == n and len(before) == n and before.mean() > 0:
            tet.append(round(float(win.mean() / before.mean()), 3))
    st = O["status"].value_counts(normalize=True).round(4).to_dict()
    out = {"ngay_dau": str(days[0].date()), "ngay_cuoi": str(days[-1].date()), "so_ngay": len(days), "so_don": int(len(O)), "so_dong": int(len(L)),
           "don_moi_ngay": round(mean, 2), "don_moi_ngay_p10_p90": [float(per_day.quantile(.1)), float(per_day.quantile(.9))],
           "he_so_thu": {str(k): round(float(v), 3) for k, v in wd.items()}, "he_so_thang": {str(k): round(float(v), 3) for k, v in mo.items()},
           "he_so_tet_so_voi_truoc_tet": tet,
           "xu_huong_3_thang_cuoi_so_dau": round(float(per_day.iloc[-90:].mean() / max(per_day.iloc[:90].mean(), 1e-9)), 3),
           "dong_moi_don": {"tb": round(float(O["n_lines"].mean()), 2), "p50": float(O["n_lines"].median()), "p90": float(O["n_lines"].quantile(.9)), "max": int(O["n_lines"].max())},
           "gia_tri_don": {q: round(float(O["amount"].quantile(v))) for q, v in (("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99))},
           "so_luong_dong_p50_p90": [float(L["qty_base"].quantile(.5)), float(L["qty_base"].quantile(.9))] if "qty_base" in L else None,
           "ti_le_trang_thai_don": {str(k): v for k, v in st.items()}}
    if "cust" in O:
        walk = O["cust"].isna() | O["cust"].astype(str).map(dm.is_walkin)
        named = O.loc[~walk, "cust"].astype(str)
        per_c = named.value_counts()
        out |= {"ti_le_don_khach_le": round(float(walk.mean()), 4), "so_khach": int(per_c.size),
                "don_moi_khach": {"tb": round(float(per_c.mean()), 2) if len(per_c) else None, "p90": float(per_c.quantile(.9)) if len(per_c) else None},
                "ti_le_khach_quay_lai": round(float((per_c >= 2).mean()), 4) if len(per_c) else None}
    if "item" in L:
        ok = L[L["status"].isin(["hoan_thanh", "dang_giao"])]
        rev = ok.groupby("item")["amount"].sum().sort_values(ascending=False)
        k = max(1, len(rev) // 10)
        out |= {"so_mat_hang_ban": int(rev.size), "ti_trong_doanh_thu_top10pct_mat_hang": round(float(rev.iloc[:k].sum() / max(rev.sum(), 1)), 4),
                "ti_trong_mat_hang_dau": round(float(rev.iloc[0] / max(rev.sum(), 1)), 4) if len(rev) else None}
        if "price" in L:
            npx = L.groupby("item")["price"].nunique()
            out["so_muc_gia_moi_mat_hang"] = {"p50": float(npx.median()), "p90": float(npx.quantile(.9))}
    if "time" in L:
        hrs = pd.to_datetime(O["time"].astype(str), errors="coerce").dt.hour
        if hrs.notna().mean() < .5:
            hrs = pd.to_numeric(O["time"].astype(str).str.extract(r"(\d{1,2}):")[0], errors="coerce")
        if hrs.notna().any():
            out["phan_bo_gio"] = {str(int(h)): round(float(v), 4) for h, v in hrs.value_counts(normalize=True).sort_index().items()}
    for c, name in (("channel", "kenh"), ("pay", "thanh_toan"), ("staff", "nhan_vien")):
        if c in O:
            out[f"ti_le_{name}"] = {str(k): round(float(v), 4) for k, v in O[c].astype(str).value_counts(normalize=True).head(8).items()}
    if "group" in L:
        out["so_nhom_hang"] = int(L["group"].nunique())
    return out


STRUCT_FEATURES = {
    "quy_doi": r"quy.?đổi|quydoi|đơn vị cơ bản|đổi ra|donvitinh|đơn vị tính|\bdvt\b", "lo": r"\blô\b|_lo\b|theolo|hạn dùng|\bhsd\b",
    "cong_no": r"công nợ|cong_no|congno|hạn nợ|hanno|han_no|\bnợ\b", "giao": r"giao", "nhap": r"nhập|nhap",
    "ton": r"tồn|ton_kho|tonkho", "tra": r"trả|tra_hang|trahang|hoàn", "thanh_toan": r"thanh.?toán|thanh_toan|soquy|sổ quỹ"}


def shop_structure(d: Path) -> dict:
    """Độ phức tạp cấu trúc của một shop, đo trên file thô: số file, số bảng/sheet, số cột, các loại thông tin có ghi."""
    files = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls", ".sqlite", ".db")]
    tabs = cols = 0
    text = " ".join(p.name.lower() for p in files)
    for p in files:
        try:
            for name, rows in raw_tables(p):
                tabs += 1
                cols += max((len([c for c in r if c not in (None, "")]) for r in rows[:6]), default=0)
                text += " " + name.lower() + " " + " ".join(str(c) for r in rows[:6] for c in r if c is not None)[:3000].lower()
        except Exception:  # noqa: BLE001
            pass
    tier = "don_gian" if tabs <= 2 else "vua" if tabs <= 9 else "phuc_tap"
    return {"files": len(files), "tables": tabs, "cols": cols, "tang": tier,
            "co_ghi": [k for k, v in STRUCT_FEATURES.items() if re.search(v, text)]}


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def shop_complexity(structs: dict, errors_dir: Path) -> dict:
    """Cấu trúc từng shop train + kết quả các lượt log có đáp án (errors/<lượt>/by_shop.json) + tương quan số bảng ↔ PA."""
    runs = {}
    for f in sorted(errors_dir.glob("*/by_shop.json")):
        bs = json.loads(f.read_text(encoding="utf-8"))
        shops = [b for b in bs if b in structs]
        runs[f.parent.name] = {
            "so_shop": len(shops),
            "spearman_so_bang_PA": spearman([structs[b]["tables"] for b in shops], [bs[b]["PA"] for b in shops]) if len(shops) >= 5 else None,
            "spearman_so_bang_ti_le_bo": spearman([structs[b]["tables"] for b in shops], [bs[b]["bo"] / bs[b]["n"] for b in shops]) if len(shops) >= 5 else None,
            "theo_tang": {t: {"shop": [b for b in shops if structs[b]["tang"] == t],
                              "PA_tb": round(statistics.mean([bs[b]["PA"] for b in shops if structs[b]["tang"] == t]), 3)}
                          for t in ("don_gian", "vua", "phuc_tap") if any(structs[b]["tang"] == t for b in shops)}}
        for b in shops:
            structs[b].setdefault("ket_qua_log", {})[f.parent.name] = {k: bs[b][k] for k in ("n", "dung", "sai", "bo", "PA")}
    tiers = collections.Counter(s["tang"] for s in structs.values())
    return {"dinh_nghia_tang": "đơn giản ≤ 2 bảng/sheet thô; vừa 3–9; phức tạp ≥ 10", "so_shop_theo_tang": dict(tiers),
            "theo_luot_log": runs, "theo_shop": structs}


def industry_summary(rows: list[dict]) -> dict:
    keys = ["don_moi_ngay", "ti_le_don_khach_le", "ti_le_khach_quay_lai", "ti_trong_doanh_thu_top10pct_mat_hang", "so_mat_hang_ban", "so_ngay"]
    out = {}
    for k in keys:
        xs = [r[k] for r in rows if r.get(k) is not None]
        if xs:
            out[k] = {"min": min(xs), "trung_vi": statistics.median(xs), "max": max(xs)}
    for k in ("dong_moi_don", "gia_tri_don"):
        out[k] = {q: statistics.median(r[k][q] for r in rows if r.get(k)) for q in rows[0][k]} if rows and rows[0].get(k) else None
    st = collections.defaultdict(list)
    for r in rows:
        for s, v in r["ti_le_trang_thai_don"].items():
            st[s].append(v)
    out["ti_le_trang_thai_don"] = {s: round(statistics.median(v), 4) for s, v in st.items()}
    for k in ("he_so_thu", "he_so_thang"):
        acc = collections.defaultdict(list)
        for r in rows:
            for x, v in r[k].items():
                acc[x].append(v)
        out[k] = {x: round(statistics.median(v), 3) for x, v in sorted(acc.items(), key=lambda kv: int(kv[0]))}
    out["he_so_tet"] = sorted(x for r in rows for x in r["he_so_tet_so_voi_truoc_tet"])
    return out


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/synth_source/data_spec"))
    ap.add_argument("--cache", default=str(ROOT / "outputs/cache"))
    a = ap.parse_args(argv)
    from datamate import dm, loader, shop as shop_mod
    from datamate.clean import norm_col
    from datamate.timeutil import LunarCalendar
    dm.cal = LunarCalendar(DATA / "calendar.csv")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    arche = collections.defaultdict(lambda: {"mo_ta": "", "shops": [], "files": collections.Counter(), "sheet_names": [], "ho_file": collections.Counter()})
    colnames = collections.defaultdict(collections.Counter)
    extra_cols = collections.Counter()
    fmt = {k: collections.defaultdict(collections.Counter) for k in ("tien", "ngay", "gio", "so_luong")}
    status_vocab, walkin_vocab = collections.defaultdict(collections.Counter), collections.Counter()
    header_info = []
    quirks = collections.defaultdict(set)
    readme_ph = collections.defaultdict(collections.Counter)
    readme_sections = collections.Counter()
    calib, structs = {}, {}
    descs = dict((c, d) for c, _, d in spec.ARCHETYPES)

    for d in sorted(p for p in (DATA / "data").iterdir() if p.is_dir()):
        b = d.name
        biz = loader.load_cached(d, Path(a.cache))
        roles = shop_mod._table_roles(biz)
        s = shop_mod.build(biz)
        ar = spec.archetype_of(biz.readme)
        ind = spec.industry_of(s.items if s is not None else [])
        A = arche[ar]
        A["mo_ta"] = descs.get(ar, "")
        A["shops"].append({"business_id": b, "industry": ind})
        # tên cột theo vai trò
        for t, df in biz.tables.items():
            inv = {c: r for r, c in roles.get(t, {}).items()}
            for c in df.columns:
                c = str(c)
                if c.startswith("_") or "__" in c:
                    continue
                if c in inv:
                    colnames[inv[c]][c] += 1
                else:
                    extra_cols[c] += 1
        # bẫy loader
        for t, ns in (biz.notes or {}).items():
            for n in ns:
                for key, pat in (("dong_ten_bao_cao_dau_sheet", r"tên báo cáo ở đầu"), ("dong_ten_bao_cao_giua_sheet", r"tên báo cáo / kỳ"),
                                 ("tieu_de_lap_lai", r"tiêu đề lặp lại"), ("dong_tong_cong", r"dòng tổng cộng"), ("dong_cong_theo_ngay", r"cộng theo ngày"),
                                 ("so_luong_kem_don_vi", r"dạng '"), ("o_trong_dong_sau_cua_don", r"dòng sau của cùng một đơn để trống"),
                                 ("dong_phu_thieu_cot", r"dòng phụ của đơn để trống"), ("dong_trung_cung_don", r"giống hệt dòng khác"),
                                 ("ten_viet_khac_nhau", r"viết khác"), ("gop_nhieu_sheet_file", r"gộp \d+ sheet/file")):
                    if re.search(pat, n):
                        quirks[key].add(b)
        # file thô: họ file, sheet, dòng trước tiêu đề, định dạng giá trị
        for f in sorted(x for x in d.iterdir() if x.name != "README.md"):
            fam = re.sub(r"(_\d+|_\d{4})$", "", f.stem)
            A["ho_file"][fam + f.suffix] += 1
            A["files"][f.suffix] += 1
            if fam != f.stem:
                quirks["tach_file_theo_phan_hoac_nam"].add(b)
            for src, rows in raw_tables(f):
                m = re.search(r"\[(.+)\]$", src)
                if m and f.suffix != ".sqlite":
                    A["sheet_names"].append(m.group(1))
                best = max(biz.tables.items(), key=lambda kv: len({norm_col(str(c)) for c in kv[1].columns} & {norm_col(str(x)) for r in rows[:15] for x in r if x}), default=None)
                if best is None:
                    continue
                t, df = best
                hi = header_index(rows, {str(c) for c in df.columns})
                if hi is None:
                    continue
                pre = [r for r in rows[:hi] if any(x not in (None, "") for x in r)]
                group_hdr = bool(pre) and sum(x not in (None, "") for x in pre[-1]) >= 2
                header_info.append({"business_id": b, "nguon": src, "so_dong_truoc_tieu_de": hi, "tieu_de_hai_tang": group_hdr,
                                    "dong_truoc": [[str(x) for x in r if x not in (None, "")][:4] for r in pre][:3]})
                if hi > 0 and not group_hdr:
                    quirks["dong_tieu_de_bao_cao"].add(b)
                if group_hdr:
                    quirks["tieu_de_hai_tang"].add(b)
                inv = {norm_col(c): r for r, c in roles.get(t, {}).items()}
                hdr = [norm_col(str(x)) if x is not None else "" for x in rows[hi]]
                for row in rows[hi + 1:]:
                    for j, v in enumerate(row[:len(hdr)]):
                        r = inv.get(hdr[j])
                        if r is None or v in (None, ""):
                            continue
                        if r in MONEY_ROLES:
                            fmt["tien"][b][kind(v, MONEY_PATTERNS)] += 1
                        elif r == "date":
                            fmt["ngay"][b][kind(v, DATE_PATTERNS)] += 1
                        elif r == "time":
                            fmt["gio"][b][kind(v, TIME_PATTERNS)] += 1
                        elif r in ("qty", "qty_base"):
                            fmt["so_luong"][b][kind(v, QTY_PATTERNS)] += 1
                        elif r == "status":
                            status_vocab[dm.norm_status(str(v)) or "?"][str(v).strip()] += 1
                        elif r == "cust" and dm.is_walkin(str(v)):
                            walkin_vocab[str(v).strip()] += 1
        # README
        sec = None
        for line in biz.readme.splitlines():
            if line.startswith("## "):
                sec = re.sub(r"`[^`]*`", "`X`", line[3:].strip())
                readme_sections[sec] += 1
            m = re.match(r"^- \*\*(.+?)\*\*: (.+)$", line)
            if m and sec and sec.startswith("Cột trong"):
                col, ph = m.group(1), m.group(2)
                r = next((rr for t in roles for rr, c in roles[t].items() if str(c) == col), None)
                readme_ph[r or "khong_vai_tro"][re.sub(r"\d[\d.,]*", "N", ph)[:120]] += 1
        structs[b] = {"industry": ind, "archetype": ar, **shop_structure(d)}
        if s is not None:
            try:
                calib[b] = {"industry": ind, "archetype": ar, **calibration(biz, s)}
            except Exception as e:  # noqa: BLE001
                calib[b] = {"industry": ind, "archetype": ar, "loi": repr(e)[:200]}

    def dump(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")

    dump("archetypes.json", {k: {"mo_ta": v["mo_ta"], "shops": v["shops"], "dinh_dang_file": dict(v["files"]), "ho_file": dict(v["ho_file"].most_common()),
                                 "ten_sheet_mau": sorted(set(v["sheet_names"]))[:12]} for k, v in sorted(arche.items())})
    dump("column_names.json", {"theo_vai_tro": {r: dict(c.most_common()) for r, c in sorted(colnames.items())},
                               "cot_khong_vai_tro": dict(extra_cols.most_common(120)), "header": header_info})
    dump("value_formats.json", {
        "tien": {b: dict(c.most_common()) for b, c in sorted(fmt["tien"].items())}, "ngay": {b: dict(c.most_common()) for b, c in sorted(fmt["ngay"].items())},
        "gio": {b: dict(c.most_common()) for b, c in sorted(fmt["gio"].items())}, "so_luong": {b: dict(c.most_common()) for b, c in sorted(fmt["so_luong"].items())},
        "tong_hop": {k: dict(sum(fmt[k].values(), collections.Counter()).most_common()) for k in fmt},
        "trang_thai": {k: dict(v.most_common(30)) for k, v in sorted(status_vocab.items())}, "khach_le": dict(walkin_vocab.most_common(30))})
    dump("quirks.json", {k: {"so_shop": len(v), "shops": sorted(v)} for k, v in sorted(quirks.items(), key=lambda kv: -len(kv[1]))})
    dump("readme_phrases.json", {"cac_muc": dict(readme_sections.most_common()), "theo_vai_tro": {r: dict(c.most_common(12)) for r, c in sorted(readme_ph.items())}})
    by_ind = collections.defaultdict(list)
    for b, c in calib.items():
        if "loi" not in c:
            by_ind[c["industry"]].append(c)
    dump("calibration.json", {"theo_nganh": {k: industry_summary(v) for k, v in sorted(by_ind.items())}, "theo_shop": calib})
    dump("canonical_schema.json", spec.CANONICAL_SCHEMA)
    dump("catalog_schema.json", spec.CATALOG_SCHEMA)
    dump("render_recipes.json", spec.RENDER_RECIPES)
    for b, st in structs.items():
        st["bay"] = sorted(k for k, v in quirks.items() if b in v)
    cx = shop_complexity(structs, out.parent / "errors")
    dump("shop_complexity.json", cx)
    dump("new_industries.json", spec.NEW_INDUSTRIES)
    dump("volume_plan.json", spec.VOLUME_PLAN)
    dump("model_routing.json", spec.MODEL_ROUTING)
    print(json.dumps({"archetypes": len(arche), "vai_tro_cot": len(colnames), "bay": {k: len(v) for k, v in quirks.items()},
                      "calib_shops": len(calib), "nganh": {k: len(v) for k, v in by_ind.items()}, "tang": cx["so_shop_theo_tang"],
                      "log": {k: {kk: v[kk] for kk in ("spearman_so_bang_PA", "spearman_so_bang_ti_le_bo")} for k, v in cx["theo_luot_log"].items()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
