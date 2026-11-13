"""Lớp tính toán theo rulebook: đưa dữ liệu mỗi shop về một khung chung, rồi tính các số đo cơ bản.

README mỗi shop ghi vai trò từng cột ("- **Mã đơn hàng**: mã đơn"), nên không cần đoán tên cột. Model viết pandas tự diễn giải
luật nên hay sai theo cùng một kiểu (đếm dòng thay vì đơn, quên bỏ cả đơn có trả hàng). Các hàm ở đây chỉ viết một lần và
được đo trên đáp án train (eval/template_eval.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from .loader import _id_column
from .clean import is_walkin, norm_col, norm_key, norm_status, norm_text, parse_time

# vai trò → mẫu so với phần mô tả cột trong README (đã bỏ dấu, chữ thường)
_ROLES = [
    ("order", r"^ma don\b"),
    ("date", r"^ngay ban\b|^ngay dat\b|^thoi diem dat don|^ngay gio dat"),
    ("status", r"^trang thai don\b"),
    ("item", r"^ten hang\b"),
    ("item_code", r"^ma hang\b"),
    ("qty_base", r"^so luong da doi ra don vi co ban"),
    ("qty", r"^so luong( ghi theo don vi| ban)?\b"),
    ("amount", r"^thanh tien sau giam"),
    ("group", r"^nhom hang\b"),
    ("cust", r"^ten khach\b"),
    ("cust_code", r"^ma khach\b"),
    ("returned_qty", r"^so luong khach tra lai"),
    ("note", r"^ghi chu\b"),
    ("unit", r"^don vi tinh\b"),
    ("big_unit", r"^don vi lon\b"),
    ("factor", r"^he so\b"),
    ("time", r"^thoi diem dat don|^gio dat don|^gio ban|^gio tao"),
    ("channel", r"^kenh ban"),
    ("pay", r"^hinh thuc thanh toan"),
    ("cust_group", r"^nhom khach"),
    ("rx", r"^thuoc ke don"),
    ("price", r"^don gia\b|^gia ban\b"),
    ("cost", r"^gia von\b"),
    ("staff", r"^(ma )?nhan vien\b|^nguoi ban\b"),
]


def parse_readme(readme: str) -> dict[str, dict[str, str]]:
    """{tên file không đuôi: {tên cột: vai trò}}. Mỗi vai trò lấy cột đầu tiên khớp."""
    out: dict[str, dict[str, str]] = {}
    cur = None
    for line in readme.splitlines():
        m = re.match(r"^#+\s*Cột trong\s+`([^`]+)`", line)
        if m:
            cur = out.setdefault(re.sub(r"\.\w+$", "", m.group(1)), {})
            continue
        if line.startswith("#"):
            cur = None
            continue
        m = re.match(r"^- \*\*(.+?)\*\*:\s*(.+)$", line)
        if m and cur is not None:
            desc = norm_key(m.group(2))
            for role, pat in _ROLES:
                if re.search(pat, desc) and role not in cur.values():
                    cur[m.group(1).strip()] = role
                    break
    return out


# README kiểu "File máy POS xuất ra" không mô tả cột; khi đó nhận theo tên cột (không dấu, '_' thay khoảng trắng)
_NAME_HINTS = {
    'order': ('ma_hoa_don', 'ma_don_hang', 'ma_don', 'so_hd', 'so_ct', 'ma_bill', 'ma_giao_dich', 'order_id', 'madonhang', 'mahoadon', 'madon', 'ma_chung_tu'),
    'date': ('ngay_ban', 'ngay_lap', 'thoi_gian', 'ngay_tao', 'ngay_mua', 'ngay_giao_dich', 'ngay_thang', 'ngayban', 'ngaylap', 'ngay'),
    'status': ('trang_thai', 'trang_thai_don', 'trang_thai_xu_ly', 'tinh_trang', 'tinhtrang', 'xu_ly', 'status', 'trangthai', 'ghi_chu_tt'),
    'item': ('ten_hang', 'ten_hang_hoa', 'ten_san_pham', 'ten_sp', 'mat_hang', 'hang_hoa', 'ten_mon', 'mon', 'hang', 'tenhang', 'tenhanghoa', 'tengoi', 'san_pham', 'noi_dung'),
    'item_code': ('ma_sku', 'ma_hang', 'ma_hang_hoa', 'ma_sp', 'ma_san_pham', 'ma_mon', 'mahanghoa', 'id_sp', 'mahang', 'masp'),
    'qty_base': ('so_luong_quy_doi', 'sl_quy_doi', 'sl_co_ban', 'slquydoi', 'sl_le', 'so_luong_le', 'sl_quy_ra_le'),
    'qty': ('so_luong', 'sl', 's_l', 'slg', 'soluong', 'so_luong_ban', 'sl_ban', 'sluong'),
    'amount': ('thanh_tien', 'thanh_tien_d', 't_tien', 'trigia', 'tri_gia', 'tien'),
    'group': ('nhom_hang', 'nhom_hang_hoa', 'nhom', 'phan_loai', 'danhmuc', 'danh_muc', 'nhomhang', 'nhom_hang_3_cap', 'loai_san_pham'),
    'cust': ('ten_khach', 'khach_hang', 'ten_khach_hang', 'khach', 'khach_hang_ten', 'ten_kh'),
    'returned_qty': ('so_luong_tra',),
    'time': ('gio_tao_don', 'gio', 'thoi_gian_tao', 'thoi_gian', 'thoigianlap', 'thoi_diem'),
    'channel': ('kenh_ban', 'kenhbanhang', 'nguon_don', 'ban_qua'),
    'pay': ('phuong_thuc_tt', 'phuong_thuc_thanh_toan', 'pttt', 'hinh_thuc_thanh_toan', 'phuongthuctt', 'thanh_toan'),
    'cust_group': ('nhom_khach_hang', 'doi_tuong_kh', 'loaikhachhang', 'loai_khach_hang', 'khach_le_si', 'hang'),
    'cust_code': ('ma_kh', 'ma_khach_hang', 'makhachhang', 'ma_kh_mua', 'so_kh'),
    'rx': ('ke_don', 'thuoc_ke_don', 'kedon', 'bantheodon', 'ban_theo_don'),
    'price': ('don_gia', 'don_gia_d', 'don_gia_ban', 'gia_ban', 'gia_ban_le', 'gia_don_vi', 'giadonvi', 'gia_san_pham', 'gia_1_dv', 'gia_le', 'gia_niem_yet', 'price', 'gia'),
    'cost': ('gia_von', 'gia_von_luc_ban', 'giavonlucban'),
    'staff': ('ma_nv_ban', 'nguoi_ban', 'ma_nhan_vien', 'manhanvien'),
    'note': ('ghi_chu', 'ghichu'),
    'unit': ('don_vi_tinh', 'dvt', 'don_vi', 'd_vi', 'donvi'),
    'big_unit': ('dv_lon', 'don_vi_quy_doi'),
    'factor': ('he_so', 'he_so_qd', 'he_so_quy_doi', 'quy_doi', 'bang', 'ti_le_quy_doi'),
}


def _fill_by_name(biz, out: dict) -> None:
    for t, df in biz.tables.items():
        cols = {norm_col(c): c for c in df.columns if not c.startswith("_") and "__" not in c}
        cols.update({k.replace("_", ""): v for k, v in list(cols.items())})  # 'TenHang', 'ten_hang' là một
        rr = out.setdefault(t, {})
        for role, hints in _NAME_HINTS.items():
            if role in rr:
                continue
            hit = next((cols[h.replace("_", "")] for h in hints if h.replace("_", "") in cols), None)
            if hit is not None and hit not in rr.values():
                rr[role] = hit
        if not rr:
            del out[t]


def _numeric(df, c):
    return pd.to_numeric(df[c], errors="coerce") if not str(df[c].dtype).startswith("datetime") else pd.Series(float("nan"), index=df.index)


def _infer_by_values(biz, out: dict) -> None:
    """Tên cột lạ: nhận vai trò từ giá trị. Chỉ chạy khi không bảng nào có cột tiền; số lượng × đơn giá phải khớp thành tiền ở 90% số dòng."""
    if any("amount" in r for r in out.values()):
        return
    best = None
    for t, df in biz.tables.items():
        cols = [c for c in df.columns if not c.startswith("_") and "__" not in c]
        nums = [c for c in cols if _numeric(df, c).notna().mean() > 0.9]
        for total in nums:
            for a in nums:
                for b in nums:
                    if not (a < b) or total in (a, b):
                        continue
                    va, vb, vt = _numeric(df, a), _numeric(df, b), _numeric(df, total)
                    if vt.abs().median() < 1000 or ((va * vb - vt).abs() <= vt.abs() * 0.3 + 1).mean() < 0.9:
                        continue
                    if best is None or len(df) > best[1]:
                        qty, price = (a, b) if va.median() <= vb.median() else (b, a)
                        best = (t, len(df), total, qty, price)
    if best is None:
        return
    t, _, total, qty, price = best
    df = biz.tables[t]
    r = out.setdefault(t, {})
    r.update(amount=total, qty=qty, price=price)
    cols = [c for c in df.columns if not c.startswith("_") and "__" not in c and c not in (total, qty, price)]
    texts = [c for c in cols if not str(df[c].dtype).startswith(("datetime", "Int", "int", "float"))]
    for c in cols:
        if str(df[c].dtype).startswith("datetime") and "date" not in r:
            r["date"] = c
    st = [c for c in texts if df[c].dropna().map(norm_status).ne("").mean() > 0.7]
    if st:
        r["status"] = st[0]
    ids = [c for c in texts if 1 < len(df) / max(df[c].nunique(), 1) < 50 and df[c].dropna().astype(str).str.fullmatch(r"[A-Za-z_-]*\d{2,}").mean() > 0.9]
    if ids:
        r["order"] = ids[0]
    names = [c for c in texts if c not in st and c not in ids and 5 <= df[c].nunique() <= 2000 and df[c].dropna().astype(str).str.len().mean() > 6]
    if names:
        r["item"] = max(names, key=lambda c: df[c].astype(str).str.len().mean())
    # đầu đơn ở bảng khác: ngày, trạng thái
    if "order" in r:
        keys = set(df[r["order"]].dropna().astype(str))
        for k, d in biz.tables.items():
            if k == t:
                continue
            for c in d.columns:
                if "__" not in c and not c.startswith("_") and d[c].astype(str).isin(keys).mean() > 0.8:
                    h = out.setdefault(k, {})
                    h["order"] = c
                    for x in d.columns:
                        if str(d[x].dtype).startswith("datetime") and "date" not in h:
                            h["date"] = x
                        elif x != c and "__" not in x and not x.startswith("_") and "status" not in h and d[x].dropna().map(norm_status).ne("").mean() > 0.7:
                            h["status"] = x
                    break


def _table_roles(biz) -> dict[str, dict[str, str]]:
    """{tên bảng đã nạp: {vai trò: cột}}, ghép mục README với bảng có đủ các cột đó; thiếu thì theo tên cột."""
    out = {}
    for _, cols in parse_readme(biz.readme).items():
        if not cols:
            continue
        best, n = None, 0
        for t, df in biz.tables.items():
            hit = sum(c in df.columns for c in cols)
            if hit > n:
                best, n = t, hit
        if best is None or n < 0.6 * len(cols):
            continue
        out.setdefault(best, {}).update({r: c for c, r in cols.items() if c in biz.tables[best].columns})
    _fill_by_name(biz, out)
    _infer_by_values(biz, out)
    # hợp đồng dữ liệu do model lập (contract.py), đã kiểm tra trên giá trị: chỉ điền chỗ còn thiếu
    for t, roles in (getattr(biz, "extra_roles", None) or {}).items():
        if t in biz.tables:
            cur = out.setdefault(t, {})
            for role, col in roles.items():
                if role not in cur and col not in cur.values():
                    cur[role] = col
    return out


def round_vnd(x) -> int:
    return int(Decimal(repr(float(x))).quantize(Decimal("1e-6")).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def round2(x) -> float:
    return float(Decimal(repr(float(x))).quantize(Decimal("1e-9")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass
class Shop:
    business_id: str
    lines: pd.DataFrame            # order, date, status, item, group, qty_base, amount, cust
    items: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    has_group: bool = False
    reliable: bool = True          # có cột trạng thái thật; suy từ ghi chú tự do dễ nhầm ('trả đủ' không phải trả hàng) thì không tự trả lời
    has_orders: bool = True        # sổ không có mã đơn: chỉ cộng được tiền và số lượng, không đếm được đơn
    biz: object = None             # bảng gốc, cho các số đo ngoài bảng bán (nhập hàng, giao hàng, tồn)

    # ---------------------------------------------------- bộ lọc
    def in_period(self, p) -> pd.DataFrame:
        return self.lines[p.mask(self.lines["date"])]

    def counted(self, p) -> pd.DataFrame:
        """Dòng của các đơn tính vào doanh thu (rulebook §1): bỏ cả đơn huỷ, đơn có trả hàng; đơn đang giao vẫn tính."""
        x = self.in_period(p)
        return x[x["status"].isin(["hoan_thanh", "dang_giao"])]

    # ---------------------------------------------------- số đo
    def revenue(self, p, item=None, group=None) -> int:
        return round_vnd(self._filter(self.counted(p), item, group)["amount"].sum())

    def n_orders(self, p, item=None, group=None) -> int:
        return int(self._filter(self.counted(p), item, group)["order"].nunique())

    def n_cancelled(self, p) -> int:
        x = self.in_period(p)
        return int(x[x["status"] == "huy"]["order"].nunique())

    def cancel_rate(self, p) -> float:
        x = self.in_period(p)
        n = x["order"].nunique()
        if n == 0:
            raise ZeroDivisionError("kỳ không có đơn")
        return round2(x[x["status"] == "huy"]["order"].nunique() / n * 100)

    def avg_order(self, p) -> int:
        n = self.n_orders(p)
        if n == 0:
            raise ZeroDivisionError("kỳ không có đơn")
        return round_vnd(self.counted(p)["amount"].sum() / n)

    def qty(self, p, item=None, group=None) -> float:
        return float(self._filter(self.counted(p), item, group)["qty_base"].sum())

    def share_group(self, p, group) -> float:
        total = self.counted(p)["amount"].sum()
        if total == 0:
            raise ZeroDivisionError("kỳ không có doanh thu")
        return round2(self._filter(self.counted(p), None, group)["amount"].sum() / total * 100)

    def rank_items(self, p, group=None) -> list:
        """Mặt hàng theo số lượng quy về đơn vị cơ bản giảm dần; bằng nhau thì theo mã tăng dần (không có mã thì theo tên)."""
        x = self._filter(self.counted(p), None, group)
        x = x[~x["item"].map(lambda v: norm_key(v) == "phi phuc vu")]  # không phải mặt hàng (rulebook §3)
        key = "item_code" if "item_code" in x else "item"
        if x["qty_base"].sum() == 0:  # shop không đọc được số lượng: không có thứ tự để xếp
            return []
        g = x.groupby(["item", key] if key != "item" else ["item"], as_index=False)["qty_base"].sum()
        g = g.sort_values(["qty_base", key], ascending=[False, True], kind="mergesort")
        return g["item"].astype(str).tolist()

    def by_shift(self, p, shift) -> pd.DataFrame:
        x = self.counted(p)
        return x[x["hour"].between(6, 13)] if shift == "sang" else x[x["hour"].between(14, 21)]

    def customers(self, p, by="money", x=None) -> list:
        """Khách theo tiền, số đơn, số ngày có mua hoặc tiền trung bình một đơn, giảm dần; bằng nhau theo mã. Có mã khách thì mỗi mã một khách, đơn không mã không phải khách."""
        x = self.counted(p) if x is None else x
        key = "cust_code" if "cust_code" in x and x["cust_code"].notna().any() else "cust"
        if key not in x:
            return pd.DataFrame(columns=["_k", "v", "name"])
        x = x[x[key].notna()]
        if key == "cust":
            x = x[~x["cust"].map(is_walkin)]
            x = x.assign(_k=x["cust"].map(norm_text))
        else:
            x = x.assign(_k=x[key])
        if by == "money":
            v = x.groupby("_k")["amount"].sum()
        elif by == "days":
            v = x.groupby("_k")["date"].apply(lambda d: d.dt.date.nunique())
        elif by == "avg_order":
            n = x.groupby("_k")["order"].nunique()
            v = (x.groupby("_k")["amount"].sum() / n)[n >= 3]
        else:
            v = x.groupby("_k")["order"].nunique()
        name = x.drop_duplicates("_k").set_index("_k")["cust"] if "cust" in x else None
        d = pd.DataFrame({"v": v}).reset_index().sort_values(["v", "_k"], ascending=[False, True], kind="mergesort")
        d["name"] = d["_k"].map(name) if name is not None else d["_k"]
        return d

    @staticmethod
    def _filter(x, item, group):
        if item is not None:
            x = x[x["item"] == item]
        if group is not None:
            x = x[x["group"] == group]
        return x


def _name_key_map(values) -> dict:
    out = {}
    for v in pd.Series(list(values)).dropna().astype(str):
        out.setdefault(norm_key(v), v)
    return out


def _truthy(v) -> bool:
    return norm_key(v) in ("true", "1", "1 0", "co", "x", "yes", "ke don")


def _hours(x) -> pd.Series:
    """Giờ tạo đơn: từ cột giờ riêng, không có thì từ cột ngày nếu có ghi giờ. Dòng trả hàng ghi giờ trả nên lấy dòng đầu của đơn."""
    src = x["time"] if "time" in x else x["date"]
    d = pd.to_datetime(src, errors="coerce")
    h = d.dt.hour if d.notna().mean() > 0.5 else src.map(lambda v: getattr(parse_time(v), "hour", None)).astype("float64")
    if "time" not in x and (h.fillna(0) == 0).all():
        return pd.Series(float("nan"), index=x.index)
    return h.groupby(x["order"]).transform("first")


def _unit_factor(biz, roles, sales, x) -> pd.Series:
    """Hệ số đổi ra đơn vị cơ bản theo (hàng, đơn vị) từ bảng quy đổi; không có thì 1."""
    src = biz.tables[sales]
    q = roles[sales].get("qty")
    cols = [c for c in (q + "__dv" if q else None, roles[sales].get("unit")) if c and c in src]
    if not cols or "item" not in x:
        return pd.Series(1.0, index=x.index)
    table, via = {}, {}
    for t, rr in roles.items():
        if {"item", "big_unit", "factor"} <= set(rr):
            d = biz.tables[t]
            # hệ số ghi '15 vỉ': một hộp bằng 15 vỉ, phải nhân tiếp hệ số của vỉ
            so = d[rr["factor"] + "__so"] if rr["factor"] + "__so" in d else d[rr["factor"]]
            dv = d[rr["factor"] + "__dv"] if rr["factor"] + "__dv" in d else [None] * len(d)
            for a, u, f, w in zip(d[rr["item"]], d[rr["big_unit"]], pd.to_numeric(so, errors="coerce"), dv):
                table[(norm_key(a), norm_key(u))] = f
                if w is not None and w == w:
                    via[(norm_key(a), norm_key(u))] = norm_key(w)
    for k, w in via.items():
        table[k] = table[k] * table.get((k[0], w), 1.0)
    units = src[cols[0]].reindex(x.index)
    if len(cols) > 1:
        units = units.where(units.notna(), src[cols[1]].reindex(x.index))
    return pd.Series([table.get((norm_key(a), norm_key(u)), 1.0) for a, u in zip(x["item"], units)], index=x.index, dtype="float64")


def _mixed_units(df, r, x) -> bool:
    """Bảng bán có cột đơn vị và cùng một mặt hàng ghi nhiều đơn vị khác nhau."""
    col = r.get("qty")
    src = df[col + "__dv"] if col and col + "__dv" in df else (df[r["unit"]] if "unit" in r else None)
    if src is None:  # tên cột lạ: cột chữ ngắn, ít giá trị
        for c in df.columns:
            if "__" not in c and not c.startswith("_") and not str(df[c].dtype).startswith(("datetime", "Int", "int", "float")) and df[c].nunique() <= 30:
                v = df[c].dropna().astype(str)
                if len(v) and v.str.len().max() <= 8 and norm_status(v.iloc[0]) == "":
                    src = df[c]
                    break
    if src is None or len(src) != len(x):
        return False
    u = src.map(lambda v: norm_key(v) if v == v and v is not None else None)
    return bool(u.groupby(x["item"].values).nunique().max() > 1)


_ADJUST = re.compile(r"^(voucher|phi ship|phi giao|phi van chuyen|tien ship|ship|giam gia|chiet khau|diem|doi diem|tru diem|dung \d+ diem)\b")
_HEADER_AMOUNT = ("gia_tri_hang", "tien_hang", "tong_tien_hang", "tienhang")


def _build_header_only(biz, roles) -> Shop | None:
    """Sổ mỗi dòng một đơn, không có chi tiết mặt hàng ('Tiền hàng' là doanh thu của đơn): chỉ tính được số đo cấp đơn."""
    for t, r in roles.items():
        df = biz.tables[t]
        cols = {norm_col(c): c for c in df.columns}
        amt = next((cols[h] for h in _HEADER_AMOUNT if h in cols), None)
        if amt is None or not {"order", "date", "status"} <= set(r) or re.search(r"nhap", norm_col(t)):
            continue
        x = pd.DataFrame({"order": df[r["order"]], "amount": pd.to_numeric(df[amt], errors="coerce").fillna(0),
                          "date": pd.to_datetime(df[r["date"]], errors="coerce"), "status": df[r["status"]].map(norm_status).replace("", "hoan_thanh")})
        for k in ("cust", "cust_code"):
            if k in r:
                x[k] = df[r[k]]
        x["item"], x["group"], x["qty_base"] = None, None, 0.0
        x["hour"] = _hours(x)
        x = x[x["order"].notna() & x["date"].notna()].reset_index(drop=True)
        return Shop(business_id=biz.business_id, lines=x, reliable=True, biz=biz)
    return None




def build(biz) -> Shop | None:
    """None khi README không đủ để xác định bảng bán (shop đó đi đường model viết code)."""
    roles = _table_roles(biz)
    cand = [t for t, r in roles.items() if "amount" in r and ("item" in r or "item_code" in r)
            and not re.search(r"nhap|tra_hang|ton|kho", norm_col(t))]
    if not cand:
        return _build_header_only(biz, roles)
    sales = max(cand, key=lambda t: len(biz.tables[t]))
    r = roles[sales]
    df = biz.tables[sales]
    if "order" not in r and _id_column(df) is not None:
        r = {**r, "order": _id_column(df)}
    has_orders = "order" in r
    x = pd.DataFrame({"order": df[r["order"]] if has_orders else df.index, "amount": pd.to_numeric(df[r["amount"]], errors="coerce").fillna(0)})
    for role in ("item", "item_code", "date", "status", "qty_base", "qty", "cust", "cust_code", "group", "note", "time", "channel", "pay", "price", "cost", "staff"):
        if role in r:
            # cột '5 chai' đã tách thành <cột>__so (loader)
            x[role] = df[r[role] + "__so"] if role in ("qty", "qty_base") and r[role] + "__so" in df else df[r[role]]
    # sổ ghi tắt: tên đầy đủ, loại hàng ở cột chuẩn hoá của loader
    if r.get("item") and r["item"] + "__ten" in df:
        x["item"] = df[r["item"] + "__ten"]
        if r["item"] + "__loai" in df:
            x["group"] = df[r["item"] + "__loai"]

    # đơn: ngày, trạng thái, khách có thể ở bảng đầu đơn riêng
    for t, rr in roles.items():
        if t == sales or "order" not in rr or re.search(r"nhap|thu|quy|tra_hang|giao", norm_col(t)):
            continue
        need = [k for k in ("date", "status", "cust", "cust_code", "time", "channel", "pay", "staff") if k in rr and k not in x]
        if need:
            head = biz.tables[t][[rr["order"]] + [rr[k] for k in need]].drop_duplicates(rr["order"])
            head.columns = ["order"] + need
            x = x.merge(head, on="order", how="left")
    if "date" not in x:
        return None
    # đơn thuộc kỳ theo ngày đặt: dòng trả hàng ghi ngày trả vẫn thuộc ngày đặt của đơn
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["hour"] = _hours(x)
    x["date"] = x.groupby("order")["date"].transform("min")

    # hàng: tên, nhóm từ danh mục
    for t, rr in roles.items():
        if t == sales or "item_code" not in rr or "item_code" not in x:
            continue
        need = [k for k in ("item", "group") if k in rr and k not in x]
        if need and re.search(r"nhap|ton", norm_col(t)) is None:
            m = biz.tables[t][[rr["item_code"]] + [rr[k] for k in need]].drop_duplicates(rr["item_code"])
            m.columns = ["item_code"] + need
            x = x.merge(m, on="item_code", how="left")
    if "group" not in x and "item" in x:
        for t, rr in roles.items():
            if t != sales and {"item", "group"} <= set(rr):
                m = biz.tables[t][[rr["item"], rr["group"]]].drop_duplicates(rr["item"])
                g = {norm_key(a): b for a, b in zip(m[rr["item"]], m[rr["group"]])}
                x["group"] = x["item"].map(lambda v: g.get(norm_key(v)))
                break
    # bảng bán chỉ ghi tên: lấy mã hàng từ danh mục để nối danh mục thuốc
    if "item_code" not in x and "item" in x:
        for t, rr in roles.items():
            if t != sales and {"item", "item_code"} <= set(rr):
                m = {norm_key(a): b for a, b in zip(biz.tables[t][rr["item"]], biz.tables[t][rr["item_code"]])}
                x["item_code"] = x["item"].map(lambda v: m.get(norm_key(v)))
                break
    # nhóm khách, mã khách từ danh mục khách; thuốc kê đơn từ danh mục thuốc
    for t, rr in roles.items():
        if t == sales:
            continue
        if "cust_group" in rr and "cust_group" not in x:
            d = biz.tables[t]
            key = next((k for k in ("cust_code", "cust") if k in rr and k in x), None)
            if key:
                m = dict(zip(d[rr[key]], d[rr["cust_group"]]))
                x["cust_group"] = x[key].map(m)
        if "rx" in rr and "rx" not in x:
            d = biz.tables[t]
            key = next((k for k in ("item_code", "item") if k in rr and k in x), None)
            if key:
                m = dict(zip(d[rr[key]].map(str), d[rr["rx"]]))
                x["rx"] = x[key].map(lambda v: _truthy(m.get(str(v))))
    # tên khách từ danh mục khách: câu hỏi nêu tên, đơn chỉ ghi mã
    if "cust" not in x and "cust_code" in x:
        for t, rr in roles.items():
            d = biz.tables[t]
            cols = {norm_col(c): c for c in d.columns}
            name = rr.get("cust") or next((cols[h] for h in ("ten_goi", "tengoi", "ten_kh", "ten_khach_hang", "ten_khach", "khach_hang_ten", "kh") if h in cols), None)
            code = rr.get("cust_code")
            if t != sales and code and name and re.search(r"khach", norm_col(t).lower()):
                x["cust"] = x["cust_code"].map(dict(zip(d[code], d[name])))
                break
    if "cust_group" not in x and "cust_group" in r:
        x["cust_group"] = df[r["cust_group"]]
    if "item" not in x:
        return None

    has_status = "status" in x
    # trạng thái cấp đơn: có dòng trả thì cả đơn là trả; có huỷ thì huỷ
    if "status" in x:
        st = x["status"].map(norm_status)
    elif "note" in x:
        # chỉ tin ghi chú khi mọi ghi chú đều nói về trạng thái (rulebook: không nhắc trạng thái là đơn hoàn thành)
        st = x["note"].map(norm_status)
        if (st[x["note"].notna()] != "").all():
            has_status = True
        st = st.replace("", "hoan_thanh")
    else:
        st = pd.Series("hoan_thanh", index=x.index)
    x["status"] = st.replace("", "hoan_thanh")
    q = x["qty"] if "qty" in x else x.get("qty_base")
    if q is not None:
        neg = pd.to_numeric(q, errors="coerce") < 0
        x.loc[x["order"].isin(x.loc[neg, "order"]), "status"] = "tra_hang"
    for t, rr in roles.items():
        if "returned_qty" in rr and "order" in rr:
            rt = biz.tables[t]
            ids = rt.loc[pd.to_numeric(rt[rr["returned_qty"]], errors="coerce").fillna(0) > 0, rr["order"]]
            x.loc[x["order"].isin(ids), "status"] = "tra_hang"
    # đơn có cả dòng huỷ / trả và dòng thường: lấy mức nặng nhất
    bad = x[x["status"].isin(["huy", "tra_hang"])].drop_duplicates("order").set_index("order")["status"]
    x["status"] = x["status"].astype(object).where(~x["order"].isin(bad.index), x["order"].map(bad.to_dict()).astype(object))

    # dòng phụ của đơn không phải hàng bán: doanh thu không trừ voucher, không gồm phí giao (rulebook §1)
    adj = x["item"].map(lambda v: bool(_ADJUST.search(norm_key(v))))
    if adj.any():
        x["raw_amount"] = x["amount"]  # voucher, điểm, phí ship vẫn được hỏi riêng
        x.loc[adj, "amount"] = 0
    if "qty_base" not in x:
        factor = _unit_factor(biz, roles, sales, x)
        x["qty_base"] = pd.to_numeric(x.get("qty", 0), errors="coerce") * factor
        # một mặt hàng bán nhiều đơn vị mà không đọc được hệ số: số lượng cộng lẫn đơn vị, thà không trả lời
        if (factor == 1).all() and _mixed_units(df, r, x):
            x["qty_base"] = 0.0
    x.loc[adj, "qty_base"] = 0
    x["qty_base"] = pd.to_numeric(x["qty_base"], errors="coerce").fillna(0)
    if "group" not in x:
        x["group"] = None
    x = x[x["order"].notna() & x["date"].notna()].reset_index(drop=True)
    items = sorted(v for v in x["item"].dropna().astype(str).unique() if not _ADJUST.search(norm_key(v)))
    # 'Trà sữa + trân châu' ghi chung ô: tiền, số lượng của từng món không tách được theo tên
    if sum("+" in v for v in items) > 0.2 * len(items):
        items = []
    return Shop(business_id=biz.business_id, lines=x, items=items, groups=sorted(x["group"].dropna().astype(str).unique()),
                has_group=bool(x["group"].notna().any()), reliable=has_status, has_orders=has_orders, biz=biz)
