"""Bản ghi BTC → file shop (CSV/XLSX, bẫy định dạng kiểu BTC) + README đúng schema + SELECT ánh xạ view/nợ tất định theo họ.

Họ schema (cấu trúc bảng): joined, flat, vn_cols, neg_lines (train), events (dev), monthly (FINAL).
Kiểu trình bày (`style`, theo hạt shop, tỷ lệ gần shop_complexity.json của BTC): xlsx/csv, tách file theo tháng hoặc sheet theo
tháng, dòng tên báo cáo đầu sheet, tiêu đề hai tầng, dòng tổng cộng, số lượng kèm đơn vị, ô trống ở dòng sau của đơn,
dòng trùng cùng đơn, tên viết khác nhau (hoa/thường, khoảng trắng).
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import random
from pathlib import Path

from sqlv1 import world as WB
from sqlv1.sqlgen import lit, q

_d, _ts = WB._d, WB._ts

# tỷ lệ bẫy (shop_complexity.json, 30 shop BTC train): tên viết khác 19, gộp sheet/file 13, SL kèm đơn vị 11, tên báo cáo đầu sheet 10,
# tiêu đề hai tầng 8, dòng trùng cùng đơn 8, tách file theo kỳ 5, dòng tổng cộng 4, ô trống dòng sau 3
TRAP_RATE = {"name_variants": 19 / 30, "xlsx_sheets": 13 / 30, "qty_unit": 11 / 30, "title_rows": 10 / 30, "two_level": 8 / 30,
             "dup_rows": 8 / 30, "split_files": 5 / 30, "total_row": 4 / 30, "blank_cont": 3 / 30}

STATUS_SQL = "replace(strip_accents(lower(coalesce({c}, ''))), 'đ', 'd')"
CANCEL_SQL = "(regexp_matches({s}, '(^|[^a-z])huy([^a-z]|$)') OR {s} LIKE '%khong giao duoc%')"  # 'vận chuyển' chứa 'huy' giữa từ
RETURN_SQL = "({s} LIKE '%tra hang%' OR {s} LIKE '%tra lai%' OR {s} LIKE '%doi tra%' OR {s} LIKE '%hoan tra%' OR {s} LIKE '%khach tra%')"
WALKIN_SQL = "(nullif(trim({c}), '') IS NULL OR lower(trim({c})) IN ('khách lẻ', 'kl', 'khách vãng lai', 'lẻ', 'vãng lai'))"


def is_cancel(c):
    return CANCEL_SQL.format(s=STATUS_SQL.format(c=c))


def is_return(c):
    return RETURN_SQL.format(s=STATUS_SQL.format(c=c))


def walkin(c):
    return WALKIN_SQL.format(c=c)


def style_for(seed: int, family: str, split: str) -> dict:
    r = random.Random(f"style:{seed}:{family}")
    st = {k: r.random() < p for k, p in TRAP_RATE.items()}
    if family == "monthly":
        st["split_files"] = True  # họ monthly tách file theo tháng là bản chất của họ
    if st["split_files"] and st["xlsx_sheets"]:
        st["xlsx_sheets"] = r.random() < 0.5
        st["split_files"] = not st["xlsx_sheets"]
    st["xlsx"] = st["xlsx_sheets"] or r.random() < 0.35
    return st


def _vnd(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _dmy(ts, time=True) -> str:
    t = _ts(ts) if isinstance(ts, str) and len(ts) > 10 else dt.datetime.combine(_d(ts), dt.time())
    return t.strftime("%d/%m/%Y %H:%M:%S") if time else t.strftime("%d/%m/%Y")


def _variant(r: random.Random, name: str) -> str:
    """Cùng tên, khác cách viết (hoa/thường, khoảng trắng): ingest canon_spellings và luật 'tên gọi' coi là một."""
    k = r.random()
    if k < 0.4:
        return name.lower()
    if k < 0.7:
        return "  ".join(name.split(" ", 1)) if " " in name else name + " "
    return name.upper() if len(name) < 18 else name.lower()


# ------------------------------------------------------------------ bảng logic theo họ
class Table:
    def __init__(self, file, cols, rows, desc, lines=False, month_key=None, order_col=None, order_level=()):
        self.file, self.cols, self.rows, self.desc = file, cols, rows, desc
        self.lines, self.month_key, self.order_col, self.order_level = lines, month_key, order_col, tuple(order_level)


def tables(rec: dict, family: str, st: dict) -> tuple[list[Table], dict]:
    """(bảng logic, tên cột theo vai trò) — mọi họ cùng biểu diễn một bản ghi."""
    items = {it["item"]: it for it in rec["items"]}
    cust = {c["cust"]: c for c in rec["customers"]}
    feats, opts = rec["features"], rec["opts"]
    vr = random.Random(f"variants:{rec['seed']}")
    lines = collections.defaultdict(list)
    for ln in rec["lines"]:
        lines[ln["order"]].append(ln)
    rets = collections.defaultdict(list)
    for r in rec["returns"]:
        rets[r["order"]].append(r)
    orders = {o["order"]: o for o in rec["orders"]}
    walk = lambda o: o.get("walkin_text") or ""  # noqa: E731
    cname = lambda o: (cust[o["cust"]]["name"] if o["cust"] else walk(o))  # noqa: E731

    def nm(code, var=True):
        if code == "PV":
            return "Phí phục vụ"
        n = items[code]["name"]
        return _variant(vr, n) if var and st.get("name_variants") and vr.random() < 0.06 else n

    cvar = {}

    def cn(o):
        if o["order"] not in cvar:
            n = cname(o)
            cvar[o["order"]] = _variant(vr, n) if o["cust"] and st.get("name_variants") and vr.random() < 0.08 else n
        return cvar[o["order"]]

    def qty_cell(ln):
        return f"{ln['qty']} {ln['unit']}" if st.get("qty_unit") else ln["qty"]

    T, R = [], {"family": family}
    has_staff = bool(feats.get("staff"))
    if family == "joined":
        T.append(Table("items.csv", ["item", "name", "group", "base_unit", "price"] + (["cost"] if feats.get("cost") else []) + (["rx"] if feats.get("rx") else []),
                       [[it["item"], it["name"], it["group"], it["base"], it["price"]] + ([it["cost"]] if feats.get("cost") else [])
                        + (["Có" if it.get("rx") else "Không"] if feats.get("rx") else []) for it in rec["items"]]
                       + ([["PV", "Phí phục vụ", None, "lần", None] + ([None] if feats.get("cost") else []) + ([None] if feats.get("rx") else [])] if opts.get("service_fee") else []),
                       "danh mục hàng: item = mã hàng, name = tên hàng, group = nhóm hàng, base_unit = đơn vị cơ bản, price = giá bán một đơn vị cơ bản"
                       + (", cost = giá vốn một đơn vị cơ bản" if feats.get("cost") else "") + (", rx = thuốc kê đơn (Có/Không)" if feats.get("rx") else "")
                       + ("; mã PV là dòng phí phục vụ" if opts.get("service_fee") else "")))
        T.append(Table("units.csv", ["item", "unit", "factor"], [[u["item"], u["unit"], u["factor"]] for u in rec["units"]],
                       "quy đổi đơn vị: 1 unit của mặt hàng item = factor đơn vị cơ bản (đơn vị cơ bản có factor = 1)"))
        T.append(Table("customers.csv", ["customer_id", "name", "type"], [["KL", "Khách lẻ", "Vãng lai"]] + [[c["cust"], c["name"], c["kind"]] for c in rec["customers"]],
                       "khách hàng: customer_id = mã khách, name = tên, type = nhóm khách; mã KL là khách lẻ"))
        oc = ["order_id", "date", "status", "channel", "customer", "payment"] + (["ship_fee"] if opts.get("ship_fee") else []) + (["voucher"] if opts.get("voucher") else []) + (["staff"] if has_staff else [])
        T.append(Table("orders.csv", oc, [[o["order"], o["at"], o["status_text"], o["channel"], o["cust"] or "KL", o["payment"]] + ([o["ship_fee"]] if opts.get("ship_fee") else [])
                                         + ([o["voucher"]] if opts.get("voucher") else []) + ([o.get("staff") or ""] if has_staff else []) for o in rec["orders"]],
                       "đơn hàng: order_id = mã đơn, date = thời điểm đặt, status = trạng thái, channel = kênh bán, customer = mã khách, payment = hình thức thanh toán"
                       + (", ship_fee = phí giao hàng khách trả" if opts.get("ship_fee") else "") + (", voucher = giảm giá trên cả đơn" if opts.get("voucher") else "")
                       + (", staff = nhân viên bán" if has_staff else ""), month_key=1, order_col=0))
        lc = ["order_id", "line", "item", "unit", "qty", "price"] + (["discount"] if opts.get("line_discount") else []) + ["amount"] + (["cost"] if feats.get("cost") else [])
        T.append(Table("order_lines.csv", lc, [[ln["order"], ln["line"], ln["item"], ln["unit"], qty_cell(ln), ln["price"]] + ([ln["discount"]] if opts.get("line_discount") else [])
                                              + [ln["amount"]] + ([ln.get("cost")] if feats.get("cost") else []) for ln in rec["lines"]],
                       "dòng hàng của đơn: qty theo cột unit, price = đơn giá của unit đó" + (", discount = chiết khấu dòng (đồng)" if opts.get("line_discount") else "")
                       + ", amount = thành tiền dòng" + (" = qty × price − discount" if opts.get("line_discount") else " = qty × price")
                       + (", cost = giá vốn một đơn vị bán (cùng đơn vị với price)" if feats.get("cost") else ""), lines=True, order_col=0))
        T.append(Table("returns.csv", ["return_no", "order_id", "date", "item", "unit", "qty", "amount"],
                       [[r["ret"], r["order"], r["at"], r["item"], r["unit"], r["qty"], r["amount"]] for r in rec["returns"]],
                       "phiếu trả hàng: order_id = mã đơn gốc, date = thời điểm trả, qty theo unit, amount = tiền hoàn"))
        if feats.get("debt"):
            T.append(Table("payments.csv", ["pay_no", "order_id", "date", "amount"], [[p["payment"], p["order"], p["at"], p["amount"]] for p in rec["payments"]],
                           "khoản khách trả cho đơn ghi nợ (payment 'Ghi nợ'): order_id = mã đơn, date = thời điểm trả, amount = số tiền trả"))
        R.update(kind="header_lines", orders="orders.csv", lines="order_lines.csv", o_id="order_id", o_ts="date", o_status="status", o_channel="channel",
                 o_cust_code="customer", o_payment="payment", o_ship="ship_fee" if opts.get("ship_fee") else None, o_voucher="voucher" if opts.get("voucher") else None,
                 o_staff="staff" if has_staff else None, l_id="order_id", l_item="item", l_unit="unit", l_qty="qty", l_price="price", l_amount="amount",
                 l_cost="cost" if feats.get("cost") else None, catalog="items.csv", c_code="item", c_name="name", c_group="group", c_base="base_unit",
                 c_rx="rx" if feats.get("rx") else None, units="units.csv", u_item="item", u_unit="unit", u_factor="factor", customers="customers.csv",
                 cu_code="customer_id", cu_name="name", returns="returns.csv", r_order="order_id", payments="payments.csv" if feats.get("debt") else None,
                 p_order="order_id", p_date="date", p_amount="amount", cust_by_code=True)
    elif family in ("flat", "monthly"):
        mo = family == "monthly"
        cat = "danh_muc.csv" if mo else "catalog.csv"
        T.append(Table(cat, (["ma_hang", "ten_hang", "nhom", "dvt_goc", "gia"] if mo else ["code", "name", "category", "base_unit", "price"])
                       + ((["gia_von"] if mo else ["unit_cost"]) if feats.get("cost") else []) + ((["ke_don"] if mo else ["rx"]) if feats.get("rx") else []),
                       [[it["item"], it["name"], it["group"], it["base"], it["price"]] + ([it["cost"]] if feats.get("cost") else [])
                        + (["Có" if it.get("rx") else "Không"] if feats.get("rx") else []) for it in rec["items"]],
                       "danh mục hàng: mã hàng, tên hàng, nhóm hàng, đơn vị cơ bản, giá bán một đơn vị cơ bản"
                       + (", giá vốn một đơn vị cơ bản" if feats.get("cost") else "") + (", thuốc kê đơn (Có/Không)" if feats.get("rx") else "")))
        pk = "quy_doi.csv" if mo else "packs.csv"
        T.append(Table(pk, ["ma_hang", "dvt", "he_so"] if mo else ["code", "unit", "qty_per_unit"],
                       [[u["item"], u["unit"], u["factor"]] for u in rec["units"] if u["factor"] > 1],
                       "đơn vị lớn: 1 đơn vị lớn = hệ số đơn vị cơ bản; đơn vị không có dòng ở đây là đơn vị cơ bản (hệ số 1)"))
        use_note = bool(opts.get("note_status"))
        cols = (["ma_don", "ngay_gio", "ghi_chu" if use_note else "trang_thai", "kenh", "khach", "thanh_toan", "ma_hang", "ten_hang", "dvt", "so_luong", "don_gia"]
                if mo else ["order_id", "sold_at", "note" if use_note else "status", "channel", "customer_name", "payment", "line_no", "code", "product", "pack", "qty", "price"])
        cols += (["chiet_khau"] if mo else ["discount"]) if opts.get("line_discount") else []
        cols += ["thanh_tien"] if mo else ["line_total"]
        cols += ((["gia_von"] if mo else ["unit_cost"]) if feats.get("cost") else []) + ((["nhan_vien"] if mo else ["staff"]) if has_staff else [])
        rows = []
        for o in rec["orders"]:
            ret_items = {r["item"] for r in rets[o["order"]]}
            for ln in lines[o["order"]]:
                if use_note:
                    stc = o["note"] if o["status"] != "returned" or ln["item"] in ret_items or not ret_items else ""
                else:
                    stc = o["status_text"] if o["status"] != "returned" or ln["item"] in ret_items or not ret_items else rec["status_text"]["done"][0]
                row = [o["order"], _dmy(o["at"]) if mo else o["at"], stc, o["channel"], cn(o), o["payment"]]
                row += ([ln["item"], nm(ln["item"])] if mo else [ln["line"], ln["item"], nm(ln["item"])])
                row += [ln["unit"], qty_cell(ln), ln["price"]] + ([ln["discount"]] if opts.get("line_discount") else []) + [ln["amount"]]
                row += ([ln.get("cost")] if feats.get("cost") else []) + ([o.get("staff") or ""] if has_staff else [])
                rows.append(row)
            for code, name, amt in (("SHIP", "Phí giao hàng", o.get("ship_fee", 0)), ("VOUCHER", "Voucher giảm giá", -o.get("voucher", 0))):
                if amt:
                    row = [o["order"], _dmy(o["at"]) if mo else o["at"], o["note"] if use_note else o["status_text"], o["channel"], cn(o), o["payment"]]
                    row += ([code, name] if mo else [len(lines[o["order"]]) + 1, code, name]) + ["lần", 1, amt] + ([0] if opts.get("line_discount") else []) + [amt]
                    row += ([None] if feats.get("cost") else []) + ([o.get("staff") or ""] if has_staff else [])
                    rows.append(row)
        desc = ("mỗi dòng là một dòng hàng; thông tin đơn (mã đơn, thời điểm, " + ("ghi chú" if use_note else "trạng thái") + ", kênh, khách, thanh toán) lặp ở mọi dòng của đơn"
                + ("; ghi chú ghi tay tình trạng đơn (hủy, trả, đang giao...) hoặc để trống" if use_note else "; trạng thái ghi theo dòng: dòng khách trả ghi trạng thái trả hàng")
                + "; khách lẻ ghi 'Khách lẻ', 'kl'... hoặc để trống; số lượng theo đơn vị của dòng; thành tiền = SL × đơn giá" + (" − chiết khấu" if opts.get("line_discount") else "")
                + ("; dòng mã SHIP là phí giao hàng, dòng mã VOUCHER là giảm giá cả đơn (số âm)" if (opts.get("ship_fee") or opts.get("voucher")) else "")
                + ("; dòng 'Phí phục vụ' (mã PV) là phí phục vụ" if opts.get("service_fee") else ""))
        T.append(Table("ban_hang.csv" if mo else "sales.csv", cols, rows, desc, lines=True, month_key=1, order_col=0, order_level=(1, 3, 4, 5)))
        if mo:
            T.append(Table("tra_hang.csv", ["ma_phieu", "ma_don_goc", "ngay_tra", "tien_tra"], [[r["ret"], r["order"], _dmy(r["at"]), r["amount"]] for r in rec["returns"]],
                           "phiếu khách trả hàng: mã đơn gốc, ngày trả, tiền hoàn"))
        if feats.get("debt"):
            T.append(Table("thu_no.csv" if mo else "debt_payments.csv", ["ma_don", "ngay", "so_tien"] if mo else ["order_id", "paid_on", "amount"],
                           [[p["order"], _dmy(p["at"]) if mo else p["at"], p["amount"]] for p in rec["payments"]],
                           "khoản khách trả cho đơn ghi nợ (thanh toán 'Ghi nợ'): mã đơn, ngày trả, số tiền"))
        lt = "ban_hang.csv" if mo else "sales.csv"
        R.update(kind="flat", lines=lt, l_id=cols[0], l_ts=cols[1], l_status=cols[2], status_is_note=use_note, l_channel=cols[3], l_cust_name=cols[4], l_payment=cols[5],
                 l_item="ma_hang" if mo else "code", l_name="ten_hang" if mo else "product", l_unit="dvt" if mo else "pack", l_qty="so_luong" if mo else "qty",
                 l_price="don_gia" if mo else "price", l_amount="thanh_tien" if mo else "line_total", l_cost=("gia_von" if mo else "unit_cost") if feats.get("cost") else None,
                 l_staff=("nhan_vien" if mo else "staff") if has_staff else None, catalog=cat, c_code="ma_hang" if mo else "code", c_name="ten_hang" if mo else "name",
                 c_group="nhom" if mo else "category", c_base="dvt_goc" if mo else "base_unit", c_rx=("ke_don" if mo else "rx") if feats.get("rx") else None,
                 units=pk, u_item="ma_hang" if mo else "code", u_unit="dvt" if mo else "unit", u_factor="he_so" if mo else "qty_per_unit", units_base_rows=False,
                 returns="tra_hang.csv" if mo else None, r_order="ma_don_goc", payments=("thu_no.csv" if mo else "debt_payments.csv") if feats.get("debt") else None,
                 p_order="ma_don" if mo else "order_id", p_date="ngay" if mo else "paid_on", p_amount="so_tien" if mo else "amount", cust_by_code=False)
    elif family == "vn_cols":
        T.append(Table("DanhMucHang.csv", ["Mã hàng", "Tên hàng", "Nhóm hàng", "ĐVT gốc", "Giá bán"] + (["Giá vốn"] if feats.get("cost") else []) + (["Kê đơn"] if feats.get("rx") else []),
                       [[it["item"], it["name"], it["group"], it["base"], _vnd(it["price"])] + ([_vnd(it["cost"])] if feats.get("cost") else [])
                        + (["Có" if it.get("rx") else "Không"] if feats.get("rx") else []) for it in rec["items"]]
                       + ([["PV", "Phí phục vụ", None, "lần", None] + ([None] if feats.get("cost") else []) + ([None] if feats.get("rx") else [])] if opts.get("service_fee") else []),
                       "danh mục hàng: mã hàng, tên hàng, nhóm hàng, đơn vị tính gốc (cơ bản), giá bán một đơn vị gốc" + (", giá vốn một đơn vị gốc" if feats.get("cost") else "")
                       + (", kê đơn (Có/Không)" if feats.get("rx") else "") + ("; mã PV là phí phục vụ" if opts.get("service_fee") else "")))
        qrows = []
        for it in rec["items"]:
            ups = sorted((u for u in rec["units"] if u["item"] == it["item"] and u["factor"] > 1), key=lambda u: u["factor"])
            prev = (it["base"], 1)
            for u in ups:
                if u["factor"] % prev[1] == 0 and prev[1] > 1:
                    qrows.append([it["item"], u["unit"], u["factor"] // prev[1], prev[0]])
                else:
                    qrows.append([it["item"], u["unit"], u["factor"], it["base"]])
                prev = (u["unit"], u["factor"])
        T.append(Table("QuyDoiDVT.csv", ["Mã hàng", "ĐVT", "Quy đổi", "Theo ĐVT"], qrows,
                       "quy đổi theo bậc: 1 'ĐVT' = 'Quy đổi' × 'Theo ĐVT' (có thể phải nhân hai bậc mới về đơn vị gốc); đơn vị gốc không có dòng"))
        T.append(Table("KhachHang.csv", ["Mã KH", "Tên khách hàng", "Loại khách"], [[c["cust"], c["name"], c["kind"]] for c in rec["customers"]],
                       "khách hàng có mã; trong hóa đơn khách lẻ để trống Mã KH"))
        hc = ["Số HĐ", "Ngày bán", "Trạng thái", "Kênh bán", "Mã KH", "Khách hàng", "Thanh toán"] + (["Phí giao"] if opts.get("ship_fee") else []) + (["Giảm giá HĐ"] if opts.get("voucher") else []) + (["Nhân viên"] if has_staff else [])
        T.append(Table("HoaDon.csv", hc, [[o["order"], _dmy(o["at"]), (o["status_text"] if not (o["status"] == "returned" and _hidden_return(o)) else rec["status_text"]["done"][0]),
                                          o["channel"], o["cust"] or "", cn(o), o["payment"]] + ([_vnd(o["ship_fee"])] if opts.get("ship_fee") else [])
                                         + ([_vnd(o["voucher"])] if opts.get("voucher") else []) + ([o.get("staff") or ""] if has_staff else []) for o in rec["orders"]],
                       "hóa đơn: Số HĐ, Ngày bán (dd/mm/yyyy hh:mm:ss), Trạng thái, Kênh bán, Mã KH (trống = khách lẻ), Khách hàng, Thanh toán"
                       + (", Phí giao = phí giao hàng" if opts.get("ship_fee") else "") + (", Giảm giá HĐ = giảm giá trên cả hóa đơn" if opts.get("voucher") else "")
                       + (", Nhân viên bán" if has_staff else "") + "; đơn có phiếu trả hàng là đơn trả hàng dù cột Trạng thái chưa ghi", month_key=1, order_col=0))
        lc = ["Số HĐ", "STT", "Mã hàng", "ĐVT", "SL", "Đơn giá"] + (["Chiết khấu"] if opts.get("line_discount") else []) + ["Thành tiền"] + (["Giá vốn"] if feats.get("cost") else [])
        T.append(Table("ChiTietHoaDon.csv", lc, [[ln["order"], ln["line"], ln["item"], ln["unit"], qty_cell(ln), _vnd(ln["price"])] + ([_vnd(ln["discount"])] if opts.get("line_discount") else [])
                                                 + [_vnd(ln["amount"])] + ([_vnd(ln["cost"]) if ln.get("cost") is not None else None] if feats.get("cost") else []) for ln in rec["lines"]],
                       "dòng hàng: SL theo ĐVT, Đơn giá của ĐVT đó" + (", Chiết khấu dòng" if opts.get("line_discount") else "") + ", Thành tiền = SL × Đơn giá"
                       + (" − Chiết khấu" if opts.get("line_discount") else "") + (", Giá vốn một ĐVT bán" if feats.get("cost") else ""), lines=True, order_col=0))
        T.append(Table("PhieuTraHang.csv", ["Số phiếu", "Số HĐ gốc", "Ngày trả", "Mã hàng", "SL trả", "ĐVT", "Tiền hoàn"],
                       [[r["ret"], r["order"], _dmy(r["at"]), r["item"], r["qty"], r["unit"], _vnd(r["amount"])] for r in rec["returns"]],
                       "phiếu khách trả hàng (kể cả trả một phần): Số HĐ gốc, Ngày trả, tiền hoàn"))
        if feats.get("debt"):
            T.append(Table("ThuNo.csv", ["Số phiếu thu", "Số HĐ", "Ngày thu", "Số tiền"], [[p["payment"], p["order"], _dmy(p["at"]), _vnd(p["amount"])] for p in rec["payments"]],
                           "khách trả tiền cho hóa đơn 'Ghi nợ': Số HĐ, Ngày thu, Số tiền"))
        R.update(kind="header_lines", orders="HoaDon.csv", lines="ChiTietHoaDon.csv", o_id="Số HĐ", o_ts="Ngày bán", o_status="Trạng thái", o_channel="Kênh bán",
                 o_cust_code="Mã KH", o_cust_name="Khách hàng", o_payment="Thanh toán", o_ship="Phí giao" if opts.get("ship_fee") else None,
                 o_voucher="Giảm giá HĐ" if opts.get("voucher") else None, o_staff="Nhân viên" if has_staff else None, l_id="Số HĐ", l_item="Mã hàng", l_unit="ĐVT",
                 l_qty="SL", l_price="Đơn giá", l_amount="Thành tiền", l_cost="Giá vốn" if feats.get("cost") else None, catalog="DanhMucHang.csv", c_code="Mã hàng",
                 c_name="Tên hàng", c_group="Nhóm hàng", c_base="ĐVT gốc", c_rx="Kê đơn" if feats.get("rx") else None, units="QuyDoiDVT.csv", stepwise=True,
                 u_item="Mã hàng", u_unit="ĐVT", u_factor="Quy đổi", u_via="Theo ĐVT", returns="PhieuTraHang.csv", r_order="Số HĐ gốc",
                 payments="ThuNo.csv" if feats.get("debt") else None, p_order="Số HĐ", p_date="Ngày thu", p_amount="Số tiền", cust_by_code=True)
    elif family in ("neg_lines", "events"):
        ev = family == "events"
        T.append(Table("products.csv" if ev else "items.csv", ["sku", "name", "category" if ev else "group", "base_unit", "price"]
                       + (["cost"] if feats.get("cost") else []) + (["rx"] if feats.get("rx") else []),
                       [[it["item"], it["name"], it["group"], it["base"], it["price"]] + ([it["cost"]] if feats.get("cost") else [])
                        + (["Có" if it.get("rx") else "Không"] if feats.get("rx") else []) for it in rec["items"]]
                       + ([["PV", "Phí phục vụ", None, "lần", None] + ([None] if feats.get("cost") else []) + ([None] if feats.get("rx") else [])] if opts.get("service_fee") else []),
                       "danh mục: sku = mã hàng, tên hàng, nhóm hàng, base_unit = đơn vị cơ bản, price = giá một đơn vị cơ bản" + (", cost = giá vốn một đơn vị cơ bản" if feats.get("cost") else "")
                       + (", rx = thuốc kê đơn" if feats.get("rx") else "") + ("; sku PV là phí phục vụ" if opts.get("service_fee") else "")))
        T.append(Table("conversion.csv" if ev else "unit_conversion.csv", ["sku", "unit", "to_base" if ev else "base_qty"], [[u["item"], u["unit"], u["factor"]] for u in rec["units"]],
                       "1 unit = số đơn vị cơ bản ghi ở cột cuối (đơn vị cơ bản = 1)"))
        if ev:
            hc = ["order_id", "date", "status", "channel", "name", "pay"] + (["ship_fee"] if opts.get("ship_fee") else []) + (["voucher"] if opts.get("voucher") else []) + (["staff"] if has_staff else [])
            T.append(Table("headers.csv", hc, [[o["order"], o["at"], o["status_text"], o["channel"], cn(o), o["payment"]] + ([o["ship_fee"]] if opts.get("ship_fee") else [])
                                              + ([o["voucher"]] if opts.get("voucher") else []) + ([o.get("staff") or ""] if has_staff else []) for o in rec["orders"]],
                           "một dòng mỗi đơn: date = thời điểm đặt, status = trạng thái, name = tên khách hoặc khách lẻ, pay = thanh toán"
                           + (", ship_fee = phí giao" if opts.get("ship_fee") else "") + (", voucher = giảm giá cả đơn" if opts.get("voucher") else ""), month_key=1, order_col=0))
            evr = [[ln["order"], ln["line"], ln["item"], ln["unit"], qty_cell(ln), ln["amount"], "sale", rec_at(orders, ln)] + ([ln.get("cost")] if feats.get("cost") else []) for ln in rec["lines"]]
            evr += [[r["order"], 0, r["item"], r["unit"], -r["qty"], -r["amount"], "refund", r["at"]] + ([None] if feats.get("cost") else []) for r in rec["returns"]]
            evr.sort(key=lambda x: (x[7], x[0], x[1]))
            T.append(Table("events.csv", ["order_id", "line", "sku", "unit", "qty", "amount", "event", "event_at"] + (["cost"] if feats.get("cost") else []), evr,
                           "event = 'sale' là dòng hàng của đơn (amount = thành tiền, event_at = thời điểm đặt); event = 'refund' là khách trả hàng: qty, amount ÂM, event_at = thời điểm trả"
                           + (", cost = giá vốn một đơn vị bán" if feats.get("cost") else ""), lines=True, month_key=7, order_col=0))
            R.update(kind="header_lines", orders="headers.csv", lines="events.csv", o_id="order_id", o_ts="date", o_status="status", o_channel="channel", o_cust_name="name",
                     o_payment="pay", o_ship="ship_fee" if opts.get("ship_fee") else None, o_voucher="voucher" if opts.get("voucher") else None, o_staff="staff" if has_staff else None,
                     l_id="order_id", l_item="sku", l_unit="unit", l_qty="qty", l_price=None, l_amount="amount", l_cost="cost" if feats.get("cost") else None, l_event="event",
                     catalog="products.csv", c_code="sku", c_name="name", c_group="category", c_base="base_unit", c_rx="rx" if feats.get("rx") else None,
                     units="conversion.csv", u_item="sku", u_unit="unit", u_factor="to_base", returns_in_lines=True, cust_by_code=False)
        else:
            rows = []
            for o in rec["orders"]:
                for ln in lines[o["order"]]:
                    rows.append([o["order"], o["at"], "Bán", o["status_text"], o["channel"], cn(o), o["payment"], "", ln["item"], ln["unit"], qty_cell(ln), ln["amount"]]
                                + ([ln.get("cost")] if feats.get("cost") else []) + ([o.get("staff") or ""] if has_staff else []))
            for r in rec["returns"]:
                o = orders[r["order"]]
                rows.append([r["ret"], r["at"], "Trả", rec["status_text"]["returned"][0], o["channel"], cn(o), o["payment"], r["order"], r["item"], r["unit"], -r["qty"], -r["amount"]]
                            + ([None] if feats.get("cost") else []) + ([o.get("staff") or ""] if has_staff else []))
            rows.sort(key=lambda x: (x[1], x[0]))
            T.append(Table("transactions.csv", ["bill", "time", "type", "status", "channel", "customer", "payment", "ref_bill", "sku", "unit", "qty", "amount"]
                           + (["cost"] if feats.get("cost") else []) + (["staff"] if has_staff else []), rows,
                           "sổ giao dịch chung: type = 'Bán' (dòng của đơn bán, bill = mã đơn, status = trạng thái đơn) hoặc 'Trả' (khách trả hàng, bill = mã phiếu trả, ref_bill = mã đơn gốc, "
                           "qty và amount ÂM, time = thời điểm trả); amount = thành tiền dòng" + (", cost = giá vốn một đơn vị bán" if feats.get("cost") else ""),
                           lines=True, month_key=1, order_col=0, order_level=(1, 2, 4, 5, 6, 7)))
            R.update(kind="neg", lines="transactions.csv", l_id="bill", l_ts="time", l_type="type", l_status="status", l_channel="channel", l_cust_name="customer",
                     l_payment="payment", l_ref="ref_bill", l_item="sku", l_unit="unit", l_qty="qty", l_amount="amount", l_cost="cost" if feats.get("cost") else None,
                     l_staff="staff" if has_staff else None, catalog="items.csv", c_code="sku", c_name="name", c_group="group", c_base="base_unit",
                     c_rx="rx" if feats.get("rx") else None, units="unit_conversion.csv", u_item="sku", u_unit="unit", u_factor="base_qty", cust_by_code=False)
        if feats.get("debt"):
            T.append(Table("repayments.csv" if ev else "debt_payments.csv", ["order_id", "date", "amount"], [[p["order"], p["at"], p["amount"]] for p in rec["payments"]],
                           "khoản khách trả cho đơn ghi nợ: order_id = mã đơn, date = thời điểm trả, amount = số tiền"))
            R.update(payments="repayments.csv" if ev else "debt_payments.csv", p_order="order_id", p_date="date", p_amount="amount")
    else:
        raise ValueError(family)
    _common(rec, family, T, R)
    return T, R


def rec_at(orders, ln):
    return orders[ln["order"]]["at"]


def _hidden_return(o) -> bool:
    """vn_cols: khoảng một nửa số đơn trả hàng chỉ biết qua phiếu trả (cột Trạng thái vẫn ghi hoàn thành)."""
    return o["order"][-1:].isdigit() and int(o["order"][-1]) % 2 == 0


def _common(rec, family, T, R):
    feats = rec["features"]
    vn = family == "vn_cols"
    mo = family == "monthly"
    if feats.get("purchases"):
        name = "NhapHang.csv" if vn else "nhap_hang.csv" if mo else "purchases.csv"
        cols = ["Số phiếu", "Ngày nhập", "Mã hàng", "ĐVT", "SL", "Tiền nhập", "Tình trạng"] if vn else (
            ["so_phieu", "ngay_nhap", "ma_hang", "dvt", "so_luong", "tien_nhap", "tinh_trang"] if mo else ["receipt", "date", "item", "unit", "qty", "amount", "status"])
        T.append(Table(name, cols, [[x["receipt"], _dmy(x["at"]) if (vn or mo) else x["at"], x["item"], x["unit"], x["qty"], _vnd(x["amount"]) if vn else x["amount"],
                                     "Đã hủy" if x.get("cancelled") else "Đã nhập"] for x in rec["purchases"]],
                       "phiếu nhập hàng: ngày phiếu, mã hàng, đơn vị nhập, số lượng, tổng tiền nhập của dòng, tình trạng (phiếu 'Đã hủy' không tính)"))
        R.update(purchases=name, pu_date=cols[1], pu_item=cols[2], pu_unit=cols[3], pu_qty=cols[4], pu_amount=cols[5], pu_status=cols[6])
    if feats.get("lots"):
        name = "LoHang.csv" if vn else "lo_hang.csv" if mo else "lots.csv"
        cols = ["Số lô", "Mã hàng", "Hạn dùng", "Tồn còn"] if vn else (["so_lo", "ma_hang", "han_dung", "ton_con"] if mo else ["lot", "item", "expiry", "qty_left"])
        end = _d(rec["data_end"])
        T.append(Table(name, cols, [[x["lot"], x["item"], _dmy(x["expiry"], False) if (vn or mo) else x["expiry"], x["qty_left"]] for x in rec["lots"]],
                       f"lô hàng: số lô, mã hàng, hạn dùng, tồn còn (đơn vị cơ bản) tại ngày {end.day}/{end.month}/{end.year}"))
        R.update(lots=name, lo_item=cols[1], lo_exp=cols[2], lo_qty=cols[3])
    if feats.get("price_history"):
        name = "LichSuGiaBan.csv" if vn else "lich_su_gia.csv" if mo else "price_history.csv"
        cols = ["Mã hàng", "Áp dụng từ ngày", "Giá bán"] if vn else (["ma_hang", "ap_dung_tu", "gia_moi"] if mo else ["item", "effective_from", "new_price"])
        T.append(Table(name, cols, [[x["item"], _dmy(x["from"], False) if (vn or mo) else x["from"], _vnd(x["price"]) if vn else x["price"]] for x in rec["price_history"]],
                       "lần đổi giá bán (giá một đơn vị cơ bản mới, áp dụng từ đầu ngày ghi)"))
    if feats.get("stock"):
        name = "TonDauKy.csv" if vn else "ton_dau_ky.csv" if mo else "opening_stock.csv"
        cols = ["Mã hàng", "Tồn đầu kỳ (ĐVT gốc)"] if vn else (["ma_hang", "ton_dau_ky"] if mo else ["item", "opening_qty"])
        d0 = _d(rec["start"])
        T.append(Table(name, cols, [[x["item"], x["qty"]] for x in rec["opening_stock"]],
                       f"tồn kho đầu kỳ theo đơn vị cơ bản, đếm lúc mở sổ ngày {d0.day}/{d0.month}/{d0.year} (trước mọi giao dịch)"))
        R.update(stock=name, st_item=cols[0], st_qty=cols[1])


# ------------------------------------------------------------------ ghi file với bẫy định dạng
def write_shop(rec: dict, family: str, st: dict, folder: Path, title: str) -> dict:
    """Ghi data/<shop>/ (file + README.md). Trả {file logic: [file thật ...]} và roles."""
    folder.mkdir(parents=True, exist_ok=True)
    T, R = tables(rec, family, st)
    r = random.Random(f"write:{rec['seed']}:{family}")
    written, readme_files = {}, []
    for t in T:
        if not t.rows:  # bảng rỗng (vd. không mặt hàng nào có đơn vị lớn): không phát hành file
            continue
        rows = [list(x) for x in t.rows]
        if t.lines and st.get("dup_rows") and rows:
            for i in sorted(r.sample(range(len(rows)), max(1, len(rows) // 300)), reverse=True):
                rows.insert(i + 1, list(rows[i]))
        if t.lines and st.get("blank_cont") and t.order_level and rows:
            prev = None
            for row in rows:
                if row[t.order_col] == prev:
                    for j in t.order_level:
                        if j != t.order_col:
                            row[j] = None
                prev = row[t.order_col]
        stem, ext = t.file.rsplit(".", 1)
        split = (st.get("split_files") or st.get("xlsx_sheets")) and t.month_key is not None and (t.lines or family in ("joined", "vn_cols", "events"))
        if split:
            parts, cur = collections.OrderedDict(), None
            for row in rows:  # dòng để trống ô ngày (ô trống dòng sau của đơn) đi theo tháng của dòng trước
                v = row[t.month_key]
                if v:
                    d = _d(v) if isinstance(v, str) and v[:4].isdigit() else dt.datetime.strptime(str(v)[:10], "%d/%m/%Y").date()
                    cur = (d.year, d.month)
                parts.setdefault(cur, []).append(row)
        if split and st.get("xlsx_sheets"):
            fn = f"{stem}_{rec['start'][:4]}.xlsx"
            sheets = [(f"T{m}", rs) for (y, m), rs in parts.items()]
            _write_xlsx(folder / fn, sheets, t.cols, st, r, title)
            written[t.file] = [fn]
        elif split:
            fns = []
            for (y, m), rs in parts.items():
                fn = f"{stem}_{m:02d}_{y}.{'xlsx' if st.get('xlsx') else 'csv'}"
                (_write_xlsx(folder / fn, [("Sheet1", rs)], t.cols, st, r, title) if st.get("xlsx") else _write_csv(folder / fn, rs, t.cols, st, r, title, t.lines))
                fns.append(fn)
            written[t.file] = fns
        else:
            fn = f"{stem}.{'xlsx' if st.get('xlsx') and t.lines else 'csv'}"
            (_write_xlsx(folder / fn, [("Sheet1", rows)], t.cols, st, r, title) if fn.endswith("xlsx") else _write_csv(folder / fn, rows, t.cols, st, r, title, t.lines))
            written[t.file] = [fn]
        readme_files.append((t, written[t.file]))
    (folder / "README.md").write_text(readme(rec, family, st, readme_files, title), encoding="utf-8")
    return {"files": written, "roles": R, "tables": {t.file: t.cols for t in T if t.file in written}}


def _header_block(cols, st, r, title, lines):
    out = []
    if lines and st.get("title_rows"):
        out.append([title] + [None] * (len(cols) - 1))
        if r.random() < 0.6:
            out.append(["Xuất từ sổ bán hàng"] + [None] * (len(cols) - 1))
    if lines and st.get("two_level"):
        top, half = [], len(cols) // 2
        for i in range(len(cols)):
            top.append("Thông tin đơn" if i == 0 else "Hàng hóa" if i == half else None)
        out.append(top)
    out.append(list(cols))
    return out


def _total_row(cols, rows):
    tot = [None] * len(cols)
    tot[0] = "Tổng cộng"
    for j in range(len(cols) - 1, 0, -1):
        vals = [x[j] for x in rows if isinstance(x[j], (int, float)) and not isinstance(x[j], bool)]
        if vals and len(vals) == len(rows):
            tot[j] = sum(vals)
            break
    return tot


def _write_csv(path, rows, cols, st, r, title, lines):
    out = _header_block(cols, st, r, title, lines) + rows
    if lines and st.get("total_row") and rows:
        out.append(_total_row(cols, rows))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for row in out:
            w.writerow(["" if v is None else v for v in row])


def _write_xlsx(path, sheets, cols, st, r, title):
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets:
        ws = wb.create_sheet(name)
        for row in _header_block(cols, st, r, title, True):
            ws.append(row)
        for row in rows:
            ws.append(row)
        if st.get("total_row") and rows:
            ws.append(_total_row(cols, rows))
    wb.save(path)


def readme(rec, family, st, files, title) -> str:
    d0, d1 = _d(rec["start"]), _d(rec["data_end"])
    head = [f"# {title}", "", f"Sổ bán hàng của cửa hàng, dữ liệu từ {d0.day}/{d0.month}/{d0.year} đến hết {d1.day}/{d1.month}/{d1.year}.", ""]
    body = []
    for t, fns in files:
        where = ", ".join(fns) if len(fns) <= 3 else f"{fns[0]} … {fns[-1]} ({len(fns)} file theo tháng)"
        if len(fns) == 1 and fns[0].endswith(".xlsx") and (st.get("xlsx_sheets") and t.month_key is not None and (t.lines or family in ("joined", "vn_cols", "events"))):
            where += " (mỗi sheet một tháng)"
        body.append(f"- `{where}`: {t.desc}.")
        body.append("  Cột: " + ", ".join(f"`{c}`" for c in t.cols) + ".")
    notes = []
    if st.get("title_rows") or st.get("two_level"):
        notes.append("Sổ dòng hàng có dòng tên báo cáo / dòng nhóm cột phía trên dòng tiêu đề cột.")
    if st.get("total_row"):
        notes.append("Cuối sổ dòng hàng có dòng 'Tổng cộng'.")
    if st.get("qty_unit"):
        notes.append("Ô số lượng ghi kèm đơn vị (ví dụ '3 chai').")
    if st.get("blank_cont"):
        notes.append("Các dòng sau của cùng một đơn để trống thông tin đơn (lấy theo dòng đầu đơn).")
    return "\n".join(head + body + ([""] + notes if notes else [])) + "\n"


# ------------------------------------------------------------------ tên bảng sau ingest, SELECT ánh xạ
def _src_files(src: str) -> set:
    parts = src.split(": ", 1)[1].split(", ") if src.startswith("UNION of ") else [src]
    return {x.split("#")[0] for x in parts}


def resolve_tables(meta: dict, files: dict) -> dict:
    """{file logic: [bảng/view trong DuckDB]} phủ đúng các file của nó (so tên file chính xác). Ưu tiên view gộp *_all;
    sheet/file tháng có tập cột khác (vd. tháng không có đơn hủy nên thiếu cột cờ _huy) đứng riêng như runtime thấy."""
    out = {}
    tabs = [(t["name"], _src_files(t["source"])) for t in meta["tables"]]
    for logical, fns in files.items():
        want, got, cover = set(fns), [], set()
        for name, fs in sorted(tabs, key=lambda x: -len(x[1])):
            if fs <= want and not (fs & cover):
                got.append(name)
                cover |= fs
        if cover == want:
            out[logical] = got
    return out


def relations(N: dict) -> dict:
    """{file logic: biểu thức quan hệ SQL}: một bảng → tên trong ngoặc kép; nhiều bảng → UNION ALL BY NAME."""
    return {k: q(v[0]) if len(v) == 1 else "(" + " UNION ALL BY NAME ".join(f"SELECT * FROM {q(x)}" for x in v) + ")" for k, v in N.items()}


def _factor_ctes(R, N):
    if R.get("units") not in N:
        return ["fac AS (SELECT CAST(NULL AS VARCHAR) AS fi, CAST(NULL AS VARCHAR) AS fu, CAST(NULL AS DOUBLE) AS ff WHERE false)"]
    U = N[R["units"]]
    if R.get("stepwise"):
        return [f"f1 AS (SELECT trim({q(R['u_item'])}) AS fi, trim({q(R['u_unit'])}) AS fu, TRY_CAST({q(R['u_factor'])} AS DOUBLE) AS ff, trim({q(R['u_via'])}) AS fv FROM {U})",
                "fac AS (SELECT a.fi, a.fu, a.ff * coalesce(b.ff * coalesce(c.ff, 1), 1) AS ff FROM f1 a LEFT JOIN f1 b ON b.fi = a.fi AND b.fu = a.fv "
                "LEFT JOIN f1 c ON c.fi = b.fi AND c.fu = b.fv)"]
    return [f"fac AS (SELECT trim(CAST({q(R['u_item'])} AS VARCHAR)) AS fi, trim({q(R['u_unit'])}) AS fu, any_value(TRY_CAST({q(R['u_factor'])} AS DOUBLE)) AS ff FROM {U} GROUP BY 1, 2)"]


def _unit_expr(alias, R, cols):
    """Đơn vị của dòng: cột đơn vị, hoặc phần chữ tách từ ô số lượng ('3 chai')."""
    ghi = R["l_qty"] + " (đơn vị ghi kèm)"
    parts = [f"{alias}.{q(ghi)}"] if ghi in cols else []
    if R.get("l_unit") and R["l_unit"] in cols:
        parts.append(f"{alias}.{q(R['l_unit'])}")
    return "trim(coalesce(" + ", ".join(parts) + "))" if len(parts) > 1 else f"trim({parts[0]})"


def view_sql(R: dict, N: dict, cols: dict) -> str:
    """SELECT ánh xạ (17 cột của views.VIEW_SYSTEM) cho một shop đã nạp. N: file logic → bảng; cols: bảng → cột."""
    kind = R["kind"]
    C = N[R["catalog"]]
    ctes = _factor_ctes(R, N) + [f"cat AS (SELECT trim(CAST({q(R['c_code'])} AS VARCHAR)) AS code, any_value({q(R['c_name'])}) AS cname, any_value({q(R['c_group'])}) AS cgroup, "
                                 f"any_value({q(R['c_base'])}) AS cbase FROM {C} GROUP BY 1)"]
    if kind == "header_lines":
        O, L = N[R["orders"]], N[R["lines"]]
        ret = []
        if R.get("returns") in N:
            ret.append(f"trim(CAST(l.{q(R['l_id'])} AS VARCHAR)) IN (SELECT trim(CAST({q(R['r_order'])} AS VARCHAR)) FROM {N[R['returns']]})")
        if R.get("returns_in_lines"):
            ret.append(f"trim(CAST(l.{q(R['l_id'])} AS VARCHAR)) IN (SELECT trim(CAST({q(R['l_id'])} AS VARCHAR)) FROM {L} WHERE {q(R['l_event'])} = 'refund')")
        ret_expr = "(" + " OR ".join([is_return("o." + q(R["o_status"]))] + ret) + ")"
        if R.get("cust_by_code"):
            cust_id = f"CASE WHEN nullif(trim(CAST(o.{q(R['o_cust_code'])} AS VARCHAR)), '') IS NULL OR upper(trim(CAST(o.{q(R['o_cust_code'])} AS VARCHAR))) = 'KL' THEN NULL ELSE trim(CAST(o.{q(R['o_cust_code'])} AS VARCHAR)) END"
            if R.get("o_cust_name"):
                cust_name = f"CASE WHEN {cust_id} IS NULL THEN NULL ELSE o.{q(R['o_cust_name'])} END"
            else:
                ctes.append(f"cu AS (SELECT trim(CAST({q(R['cu_code'])} AS VARCHAR)) AS k, any_value({q(R['cu_name'])}) AS n FROM {N[R['customers']]} GROUP BY 1)")
                cust_name = f"(SELECT n FROM cu WHERE cu.k = {cust_id})"
        else:
            cust_id = "NULL"
            cust_name = f"CASE WHEN {walkin('o.' + q(R['o_cust_name']))} THEN NULL ELSE o.{q(R['o_cust_name'])} END"
        where = f"WHERE l.{q(R['l_event'])} = 'sale'" if R.get("l_event") else ""
        unit = _unit_expr("l", R, cols[R["lines"]])
        ord_k = f"trim(CAST(o.{q(R['o_id'])} AS VARCHAR))"
        return ("WITH " + ",\n".join(ctes) + f"""
SELECT trim(CAST(l.{q(R['l_id'])} AS VARCHAR)) AS order_id, CAST(o.{q(R['o_ts'])} AS DATE) AS order_date, CAST(o.{q(R['o_ts'])} AS TIMESTAMP) AS order_ts,
  trim(CAST(l.{q(R['l_item'])} AS VARCHAR)) AS product_code, cat.cgroup AS product_group, coalesce(cat.cname, CAST(l.{q(R['l_item'])} AS VARCHAR)) AS product_name,
  TRY_CAST(l.{q(R['l_qty'])} AS DOUBLE) * coalesce(fac.ff, 1) AS qty, cat.cbase AS unit, TRY_CAST(l.{q(R['l_amount'])} AS DOUBLE) AS amount,
  {cust_id} AS customer_id, {cust_name} AS customer_name, {('o.' + q(R['o_staff'])) if R.get('o_staff') else 'NULL'} AS staff,
  o.{q(R['o_channel'])} AS channel, o.{q(R['o_payment'])} AS payment,
  {is_cancel('o.' + q(R['o_status']))} AS is_cancelled, {ret_expr} AS is_returned,
  CASE WHEN coalesce(cat.cname, '') = 'Phí phục vụ' THEN 'service' ELSE 'item' END AS line_kind
FROM {L} l JOIN {O} o ON {ord_k} = trim(CAST(l.{q(R['l_id'])} AS VARCHAR))
LEFT JOIN cat ON cat.code = trim(CAST(l.{q(R['l_item'])} AS VARCHAR))
LEFT JOIN fac ON fac.fi = trim(CAST(l.{q(R['l_item'])} AS VARCHAR)) AND fac.fu = {unit}
{where}""")
    if kind == "flat":
        L = N[R["lines"]]
        st = "l." + q(R["l_status"])
        ret = [is_return(st)]
        if R.get("returns") in N:
            ret.append(f"trim(CAST(l.{q(R['l_id'])} AS VARCHAR)) IN (SELECT trim(CAST({q(R['r_order'])} AS VARCHAR)) FROM {N[R['returns']]})")
        unit = _unit_expr("l", R, cols[R["lines"]])
        code = f"trim(CAST(l.{q(R['l_item'])} AS VARCHAR))"
        return ("WITH " + ",\n".join(ctes) + f"""
SELECT trim(CAST(l.{q(R['l_id'])} AS VARCHAR)) AS order_id, CAST(l.{q(R['l_ts'])} AS DATE) AS order_date, CAST(l.{q(R['l_ts'])} AS TIMESTAMP) AS order_ts,
  {code} AS product_code, cat.cgroup AS product_group, coalesce(cat.cname, l.{q(R['l_name'])}) AS product_name,
  TRY_CAST(l.{q(R['l_qty'])} AS DOUBLE) * coalesce(fac.ff, 1) AS qty, cat.cbase AS unit, TRY_CAST(l.{q(R['l_amount'])} AS DOUBLE) AS amount,
  NULL AS customer_id, CASE WHEN {walkin('l.' + q(R['l_cust_name']))} THEN NULL ELSE l.{q(R['l_cust_name'])} END AS customer_name,
  {('l.' + q(R['l_staff'])) if R.get('l_staff') else 'NULL'} AS staff, l.{q(R['l_channel'])} AS channel, l.{q(R['l_payment'])} AS payment,
  bool_or({is_cancel(st)}) OVER (PARTITION BY l.{q(R['l_id'])}) AS is_cancelled, bool_or({' OR '.join(ret)}) OVER (PARTITION BY l.{q(R['l_id'])}) AS is_returned,
  CASE WHEN {code} IN ('SHIP', 'VOUCHER') THEN 'adjust' WHEN {code} = 'PV' THEN 'service' ELSE 'item' END AS line_kind
FROM {L} l
LEFT JOIN cat ON cat.code = {code}
LEFT JOIN fac ON fac.fi = {code} AND fac.fu = {unit}""")
    if kind == "neg":
        L = N[R["lines"]]
        st = "l." + q(R["l_status"])
        unit = _unit_expr("l", R, cols[R["lines"]])
        code = f"trim(CAST(l.{q(R['l_item'])} AS VARCHAR))"
        bill = f"trim(CAST(l.{q(R['l_id'])} AS VARCHAR))"
        return ("WITH " + ",\n".join(ctes) + f",\nret AS (SELECT DISTINCT trim(CAST({q(R['l_ref'])} AS VARCHAR)) AS o FROM {L} WHERE {q(R['l_type'])} = 'Trả')" + f"""
SELECT {bill} AS order_id, CAST(l.{q(R['l_ts'])} AS DATE) AS order_date, CAST(l.{q(R['l_ts'])} AS TIMESTAMP) AS order_ts,
  {code} AS product_code, cat.cgroup AS product_group, coalesce(cat.cname, {code}) AS product_name,
  TRY_CAST(l.{q(R['l_qty'])} AS DOUBLE) * coalesce(fac.ff, 1) AS qty, cat.cbase AS unit, TRY_CAST(l.{q(R['l_amount'])} AS DOUBLE) AS amount,
  NULL AS customer_id, CASE WHEN {walkin('l.' + q(R['l_cust_name']))} THEN NULL ELSE l.{q(R['l_cust_name'])} END AS customer_name,
  {('l.' + q(R['l_staff'])) if R.get('l_staff') else 'NULL'} AS staff, l.{q(R['l_channel'])} AS channel, l.{q(R['l_payment'])} AS payment,
  {is_cancel(st)} AS is_cancelled, ({is_return(st)} OR {bill} IN (SELECT o FROM ret)) AS is_returned,
  CASE WHEN {code} = 'PV' THEN 'service' ELSE 'item' END AS line_kind
FROM {L} l
LEFT JOIN cat ON cat.code = {code}
LEFT JOIN fac ON fac.fi = {code} AND fac.fu = {unit}
WHERE l.{q(R['l_type'])} = 'Bán'""")
    raise ValueError(kind)


def debt_sqls(R: dict, N: dict) -> tuple[str, str] | None:
    """(SELECT đơn ghi nợ, SELECT khoản trả) cho debts.build, hoặc None khi shop không ghi nợ."""
    if R.get("payments") not in N:
        return None
    P = N[R["payments"]]
    pay = f"SELECT trim(CAST({q(R['p_order'])} AS VARCHAR)) AS order_id, CAST({q(R['p_date'])} AS DATE) AS pay_date, TRY_CAST({q(R['p_amount'])} AS DOUBLE) AS amount FROM {P}"
    kind = R["kind"]
    if kind == "header_lines":
        O, L = N[R["orders"]], N[R["lines"]]
        if R.get("cust_by_code"):
            cid = f"CASE WHEN nullif(trim(CAST(o.{q(R['o_cust_code'])} AS VARCHAR)), '') IS NULL OR upper(trim(CAST(o.{q(R['o_cust_code'])} AS VARCHAR))) = 'KL' THEN NULL ELSE trim(CAST(o.{q(R['o_cust_code'])} AS VARCHAR)) END"
            cname = f"o.{q(R['o_cust_name'])}" if R.get("o_cust_name") else f"(SELECT any_value({q(R['cu_name'])}) FROM {N[R['customers']]} c WHERE trim(CAST(c.{q(R['cu_code'])} AS VARCHAR)) = {cid})"
        else:
            cid = "NULL"
            cname = f"CASE WHEN {walkin('o.' + q(R['o_cust_name']))} THEN NULL ELSE o.{q(R['o_cust_name'])} END"
        ev = f" WHERE {q(R['l_event'])} = 'sale'" if R.get("l_event") else ""
        ship = f" + coalesce(TRY_CAST(o.{q(R['o_ship'])} AS DOUBLE), 0)" if R.get("o_ship") else ""
        vou = f" - coalesce(TRY_CAST(o.{q(R['o_voucher'])} AS DOUBLE), 0)" if R.get("o_voucher") else ""
        orders = f"""WITH lt AS (SELECT trim(CAST({q(R['l_id'])} AS VARCHAR)) AS k, SUM(TRY_CAST({q(R['l_amount'])} AS DOUBLE)) AS amt FROM {L}{ev} GROUP BY 1)
SELECT trim(CAST(o.{q(R['o_id'])} AS VARCHAR)) AS order_id, CAST(o.{q(R['o_ts'])} AS DATE) AS order_date, {cid} AS customer_id, {cname} AS customer_name,
  coalesce(lt.amt, 0){vou}{ship} AS total_due, {is_cancel('o.' + q(R['o_status']))} AS is_cancelled, o.{q(R['o_channel'])} = 'Sàn TMĐT' OR o.{q(R['o_payment'])} = 'COD' AS no_debt,
  TRUE AS is_credit
FROM {O} o LEFT JOIN lt ON lt.k = trim(CAST(o.{q(R['o_id'])} AS VARCHAR))
WHERE o.{q(R['o_payment'])} = 'Ghi nợ'"""
        return orders, pay
    L = N[R["lines"]]
    st = q(R["l_status"])
    typ = f" AND {q(R['l_type'])} = 'Bán'" if kind == "neg" else ""
    orders = f"""SELECT trim(CAST({q(R['l_id'])} AS VARCHAR)) AS order_id, CAST(min({q(R['l_ts'])}) AS DATE) AS order_date, NULL AS customer_id,
  CASE WHEN {walkin('any_value(' + q(R['l_cust_name']) + ')')} THEN NULL ELSE any_value({q(R['l_cust_name'])}) END AS customer_name,
  SUM(TRY_CAST({q(R['l_amount'])} AS DOUBLE)) AS total_due, bool_or({is_cancel(st)}) AS is_cancelled,
  any_value({q(R['l_channel'])}) = 'Sàn TMĐT' AS no_debt, TRUE AS is_credit
FROM {L} WHERE {q(R['l_payment'])} = 'Ghi nợ'{typ}
GROUP BY 1"""
    return orders, pay


def raw_info(R: dict, N: dict) -> dict:
    """Tên bảng (biểu thức quan hệ đã gộp) và cột thô cho SQL mẫu ngoài view."""
    out = {"family": R["family"], "kind": R["kind"]}
    for k in ("purchases", "lots", "stock", "catalog", "units", "lines", "orders"):
        if R.get(k) and R[k] in N:
            out[k] = N[R[k]]
    keys = ("pu_date", "pu_item", "pu_unit", "pu_qty", "pu_amount", "pu_status", "lo_item", "lo_exp", "lo_qty", "st_item", "st_qty", "c_code", "c_name",
            "c_rx", "u_item", "u_unit", "u_factor", "u_via", "stepwise", "l_id", "l_item", "l_unit", "l_qty", "l_price", "l_cost", "l_amount", "l_event",
            "l_type", "l_ts", "o_id", "o_ts")
    out.update({k: R[k] for k in keys if R.get(k) is not None})
    return out
