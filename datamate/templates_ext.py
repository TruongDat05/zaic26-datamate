"""Các dạng câu mở rộng của lớp tính toán: so nhiều tháng, hiệu ứng tăng giá, giá bán theo ngày, size, nhân viên, lãi gộp, nhập hàng, giao hàng.

Cùng nguyên tắc với templates.py: chỉ trả lời khi nhận chắc dạng câu, đo trên đáp án train bằng eval/template_eval.py.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import threading
from pathlib import Path

import pandas as pd

from . import timeutil as tu
from .clean import norm_col, norm_key
from .normq import normalize
from .shop import round2, round_vnd
from .templates import _D, _FILL, _MONTH, _SLANG, _ans, _clip, _channel_values, _keep, _out_of_data, _parse_period, _year, find_name


_LOCK = threading.RLock()
RULEBOOK = ""  # rulebook của bộ đang chạy, Agent gán khi khởi tạo

# Dạng câu → câu chữ dòng luật nó mã hoá. Rulebook của bộ đang chạy đổi dòng đó thì dạng câu chuyển cho model đọc rulebook.
_R_REV = "Đơn tính vào doanh thu | đơn **không huỷ** và **không trả hàng**"
_R_ORD = "Số đơn | chỉ tính đơn không huỷ, không trả"
_R_CHANGE = "Phần trăm thay đổi, tăng trưởng | so với **kỳ trước**"
_R_RANK = "Bán chạy | theo **số lượng quy về đơn vị cơ bản**"
_R_CUST = "Xác định khách | Shop có mã khách"
_R_DEBT = "Nợ một đơn tại ngày X | phần chưa trả của **tổng đơn**"
_R_MOVE = "Nợ thêm, trả nợ, thu nợ trong kỳ | chỉ đơn **ghi nợ**"
_R_SHIFT = "Ca sáng, ca chiều | theo giờ tạo đơn"
_R_PAY = "Doanh thu theo hình thức thanh toán | theo hình thức **ghi trên đơn**"
_R_PROFIT = "Lãi gộp | doanh thu (§1) trừ giá vốn"
RULE_OF = {
    "revenue": _R_REV, "g_rev": _R_REV, "rev_months": _R_REV, "rev_list": _R_REV, "rev_trend": _R_REV, "best_day": _R_REV, "best_month": _R_REV, "days_over": _R_REV, "g_days_over": _R_REV,
    "orders_over": _R_REV, "g_orders_over": _R_REV, "share_group": "Nhóm hàng | theo cột nhóm hàng", "g_share": _R_REV,
    "n_orders": _R_ORD, "g_qty": _R_REV, "qty": _R_REV, "n_cancelled": "Đơn bị huỷ | đơn có trạng thái huỷ", "cancel_rate": "Tỉ lệ đơn huỷ trong kỳ | phần trăm đơn huỷ",
    "avg_order": "Giá trị trung bình một đơn | làm tròn về đồng", "g_avg": "Giá trị trung bình một đơn | làm tròn về đồng",
    "rev_change_pct": _R_CHANGE, "g_rev_change": _R_CHANGE, "rev_change_2m": _R_CHANGE, "rev_yoy": _R_CHANGE, "shift": _R_SHIFT, "shift_change": _R_SHIFT,
    "channel_rev": _R_PAY, "channel_list": _R_PAY, "channel_share": _R_PAY, "rx_rev": "Thuốc kê đơn | **Thuốc kê đơn** = sản phẩm ghi kê đơn", "rx_share": "Thuốc kê đơn | **Thuốc kê đơn** = sản phẩm ghi kê đơn",
    "active_rev": "Doanh thu theo hoạt chất | doanh thu (§1)", "top_item": _R_RANK, "top_items": _R_RANK, "size_qty": "Size M, size L | món có tên kết thúc", "unit_total": "Bán được bao nhiêu <đơn vị> | cộng số lượng",
    "items_ge_n": _R_ORD, "g_items_ge": _R_ORD, "cust_ge_n": _R_CUST, "cust_share_ge": _R_CUST, "top_customer": _R_CUST, "top_customers": _R_CUST, "top_customers2": _R_CUST,
    "staff_best": "Nhân viên bán giỏi nhất | theo **doanh thu**", "staff_rev": _R_REV, "profit": _R_PROFIT, "profit_best": "Hàng lời nhất | theo **tổng lãi gộp**", "net_profit": "Lãi ròng một tháng | lãi gộp của tháng",
    "purchase_money": "Tiền nhập hàng trong kỳ | tổng tiền các phiếu nhập", "top_supplier": "Nhà cung cấp nhập nhiều nhất = nhà cung cấp có tiền nhập cao nhất", "avg_purchase": "Giá nhập bình quân một mặt hàng trong tháng | bình quân **theo số lượng nhập**",
    "in_minus_out": "Nhập trừ bán, chênh nhập và bán | phần **bán**", "stock": "Đến hết ngày X, tính đến hôm nay | hỏi số còn lại", "stock_big": "Tồn theo đơn vị lớn | list 2 số", "stock_value": "Giá trị hàng tồn kho | tồn **hết hôm nay**",
    "price_on_day": "Giá bán X ngày D | đơn giá ghi trên dòng bán", "price_effect": "Hiệu ứng tăng giá | lần tăng giá", "stale_count": "Mặt hàng không bán được trong kỳ (ế) | mặt hàng cửa hàng", "deliveries": "Giao nhiều đợt | **Số đợt giao trong tháng X**",
    "lot_nearest": "Lô gần hết hạn nhất của một thuốc | trong các lô còn tồn", "debt_total": _R_DEBT, "debt_top": _R_DEBT, "debt_n_cust": _R_DEBT, "debt_bands": "Tuổi nợ | số ngày tính từ **ngày đặt**", "debt_collected": _R_MOVE, "debt_cust_period": _R_MOVE,
    "un_opinion": "Thông tin file không ghi thì không suy ra", "un_period": "kỳ nằm **hoàn toàn** ngoài khoảng ngày có dữ liệu",
}


_RULE_LINES = json.loads((Path(__file__).parent / "rule_lines.json").read_text(encoding="utf-8"))  # đoạn đầu dòng luật → cả dòng ở rulebook train


def rule_ok(template: str) -> bool:
    """Dạng câu đã đo trên train và cả dòng luật của nó còn nguyên văn trong rulebook đang chạy."""
    frag = RULE_OF.get(template)
    if frag is None:
        return False
    return not RULEBOOK or _RULE_LINES.get(frag, frag) in RULEBOOK


def _is_rate(question: str, q: str) -> bool:
    return "%" in question or bool(re.search(r"phan tram|\bti le\b|\bty le\b|ti trong|ty trong", q))


def _months(q: str, today) -> list:
    return [tu.month(_year(m[1]), int(m[0])) for m in _MONTH.findall(q)]


def _rev_months(q, question, today, shop):
    if not re.search(r"\b(doanh thu|ban duoc nhieu tien|thu ve)\b", q) or re.search(r"\b(nhom|khach|ca sang|ca chieu)\b", q) or find_name(q, shop.items):
        return None
    ms = _months(q, today)
    # 'tháng 3, tháng 4, tháng 5 năm 2025': năm ghi một lần ở cuối
    bare = re.findall(r"\b(?:thang|th|t) ?(\d{1,2})\b(?! 20\d{2})", q)
    y = re.search(r"\bnam (20\d{2})\b", q)
    if not ms and len(bare) >= 2 and y:
        ms = [tu.month(int(y[1]), int(m)) for m in bare]
    # 'tháng 5, 6, 7/2025', 'th7, th8, th9/2026', 'các tháng 10, 11, 12 năm 2025'
    lst = re.search(r"\b(?:cac )?(?:thang|th|t) ?(\d{1,2})((?: (?:va |thang |th|t)?\d{1,2})+) (?:nam )?(20\d{2}|2[4-7])\b", q)
    if lst and len(ms) <= 1:
        nums = [int(lst[1])] + [int(v) for v in re.findall(r"\d{1,2}", lst[2])]
        if all(1 <= v <= 12 for v in nums) and nums == sorted(nums):
            ms = [tu.month(_year(lst[3]), v) for v in nums]
    # từ tháng A đến tháng B
    m = re.search(r"\b(?:tu )?(?:thang |th ?|t ?)?(\d{1,2}) (20\d{2}|2[4-7])(?: tu)? (?:den|toi) (?:thang |th ?|t ?)?(\d{1,2}) (20\d{2}|2[4-7])\b", q)
    if m:
        ms = tu.months_between(_year(m[2]), int(m[1]), _year(m[4]), int(m[3]))
    n = re.search(r"\b(\d) thang\b|\bghi (\d) so\b", q)
    if n and len(ms) != int(n[1] or n[2]):  # câu nêu số tháng khác số tháng đọc được: đọc sai, không đoán
        return None
    if len(ms) < 2 or ms[-1].start > today or any(_out_of_data(shop, x) for x in ms):
        return None
    vals = [shop.revenue(_clip(x, today)) for x in ms]
    if re.search(r"\btang lien tuc\b|\bluon cao hon\b|\bcao hon\b.*\bcao hon\b|\btang deu\b", q):
        return _ans("boolean", all(b > a for a, b in zip(vals, vals[1:])), "rev_trend")
    if re.search(r"\bgiam lien tuc\b|\bluon thap hon\b", q):
        return _ans("boolean", all(b < a for a, b in zip(vals, vals[1:])), "rev_trend")
    if re.search(r"\bthang nao\b", q) and re.search(r"\bcao nhat\b|\bnhieu nhat\b", q):
        i = max(range(len(vals)), key=lambda k: (vals[k], -k))
        return _ans("date", ms[i].start.strftime("%Y-%m"), "best_month")
    if re.search(r"\bthang nao\b", q) and re.search(r"\bthap nhat\b|\bit nhat\b", q):
        i = min(range(len(vals)), key=lambda k: (vals[k], k))
        return _ans("date", ms[i].start.strftime("%Y-%m"), "worst_month")
    if len(ms) == 2 and re.search(r"\b(so|hon|kem|tang|giam|chenh)\b", q) and _is_rate(question, q):
        # kỳ hỏi nêu trước hoặc sau 'so với': kỳ sau chữ 'so' là mốc
        so = re.search(r"(?:^| )so ", q)
        pos = [x.start() for x in _MONTH.finditer(q)]
        if so is None or len(pos) != 2:
            return None
        base = 0 if pos[0] > so.start() else 1
        if not vals[base]:
            return None
        return _ans("percent", round2((vals[1 - base] - vals[base]) / vals[base] * 100), "rev_change_2m")
    if re.search(r"\bmoi thang\b|\btung thang\b|\btheo thu tu\b|\blan luot\b", q):
        return _ans("list", vals, "rev_list")
    return None


def _rev_yoy(q, question, today, shop):
    if not re.search(r"\b(cung ky|cung thang)\b.*\bnam (?:truoc|ngoai)\b", q) or not re.search(r"\bdoanh thu\b", q) or re.search(r"\b(nhom|khach|ca sang|ca chieu)\b", q):
        return None
    if not _is_rate(question, q):
        return None
    parsed = _parse_period(re.sub(r"\b(cung ky|cung thang)\b.*\bnam (?:truoc|ngoai)\b", " ", q), today)
    if parsed is None or parsed[0].start > today or find_name(q, shop.items):
        return None
    p = _clip(parsed[0], today)
    prev = tu.same_period_last_year(p)
    if _out_of_data(shop, p) or _out_of_data(shop, prev):
        return None
    old = shop.revenue(prev)
    return _ans("percent", round2((shop.revenue(p) - old) / old * 100), "rev_yoy") if old else None


def _price_effect(q, question, today, shop):
    """Đổi giá ngày D: số lượng N ngày kể từ D so với N ngày ngay trước (rulebook §5)."""
    if not re.search(r"\b(tang gia|len gia|doi gia)\b", q) or not shop.items:
        return None
    n = re.search(r"\b(\d+) ngay\b", q)
    d = re.search(_D, q)
    item = find_name(q, shop.items)
    if not (n and d and item):
        return None
    day = dt.date(int(d[3]), int(d[2]), int(d[1]))
    k = int(n[1])
    after = tu.date_range(day, day + dt.timedelta(days=k - 1))
    before = tu.date_range(day - dt.timedelta(days=k), day - dt.timedelta(days=1))
    new, old = shop.qty(after, item), shop.qty(before, item)
    return _ans("percent", round2((new - old) / old * 100), "price_effect") if old else None


def _price_on_day(q, question, today, shop):
    if "price" not in shop.lines or not re.search(r"\bgia\b", q) or re.search(r"\b(nhap|von|tang|len gia|doi gia)\b", q):
        return None
    d = re.search(_D, q)
    item = find_name(q, shop.items)
    if not (d and item):
        return None
    x = shop.lines[(shop.lines["date"].dt.date == dt.date(int(d[3]), int(d[2]), int(d[1]))) & (shop.lines["item"] == item)]
    if x.empty:
        return None
    price = pd.to_numeric(x["price"], errors="coerce")
    q_line = pd.to_numeric(x["qty"], errors="coerce") if "qty" in x else None
    f = (x["qty_base"] / q_line).where(q_line.notna() & (q_line != 0), 1.0) if q_line is not None else 1.0
    if (x["qty_base"] == 0).all():
        return None
    vals = (price / f).dropna().round(6).unique()
    return _ans("number", round_vnd(vals[0]), "price_on_day") if len(vals) == 1 else None


def _size(q, question, today, shop):
    m = re.search(r"\bsize (m|l)\b", q)
    if not m or not shop.items or not re.search(r"\bly\b", q) or find_name(q, shop.groups) or _is_rate(question, q) or re.search(r"\b(doanh thu|tien)\b", q):
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    p = _clip(parsed[0], today)
    x = shop.counted(p)
    hit = x["item"].map(lambda v: norm_key(v).endswith(f" {m[1]}"))
    v = int(x.loc[hit, "qty_base"].sum())
    return _ans("number", v, "size_qty") if v else None


def _staff(q, question, today, shop):
    if "staff" not in shop.lines or not re.search(r"\b(nhan vien|nguoi ban|nv\d+)\b", q):
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    p = _clip(parsed[0], today)
    x = shop.counted(p)
    code = re.search(r"\bnv ?(\d+)\b", q)
    if code and re.search(r"\b(ban duoc|doanh thu|tien)\b", q):
        hit = x["staff"].map(lambda v: norm_key(v).replace(" ", "") == "nv" + code[1])
        return _ans("number", round_vnd(x.loc[hit, "amount"].sum()), "staff_rev") if hit.any() else None
    if re.search(r"\b(ban gioi nhat|doanh thu cao nhat|ban duoc nhieu tien nhat)\b", q):
        g = x.groupby("staff")["amount"].sum().reset_index().sort_values(["amount", "staff"], ascending=[False, True], kind="mergesort")
        return _ans("string", str(g["staff"].iloc[0]), "staff_best") if len(g) else None
    return None


def _profit(q, question, today, shop):
    if "cost" not in shop.lines or "qty" not in shop.lines or not re.search(r"\b(lai gop|loi gop|loi nhat|lai nhat)\b", q) or re.search(r"\b(lai rong|nhom|khach)\b", q):
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    p = _clip(parsed[0], today)
    x = shop.counted(p)
    profit = x["amount"] - pd.to_numeric(x["qty"], errors="coerce").fillna(0) * pd.to_numeric(x["cost"], errors="coerce").fillna(0)
    if re.search(r"\b(hang|mon|mat hang|san pham) nao\b|\bnao\b.*\bnhat\b|\b(san pham|mat hang|mon)\b.*\bnhat\b", q):
        key = "item_code" if "item_code" in x else "item"
        by = [x[key], x["item"]] if key != "item" else [x["item"]]
        g = profit.groupby(by).sum().reset_index(name="v").sort_values(["v", key], ascending=[False, True], kind="mergesort")
        return _ans("string", str(g["item"].iloc[0]), "profit_best") if len(g) else None
    item = find_name(q, shop.items)
    if item:
        profit = profit[x["item"] == item]
    return _ans("number", round_vnd(profit.sum()), "profit") if len(x) and not _out_of_data(shop, p) else None


_PUR_DATE = ("ngay_nhap", "thoi_gian", "ngay_nhap_kho", "ngaynhapkho", "ngay")
_PUR_TOTAL = ("can_tra_ncc", "tonggiatrinhap", "tong_gia_tri_nhap", "thanh_tien")
_PUR_ST = ("trang_thai", "tinhtrang", "tinh_trang", "trang_thai_phieu")
_PUR_NOTE = ("ghi_chu", "ghi_chu", "note", "noi_dung", "luu_y", "mo_ta", "dien_giai", "ghi_chu_")


def _cols(df) -> dict:
    return {norm_col(c): c for c in df.columns if "__" not in c and not c.startswith("_")}


def _purchases(shop):
    """(dòng phiếu nhập: date, total, cancelled) hoặc None."""
    if shop.biz is None:
        return None
    for t, df in shop.biz.tables.items():
        nt = norm_col(t).lower()
        if "nhap" not in nt or "chi_tiet" in nt or "chitiet" in nt:
            continue
        c = _cols(df)
        date = next((c[h] for h in _PUR_DATE if h in c), None)
        total = next((c[h] for h in _PUR_TOTAL if h in c), None)
        if date is None or total is None:
            continue
        st = next((c[h] for h in _PUR_ST if h in c), None)
        note = next((c[h] for h in _PUR_NOTE if h in c), None)
        src = st or note
        cancelled = df[src].map(lambda v: "huy" in norm_key(v)) if src else pd.Series(False, index=df.index)
        return pd.DataFrame({"date": pd.to_datetime(df[date], errors="coerce"), "total": pd.to_numeric(df[total], errors="coerce").fillna(0), "cancelled": cancelled})
    return None


def _purchase_money(q, question, today, shop):
    if not re.search(r"\bnhap\b|\bcat hang\b|\bdanh hang ve\b", q) or not re.search(r"\b(tien|chi nhap|bo ra|gia tri)\b", q) or re.search(r"\b(gia nhap|binh quan|nha cung cap|ncc|tru|chenh)\b", q) or find_name(q, shop.items):
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    pur = _purchases(shop)
    if pur is None:
        return None
    p = _clip(parsed[0], today)
    x = pur[p.mask(pur["date"]) & ~pur["cancelled"]]
    return _ans("number", round_vnd(x["total"].sum()), "purchase_money") if len(x) else None


def _deliveries(q, question, today, shop):
    if not re.search(r"\b(dot|luot)\b.*\b(giao|cho)\b|\b(giao|cho)\b.*\b(dot|luot)\b", q) or shop.biz is None:
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today or re.search(r"\b(khach|xe|tai xe|nguoi)\b.*\bnao\b", q):
        return None
    p = _clip(parsed[0], today)
    for t, df in shop.biz.tables.items():
        if "giao" not in norm_col(t).lower():
            continue
        c = _cols(df)
        date = next((c[h] for h in ("ngay_giao_hang", "ngay_cho", "thoi_gian_giao", "ngay_giao") if h in c), None)
        order = next((c[h] for h in ("ma_don_hang", "ma_bill", "so_don", "ma_don") if h in c), None)
        lan = next((c[h] for h in ("lan_giao", "luot_giao", "dot_giao") if h in c), None)
        if not (date and order and lan):
            continue
        x = df[p.mask(pd.to_datetime(df[date], errors="coerce"))]
        return _ans("number", int(x[[order, lan]].drop_duplicates().shape[0]), "deliveries")
    return None


def _cust_share_ge(q, question, today, shop):
    m = re.search(r"\b(?:tu|it nhat) (\d+) don\b", q)
    if not m or not _is_rate(question, q) or not re.search(r"\bkhach\b", q) or re.search(r"\bnhom\b", q):
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    d = shop.customers(_clip(parsed[0], today), "orders")
    return _ans("percent", round2((d["v"] >= int(m[1])).sum() / len(d) * 100), "cust_share_ge") if len(d) else None


# ---------------------------------------------------- khung chung: số đo + kỳ + bộ lọc
_GENERIC_FILL = _FILL | set("""doanh thu tien ban thu ve mang dem lai duoc don hoa cac gia tri khach mua het tong cong tat ca hang chiem phan tram ti le ty dong gop phan
    trong so bao nhieu may tinh rieng moi ke nhe ne tiem cua shop quan nha thuoc hieu chot duoc dat sop xiu nho nhanh ban tra bang qua tai ra""".split()) - {"lan", "luot"}
_UNIT_WORDS = set("ly chai hop goi cai cay vien thung kg bao lon tui suat phan dia to mon cuon lo tuyp vi tam bich binh bo cuc mieng con".split())


def _sales_filter(q, shop, x):
    """(dòng đã lọc, câu còn lại) theo tên hàng, nhóm, nhóm khách, kênh, thanh toán, ca, kê đơn; không xử lý được thì None."""
    rest = q
    item = find_name(q, shop.items)
    group = find_name(q, shop.groups) if item is None else None
    if re.search(r"\bnhom\b", q) and group is None and not re.search(r"\bkhach (?:hang )?(?:nhom )?|\bnhom (si|le|than thiet|vip|moi|thuong)\b", q):
        return None
    for name in filter(None, [item, group]):
        rest = rest.replace(norm_key(name), " ")
    if item:
        x = x[x["item"] == item]
    if group:
        x = x[x["group"] == group]
    m_g = re.search(r"\b(?:khach (?:hang )?)?(?:thuoc )?nhom (si|le|than thiet|vip|moi|thuong)\b|\bkhach (si|le|than thiet|vip)\b", rest)
    if m_g:
        name = m_g[1] or m_g[2]
        if "cust_group" not in x:
            return None
        names = [g for g in x["cust_group"].dropna().astype(str).unique() if norm_key(g) == name]
        if not names:
            return None
        x = x[x["cust_group"] == names[0]]
        rest = rest.replace(m_g[0], " ")
    ca = re.findall(r"\bca (sang|chieu)\b", rest)
    if ca:
        if len(set(ca)) > 1 or "hour" not in x or x["hour"].isna().all():
            return None
        x = x[x["hour"].between(6, 13)] if ca[0] == "sang" else x[x["hour"].between(14, 21)]
        rest = re.sub(r"\bca (sang|chieu)\b", " ", rest)
    vals = _channel_values(rest)
    if len(vals) > 1:
        return None
    for col, pat, phrase in vals:
        if col not in x:
            return None
        hit = x[col].map(lambda v: bool(re.search(pat, norm_key(v))))
        if not hit.any():
            return None
        x = x[hit]
        rest = rest.replace(phrase, " ")
    if re.search(r"\bke don\b", rest):
        if "rx" not in x:
            return None
        neg = bool(re.search(r"\b(khong|hong|hok|k|tru)\b", rest))
        x = x[x["rx"] != neg]
        rest = re.sub(r"\b(khong|hong|hok|k|tru|thuoc|ke don)\b", " ", rest)
    return x, rest, item, group


def _generic_sales(q, question, today, shop):
    """Câu một số đo trên bảng bán: chỉ khi mọi từ còn lại đều là từ đệm đã biết."""
    if re.search(r"\b(no|nhap|ton|lai gop|lo|han|gia|chenh|hon|kem|tang|giam|so voi|so vs|nhat)\b|\bkhach (hang )?(nam|nu)\b|\b(may cu|bot|gop|bao hanh|ship|phi|voucher|khuyen mai|phu kien|khong tinh|ko tinh|bo ra)\b", q):
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    p = _clip(parsed[0], today)
    if _out_of_data(shop, p):
        return None
    x = shop.counted(p)
    f = _sales_filter(parsed[1], shop, x)
    if f is None:
        return None
    y, rest, item, group = f
    rate = _is_rate(question, q)
    is_avg = bool(re.search(r"\b(binh quan|trung binh)\b", rest))
    words = set(re.sub(r"\b(binh quan|trung binh|lan luot)\b", " ", rest).split()) - _GENERIC_FILL
    if re.search(r"\b(khach|nguoi|ai)\b", rest):  # còn nhắc đến khách cụ thể mà chưa lọc được
        return None
    if words - _UNIT_WORDS - {"so"}:
        return None
    is_rev = bool(re.search(r"\b(doanh thu|tien|thu ve)\b", rest))
    unit = words & _UNIT_WORDS
    if is_avg and not unit and re.search(r"\bdon\b", rest):
        n = y["order"].nunique()
        return _ans("number", round_vnd(y["amount"].sum() / n), "g_avg") if n else None
    if is_avg:  # 'bình quân' mà không rõ theo đơn: không đoán
        return None
    if rate:
        if y is x or not is_rev or not x["amount"].sum():
            return None
        return _ans("percent", round2(y["amount"].sum() / x["amount"].sum() * 100), "g_share")
    if is_rev:
        v = y["amount"].sum()
        return _ans("number", round_vnd(v), "g_rev") if v else None
    if re.search(r"\b(bao nhieu|may) (don|hoa don)\b|\b(don|hoa don) bao nhieu\b|\bso don\b|\btong so don\b", rest) and shop.has_orders:
        return _ans("number", int(y["order"].nunique()), "g_orders")
    if item is not None and re.search(r"\bban\b|\bbao nhieu\b", rest):
        v = float(y["qty_base"].sum())
        return _ans("number", int(v) if v.is_integer() else v, "g_qty") if v else None
    return None


def _money_threshold(q: str):
    m = re.search(r"\b(\d+)\s*(?:tr|trieu)\s*(\d*)\b", q)
    if m:
        return int(m[1]) * 1_000_000 + (int(m[2]) * 10 ** (6 - len(m[2])) if m[2] else 0)
    m = re.search(r"\b(\d+)\s*k\b", q)
    if m:
        return int(m[1]) * 1000
    m = re.search(r"\b(\d{1,3}(?: \d{3})+)\s*(?:d|dong)?\b", q)
    return int(m[1].replace(" ", "")) if m else None


def _order_threshold(q, question, today, shop):
    if not re.search(r"\btro len\b", q) or not shop.has_orders:
        return None
    v = _money_threshold(q)
    t = re.sub(r"\b\d+\s*(?:tr|trieu)\s*\d*\b|\b\d+\s*k\b|\b\d{1,3}(?: \d{3})+\s*(?:d|dong)?\b", " ", q)
    parsed = _parse_period(t, today)
    if v is None or parsed is None or parsed[0].start > today or find_name(q, shop.items) or re.search(r"\b(nhom|khach)\b", q):
        return None
    p = _clip(parsed[0], today)
    x = shop.counted(p)
    if re.search(r"\bmay ngay\b|\bbao nhieu ngay\b|\bso ngay\b", q):
        return _ans("number", int((x.groupby(x["date"].dt.date)["amount"].sum() >= v).sum()), "g_days_over")
    if re.search(r"\bdon\b", q):
        return _ans("number", int((x.groupby("order")["amount"].sum() >= v).sum()), "g_orders_over")
    return None


def _items_in_orders(q, question, today, shop):
    m = re.search(r"\b(?:o |tu |it nhat |trong it nhat |co mat o )*(\d+) don\b", q)
    if not m or not re.search(r"\b(mat hang|mon|san pham)\b", q) or not re.search(r"\b(tro len|it nhat)\b", q) or find_name(q, shop.items):
        return None
    parsed = _parse_period(q.replace(m[0], " "), today)
    if parsed is None or parsed[0].start > today:
        return None
    x = shop.counted(_clip(parsed[0], today))
    x = x[~x["item"].map(lambda v: norm_key(v) == "phi phuc vu")]
    return _ans("number", int((x.groupby("item")["order"].nunique() >= int(m[1])).sum()), "g_items_ge")


def _rev_change_fuzzy(q, question, today, shop):
    """'... tăng hay giảm mấy % so với cùng kỳ tháng trước', mọi cách xếp từ."""
    if not re.search(r"\bdoanh thu\b|\btien ban\b", q) or not _is_rate(question, q) or re.search(r"\b(nhom|khach|ca sang|ca chieu)\b", q) or find_name(q, shop.items):
        return None
    if not re.search(r"\b(cung ky thang truoc|thang truoc|tuan truoc|ky truoc|thang lien truoc)\b", q):
        return None
    t = re.sub(r"\b(?:so(?: voi| vs)? )?(?:cung ky )?(?:thang|tuan|ky) (?:lien )?truoc\b", " ", q)
    parsed = _parse_period(t, today)
    if parsed is None or parsed[0].start > today:
        return None
    p = _clip(parsed[0], today)
    prev = tu.previous(p)
    if _out_of_data(shop, p) or _out_of_data(shop, prev):
        return None
    old = shop.revenue(prev)
    return _ans("percent", round2((shop.revenue(p) - old) / old * 100), "g_rev_change") if old else None


def _active(q, question, today, shop):
    m = re.search(r"\bhoat chat (.+?)(?= thang| quy| nam| tuan| tu | hom| trong| ban| chiem| la | thu |$)", q)
    if not m or shop.biz is None:
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    for t, df in shop.biz.tables.items():
        c = _cols(df)
        col = next((c[h] for h in ("hoat_chat", "thanhphanchinh", "thanh_phan_chinh") if h in c), None)
        code = next((c[h] for h in ("ma_hang", "ma_sku", "mahanghoa", "ma_hh") if h in c), None)
        if not (col and code) or "item_code" not in shop.lines:
            continue
        codes = set(df.loc[df[col].map(lambda v: norm_key(v) == m[1].strip()), code].astype(str))
        if not codes:
            return None
        x = shop.counted(_clip(parsed[0], today))
        return _ans("number", round_vnd(x.loc[x["item_code"].astype(str).isin(codes), "amount"].sum()), "active_rev")
    return None


def _stale(q, question, today, shop):
    """Mặt hàng đã từng bán trước kỳ nhưng trong kỳ không bán được (rulebook §5)."""
    if not re.search(r"\b(mat hang|mon|san pham|hang)\b", q) or not re.search(r"\bkhong ban duoc\b|\bban khong duoc\b|\bk ban duoc\b|\bkhong ban ra duoc\b", q) or not re.search(r"\b(may|bao nhieu|so)\b", q):
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    p = _clip(parsed[0], today)
    x = shop.lines[shop.lines["status"].isin(["hoan_thanh", "dang_giao"])]
    x = x[~x["item"].map(lambda v: norm_key(v) == "phi phuc vu")]
    before = set(x.loc[x["date"].dt.date < p.start, "item"])
    now = set(shop.counted(p)["item"])
    return _ans("number", len(before - now), "stale_count")


_FAMILIES = [_order_threshold, _items_in_orders, _rev_change_fuzzy, _active, _stale, _generic_sales, _price_effect, _price_on_day, _rev_months, _rev_yoy, _size, _staff, _profit, _purchase_money, _deliveries, _cust_share_ge]


def answer_ext(question: str, as_of: str, shop) -> dict | None:
    q = normalize(question, _keep(shop))
    for pat, rep in _SLANG:
        q = re.sub(pat, rep, q)
    today = dt.date.fromisoformat(as_of)
    for f in _FAMILIES:
        r = f(q, question, today, shop)
        if r is not None:
            return r
    return None


# ---------------------------------------------------- công nợ (rulebook §4)
_TOTAL_HINTS = ("tong_sau_giam_gia", "tongthanhtoan", "tong_thanh_toan", "tong_don", "phai_thu", "tong_bill", "tong_tien", "tongtien", "tong_phieu", "thanh_tien")
_PAY_VAL = ("gia_tri", "so_tien_thu", "sotienthu", "so_tien", "tien_thu", "tien")
_PAY_TABLE = re.compile(r"thanh_toan|soquy|so_quy|phieuthu|phieu_thu")


def _once(shop, name, make):
    """Tính một lần cho mỗi shop. Nhiều luồng cùng hỏi: khoá theo shop, không để luồng khác đọc giá trị dở (luật 8)."""
    with _LOCK:
        if name not in shop.__dict__:
            shop.__dict__[name] = make(shop)
        return shop.__dict__[name]


def _debt_book(shop):
    return _once(shop, "_debt", _make_debt_book)


def _make_debt_book(shop):
    """(đơn: order, date, total, status, method, cust; khoản trả: order, date, amount); None nếu shop không có sổ thu theo đơn."""
    if shop.biz is None or not shop.has_orders:
        return None
    head = shop.lines.drop_duplicates("order").set_index("order")
    pay = None
    for t, df in shop.biz.tables.items():
        if not _PAY_TABLE.search(norm_col(t).lower()):
            continue
        c = _cols(df)
        val = next((c[h] for h in _PAY_VAL if h in c), None)
        dates = [k for k in df.columns if "__" not in k and pd.api.types.is_datetime64_any_dtype(df[k])]
        ids = [k for k in df.columns if "__" not in k and df[k].astype(str).isin(head.index.astype(str)).mean() > 0.5]
        if val and dates and ids:
            pay = pd.DataFrame({"order": df[ids[0]].astype(str), "date": df[dates[0]].dt.normalize(), "amount": pd.to_numeric(df[val], errors="coerce").fillna(0)})
            break
    if pay is None:
        return None
    total = method = None
    for t, df in shop.biz.tables.items():
        c = _cols(df)
        tot = next((c[h] for h in _TOTAL_HINTS if h in c), None)
        if tot is None or "chi_tiet" in norm_col(t).lower() or "chitiet" in norm_col(t).lower():
            continue
        ids = [k for k in df.columns if "__" not in k and df[k].astype(str).isin(head.index.astype(str)).mean() > 0.5]
        if not ids:
            continue
        d = df.drop_duplicates(ids[0]).set_index(df.drop_duplicates(ids[0])[ids[0]].astype(str))
        total = pd.to_numeric(d[tot], errors="coerce")
        m = next((c[h] for h in ("phuong_thuc_thanh_toan", "phuongthuctt", "phuong_thuc_tt", "tra_bang", "hinh_thuc_thanh_toan", "hinh_thuc_tt", "pttt") if h in c), None)
        method = d[m].map(norm_key) if m else None
        break
    if total is None or method is None:
        return None
    o = head.reset_index()[["order", "date", "status"] + [k for k in ("cust", "cust_code") if k in head.columns]].copy()
    o["key"] = o["order"].astype(str)
    o["total"] = o["key"].map(total).fillna(0)
    o["method"] = o["key"].map(method).fillna("")
    o["day"] = o["date"].dt.normalize()
    o = o[o["status"] != "huy"]
    return o, pay


def _is_credit(m: str) -> bool:
    return m in ("debt", "ghi no", "cong no", "no") or m.startswith("ghi no")


def _owed(shop, day):
    """Số còn nợ theo đơn lúc hết ngày; bỏ đơn COD, đơn qua sàn."""
    o, pay = _debt_book(shop)
    x = o[(o["day"] <= pd.Timestamp(day)) & ~o["method"].str.contains("cod|thu ho|san")].copy()
    paid = pay[pay["date"] <= pd.Timestamp(day)].groupby("order")["amount"].sum()
    x["owed"] = x["total"] - x["key"].map(paid).fillna(0)
    return x[x["owed"] > 0.5]


def _cust_key(x):
    if "cust_code" in x and x["cust_code"].notna().any():
        return x["cust_code"].where(x["cust_code"].notna())
    if "cust" in x:
        from .clean import is_walkin, norm_text
        return x["cust"].map(lambda v: None if v is None or v != v or is_walkin(v) else norm_text(v))
    return None


def _debt(q, question, today, shop):
    if not re.search(r"\b(no|cong no|mua chiu|chua tra het|chua thu|thieu tien)\b", q) or _debt_book(shop) is None:
        return None
    o, pay = _debt_book(shop)
    # nợ thêm, trả nợ trong kỳ
    if re.search(r"\b(no them|mua chiu them|ghi no them)\b", q) and re.search(r"\b(tra|tra bot)\b", q):
        parsed = _parse_period(q, today)
        names = shop.lines["cust"].dropna().astype(str).unique() if "cust" in shop.lines else []
        name = find_name(q, list(names))
        if parsed is None or not name:
            return None
        p = parsed[0]
        if "cust" not in o:
            return None
        c = o[o["cust"].map(lambda v: norm_key(v) == norm_key(name))]
        c = c[c["method"].map(_is_credit)]
        pp = pay.merge(c[["key", "day"]], left_on="order", right_on="key")
        same = pp[pp["date"] == pp["day"]].groupby("order")["amount"].sum()
        inp = c[p.mask(c["day"])]
        added = (inp["total"] - inp["key"].map(same).fillna(0)).clip(lower=0).sum()
        later = pp[(pp["date"] > pp["day"]) & p.mask(pp["date"])]["amount"].sum()
        if not added and not later:  # cả hai bằng 0 gần như luôn do nhận sai khách
            return None
        return _ans("list", [round_vnd(added), round_vnd(later)], "debt_cust_period")
    # thu nợ trong kỳ
    if re.search(r"\b(tra no|thu hoi|tra bot no|doi duoc|thu no)\b", q) and not re.search(r"\b(den het|tinh den|toi het)\b", q):
        parsed = _parse_period(q, today)
        if parsed is None or parsed[0].start > today or find_name(q, list(shop.lines["cust"].dropna().astype(str).unique()) if "cust" in shop.lines else []):
            return None
        p = _clip(parsed[0], today)
        c = o[o["method"].map(_is_credit)]
        pp = pay.merge(c[["key", "day"]], left_on="order", right_on="key")
        return _ans("number", round_vnd(pp[(pp["date"] > pp["day"]) & p.mask(pp["date"])]["amount"].sum()), "debt_collected")
    # còn nợ tại ngày
    d = re.search(_D, q)
    bands = bool(re.search(r"\btuoi no\b", q) and re.search(r"\b0\D+7\b", q))
    day = dt.date(int(d[3]), int(d[2]), int(d[1])) if d else today if re.search(r"\b(hom nay|hien tai|hien nay)\b", q) else None
    if day is None:
        return None
    x = _owed(shop, day)
    age = re.search(r"\b(?:qua|hon|tren) (\d+) ngay\b", q)
    if age and not bands:
        x = x[(pd.Timestamp(day) - x["day"]).dt.days > int(age[1])]
    key = _cust_key(x)
    if bands:  # 0–7, 8–15, trên 15 ngày, theo thứ tự câu hỏi
        age_d = (pd.Timestamp(day) - x["day"]).dt.days
        return _ans("list", [round_vnd(x.loc[age_d.between(0, 7), "owed"].sum()), round_vnd(x.loc[age_d.between(8, 15), "owed"].sum()), round_vnd(x.loc[age_d > 15, "owed"].sum())], "debt_bands")
    k = re.search(r"\b(?:top )?(\d{1,2}) khach\b", re.sub(_D, " ", q))
    if k and key is not None:
        x = x.assign(_k=key)
        x = x[x["_k"].notna()]
        g = x.groupby("_k")["owed"].sum().reset_index().sort_values(["owed", "_k"], ascending=[False, True], kind="mergesort").head(int(k[1]))
        names = x.drop_duplicates("_k").set_index("_k")["cust"] if "cust" in x else None
        rows = [[str(names[a]) if names is not None else str(a), round_vnd(b)] for a, b in zip(g["_k"], g["owed"])]
        return _ans("table", {"columns": ["khach_hang", "so_tien_no"], "rows": rows}, "debt_top")
    if re.search(r"\b(bao nhieu|may) khach\b|\bso khach\b|\bkhach\b.*\b(bao nhieu|nhieu) nguoi\b|\bdem\b.*\bkhach\b", q) and key is not None:
        return _ans("number", int(key.dropna().nunique()), "debt_n_cust")
    return _ans("number", round_vnd(x["owed"].sum()), "debt_total")


# ---------------------------------------------------- tồn kho, nhập hàng theo mặt hàng
def _stock(q, question, today, shop):
    if not re.search(r"\b(ton|trong kho|con lai|con bao nhieu|con may)\b", q) or re.search(r"\b(no|gia von|lo|han|het hang)\b", q) or shop.biz is None:
        return None
    last = shop.lines["date"].max().date()
    d = re.search(_D, q)
    if (d and dt.date(int(d[3]), int(d[2]), int(d[1])) != last) or today < last or not (d or re.search(r"\bhom nay\b", q)):
        return None
    for t, df in shop.biz.tables.items():
        if not re.search(r"ton", norm_col(t).lower()) or "the_kho" in norm_col(t).lower():
            continue
        c = _cols(df)
        val = next((c[h] for h in ("ton_cuoi_ky", "ton_cuoi", "ton_thuc", "ton_kho") if h in c), None)
        name = next((c[h] for h in ("ten_hang", "hang_hoa", "product", "ten_san_pham") if h in c), None)
        if not (val and name):
            continue
        hit = find_name(q, df[name].dropna().astype(str).tolist())
        if not hit:
            return None
        v = float(pd.to_numeric(df.loc[df[name] == hit, val], errors="coerce").sum())
        if re.search(r"\b(le|chan|nguyen)\b", q):
            return None
        return _ans("number", int(v) if v.is_integer() else v, "stock")
    return None


def _purchase_lines(shop):
    return _once(shop, "_pur", _make_purchase_lines)


def _make_purchase_lines(shop):
    """Dòng nhập: item, qty, money, date, cancelled, supplier; None nếu không ghép được."""
    if shop.biz is None:
        return None
    flat = detail = head = None
    for t, df in shop.biz.tables.items():
        nt = norm_col(t).lower()
        if "nhap" not in nt:
            continue
        c = _cols(df)
        has_item = any(h in c for h in ("ten_mat_hang", "product", "hang_hoa", "ten_hang", "ma_hang", "mahanghoa", "hang", "ma_hang_ban"))
        has_qty = any(h in c for h in ("sl_nhap", "so_luong", "sl"))
        has_date = any(h in c for h in ("ngay_nhap", "ngay", "ngaynhapkho"))
        if has_item and has_qty and has_date:
            flat = df
        elif has_item and has_qty:
            detail = df
        elif has_date or "thoi_gian" in c:
            head = df
    src = flat
    if src is None and detail is not None and head is not None:
        cd, ch = _cols(detail), _cols(head)
        key = next((h for h in ("ma_nhap_hang", "maphieunhap", "ma_phieu_nhap") if h in cd and h in ch), None)
        if key is None:
            return None
        src = detail.merge(head.drop_duplicates(ch[key])[[ch[key]] + [v for k, v in ch.items() if k in ("thoi_gian", "ngaynhapkho", "ngay_nhap", "trang_thai", "tinhtrang", "ma_ncc", "manhacungcap")]], left_on=cd[key], right_on=ch[key], suffixes=("", "_h"))
    if src is None:
        return None
    c = _cols(src)
    date = next((c[h] for h in ("ngay_nhap", "ngaynhapkho", "thoi_gian", "ngay") if h in c), None)
    qty = next((c[h] for h in ("sl_nhap", "so_luong", "sl") if h in c), None)
    price = next((c[h] for h in ("gia_nhap", "don_gia", "dongianhap", "don_gia_nhap") if h in c), None)
    item = next((c[h] for h in ("ten_mat_hang", "product", "hang_hoa", "ten_hang", "hang") if h in c), None)
    code = next((c[h] for h in ("ma_hang", "mahanghoa", "ma_hang_ban") if h in c), None)
    if not (date and qty and price and (item or code)):
        return None
    st = next((c[h] for h in _PUR_ST + _PUR_NOTE if h in c), None)
    x = pd.DataFrame({"date": pd.to_datetime(src[date], errors="coerce"), "qty": pd.to_numeric(src[qty], errors="coerce"), "price": pd.to_numeric(src[price], errors="coerce")})
    x["cancelled"] = src[st].map(lambda v: "huy" in norm_key(v)) if st else False
    u = next((c[h] for h in ("dvt_nhap", "dvt", "don_vi_tinh", "donvitinh") if h in c), None)
    if u:
        x["unit"] = src[u]
    if item:
        x["item"] = src[item]
    else:
        names = shop.lines.dropna(subset=["item_code"]).drop_duplicates("item_code").set_index("item_code")["item"] if "item_code" in shop.lines else {}
        x["item"] = src[code].map(names)
    sup = next((c[h] for h in ("nha_cung_cap", "ma_ncc", "manhacungcap") if h in c), None)
    if sup:
        x["supplier"] = src[sup]
    return x


def _avg_purchase(q, question, today, shop):
    if not re.search(r"\bnhap\b", q) or not re.search(r"\bbinh quan\b", q) or not re.search(r"\bgia\b|\btien\b", q):
        return None
    x = _purchase_lines(shop)
    parsed = _parse_period(q, today)
    if x is None or parsed is None:
        return None
    item = find_name(q, x["item"].dropna().astype(str).unique().tolist())
    if not item:
        return None
    y = x[(x["item"] == item) & parsed[0].mask(x["date"]) & ~x["cancelled"]]
    if y["qty"].sum() <= 0:
        return None
    return _ans("number", round_vnd((y["qty"] * y["price"]).sum() / y["qty"].sum()), "avg_purchase")


def _top_supplier(q, question, today, shop):
    if not re.search(r"\b(nha cung cap|ncc)\b", q) or not re.search(r"\b(nhieu|cao nhat|lon nhat)\b", q):
        return None
    x = _purchase_lines(shop)
    parsed = _parse_period(q, today)
    if x is None or parsed is None or "supplier" not in x:
        return None
    y = x[parsed[0].mask(x["date"]) & ~x["cancelled"]]
    g = (y["qty"] * y["price"]).groupby(y["supplier"]).sum().reset_index(name="v").sort_values(["v", "supplier"], ascending=[False, True], kind="mergesort")
    return _ans("string", str(g["supplier"].iloc[0]), "top_supplier") if len(g) else None


_FAMILIES[:0] = [_debt, _stock, _avg_purchase, _top_supplier]


# ---------------------------------------------------- bán được bao nhiêu <đơn vị>, cộng mọi mặt hàng (rulebook §3)
_BASE_UNIT = ("don_vi_co_ban", "don_vi_goc", "dv_goc", "donvicoso", "dv_nho", "dvt", "don_vi", "don_vi_tinh", "donvi")


def _base_units(shop) -> dict:
    return _once(shop, "_units", _make_base_units)


def _make_base_units(shop) -> dict:
    out = {}
    for t, df in (shop.biz.tables.items() if shop.biz is not None else []):
        c = _cols(df)
        name = next((c[h] for h in ("ten_hang", "ten_hang_hoa", "ten_san_pham", "ten_mon", "hang", "hang_ban", "mon", "san_pham", "tengoi", "ten_sp") if h in c), None)
        unit = next((c[h] for h in _BASE_UNIT if h in c), None)
        if name and unit and re.search(r"danh|san_pham|hang_hoa|hanghoa|quy_doi|quydoi", norm_col(t).lower()):
            for a, b in zip(df[name], df[unit]):
                out.setdefault(norm_key(a), norm_key(b))
    return out


def _unit_total(q, question, today, shop):
    m = re.search(r"\b(?:bao nhieu|may) (ly|chai|hop|goi|cai|cay|vien|thung|kg|bao|lon|tui|suat|phan|dia|to|cuon|tuyp|vi|tam|bich|binh|m3|bo|noi)\b", q)
    if not m or not shop.items or find_name(q, shop.items) or find_name(q, shop.groups) or re.search(r"\b(nhom|size|gia|khach|nhap|ton|kho|ca sang|ca chieu|doanh thu|tien)\b", q):
        return None
    parsed = _parse_period(q, today)
    units = _base_units(shop)
    if parsed is None or parsed[0].start > today or not units:
        return None
    # từ lạ có thể là tên hàng không khớp ('Cháo ngao bán được mấy nồi'): không cộng cả tiệm
    if set(parsed[1].split()) - _GENERIC_FILL - _UNIT_WORDS - {m[1], "m3", "noi", "gom", "het", "hang", "tinh", "theo", "cong", "mat", "tat", "ca", "do", "uong", "toi", "nay", "nguyen", "tho"}:
        return None
    x = shop.counted(_clip(parsed[0], today))
    hit = x["item"].map(lambda v: units.get(norm_key(v)) == m[1])
    return _ans("number", int(x.loc[hit, "qty_base"].sum()), "unit_total") if hit.any() else None


_FAMILIES.append(_unit_total)






def _top_cust_any(q, question, today, shop):
    """'khách nào tổng tiền mua hàng cao nhất', 'top 5 khách chi nhiều tiền nhất' khi câu lẫn chữ đệm lạ."""
    if not re.search(r"\bkhach\b", q) or not re.search(r"\bnhat\b", q) or not re.search(r"\b(tien|don|lan)\b", q) or _is_rate(question, q) or re.search(r"\b(no|nhom|nhap|cung cap)\b", q):
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    by = "money" if re.search(r"\btien\b", q) else "orders"
    d = shop.customers(_clip(parsed[0], today), by)
    if d.empty:
        return None
    k = re.search(r"\b(?:top )?(\d{1,2}) khach\b", q)
    return _ans("list", d["name"].astype(str).head(int(k[1])).tolist(), "top_customers2") if k else _ans("string", str(d["name"].iloc[0]), "top_customer2")


_FAMILIES[:0] = [_top_cust_any]


# ---------------------------------------------------- hỏi điều dữ liệu không ghi (rulebook §7)
_OPINION = re.compile(r"\b(lanh nghe|tan tam|quy men|hay khen|than thien|ngon nhat|chat luong tot|trung thuc|kho tinh|nhiet tinh|uy tin|dang tin cay|tay nghe|phuc vu)\b|\bba bau\b|\bmoi lan may\b|\bvi sao\b")


def _unanswerable(q, question, today, shop):
    if _OPINION.search(q) and not re.search(r"\b(doanh thu|tien ban)\b", q):
        return {"type": "unanswerable", "reason": "dữ liệu không ghi thông tin này", "template": "un_opinion"}
    if not re.search(r"\b(doanh thu|ban duoc|bao nhieu don|may don|so don)\b", q) or re.search(r"\b(nhap|no|ton|kho)\b", q):
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today or not _out_of_data(shop, _clip(parsed[0], today)):
        return None
    return {"type": "unanswerable", "reason": "kỳ hỏi nằm ngoài khoảng dữ liệu", "template": "un_period"}


# ---------------------------------------------------- tồn theo đơn vị lớn, giá trị tồn, nhập trừ bán, lãi ròng, lô
def _unit_table(shop) -> dict:
    return _once(shop, "_unit_tb", _make_unit_table)


def _make_unit_table(shop) -> dict:
    from .shop import _table_roles
    table, via = {}, {}
    names = {}
    if "item_code" in shop.lines:
        names = dict(zip(shop.lines["item_code"].astype(str), shop.lines["item"]))
    for t, rr in _table_roles(shop.biz).items():
        d = shop.biz.tables[t]
        c = _cols(d)
        # bảng quy đổi theo mã hàng: 'ĐVT', 'Quy đổi'
        big = rr.get("big_unit") or next((c[h] for h in ("dvt", "don_vi_tinh", "donvitinh", "donvitinh") if h in c and ("quy_doi" in c or "hesoquydoi" in c)), None)
        fac = rr.get("factor") or next((c[h] for h in ("quy_doi", "hesoquydoi", "bang") if h in c), None)
        key = rr.get("item")
        src = d[key] if key else None
        if key is None and big and fac:
            code = rr.get("item_code") or next((c[h] for h in ("ma_hang", "mahanghoa", "ma_sp") if h in c), None)
            if code:
                src = d[code].astype(str).map(names)
        if src is not None and big and fac:
            so = d[fac + "__so"] if fac + "__so" in d else d[fac]
            dv = d[fac + "__dv"] if fac + "__dv" in d else [None] * len(d)
            for a, u, f, w in zip(src, d[big], pd.to_numeric(so, errors="coerce"), dv):
                table[(norm_key(a), norm_key(u))] = f
                if w is not None and w == w:
                    via[(norm_key(a), norm_key(u))] = norm_key(w)
    for k, w in via.items():
        table[k] = table[k] * table.get((k[0], w), 1.0)
    return table


def _stock_table(shop):
    for t, df in shop.biz.tables.items():
        n = norm_col(t).lower()
        if "ton" not in n or "the_kho" in n:
            continue
        c = _cols(df)
        val = next((c[h] for h in ("ton_cuoi_ky", "ton_cuoi", "ton_thuc", "ton_kho") if h in c), None)
        name = next((c[h] for h in ("ten_hang", "hang_hoa", "product", "ten_san_pham") if h in c), None)
        code = next((c[h] for h in ("ma_hang", "id_sp", "ma_sp") if h in c), None)
        if val and name:
            return df, name, val, code
    return None


def _stock_big(q, question, today, shop):
    if not re.search(r"\b(chan|nguyen)\b", q) or not re.search(r"\ble\b", q) or shop.biz is None:
        return None
    last = shop.lines["date"].max().date()
    if today < last:
        return None
    st = _stock_table(shop)
    if st is None:
        return None
    df, name, val, _ = st
    hit = find_name(q, df[name].dropna().astype(str).tolist())
    if not hit:
        return None
    v = float(pd.to_numeric(df.loc[df[name] == hit, val], errors="coerce").sum())
    table = _unit_table(shop)
    units = [u for (a, u) in table if a == norm_key(hit)]
    big = [u for u in units if re.search(r"\b(?:may|bao nhieu) " + re.escape(u) + r"\b", q)]
    if len(big) < 1:
        return None
    f = table[(norm_key(hit), big[0])]
    return _ans("list", [int(v // f), int(v % f)], "stock_big") if f and f == f else None


def _stock_value(q, question, today, shop):
    if not re.search(r"\bgia von\b", q) or not re.search(r"\b(kho|ton|con)\b", q) or shop.biz is None:
        return None
    last = shop.lines["date"].max().date()
    d = re.search(_D, q)
    if today < last or (d and dt.date(int(d[3]), int(d[2]), int(d[1])) != last):
        return None
    st = _stock_table(shop)
    if st is None:
        return None
    df, name, val, code = st
    for t, cat in shop.biz.tables.items():
        c = _cols(cat)
        cost = next((c[h] for h in ("gia_von", "giavonhang", "gia_von_hien_tai") if h in c), None)
        key = next((c[h] for h in ("ma_hang", "mahanghoa", "ma_sp", "ma_h_hoa") if h in c), None) if code else next((c[h] for h in ("ten_hang", "hang", "ten_hang_hoa", "tengoi") if h in c), None)
        if cost and key and "chi_tiet" not in norm_col(t).lower() and "chitiet" not in norm_col(t).lower():
            m = dict(zip(cat[key].astype(str) if code else cat[key].map(norm_key), pd.to_numeric(cat[cost], errors="coerce")))
            ks = df[code].astype(str) if code else df[name].map(norm_key)
            v = pd.to_numeric(df[val], errors="coerce").fillna(0)
            return _ans("number", round_vnd((v.clip(lower=0) * ks.map(m).fillna(0)).sum()), "stock_value")
    return None


def _net_profit(q, question, today, shop):
    if not re.search(r"\blai rong\b|\bloi that\b|\btru (?:het )?chi phi\b", q) or "cost" not in shop.lines or "qty" not in shop.lines or shop.biz is None:
        return None
    parsed = _parse_period(q, today)
    if parsed is None or parsed[0].start > today:
        return None
    p = _clip(parsed[0], today)
    for t, df in shop.biz.tables.items():
        if "chi_phi" not in norm_col(t).lower() and "chiphi" not in norm_col(t).lower():
            continue
        c = _cols(df)
        date = next((c[h] for h in ("ngay_chi", "ngay", "ngay_mua") if h in c), None)
        val = next((c[h] for h in ("so_tien", "tien_chi", "tien") if h in c), None)
        if not (date and val):
            return None
        x = shop.counted(p)
        gross = (x["amount"] - pd.to_numeric(x["qty"], errors="coerce").fillna(0) * pd.to_numeric(x["cost"], errors="coerce").fillna(0)).sum()
        return _ans("number", round_vnd(gross - pd.to_numeric(df.loc[p.mask(pd.to_datetime(df[date], errors="coerce")), val], errors="coerce").sum()), "net_profit")
    return None


def _in_minus_out(q, question, today, shop):
    if not re.search(r"\bnhap\b", q) or not re.search(r"\btru\b|\bchenh\b", q) or not re.search(r"\bban\b", q):
        return None
    x = _purchase_lines(shop)
    parsed = _parse_period(q, today)
    if x is None or parsed is None or "unit" not in x:
        return None
    item = find_name(q, x["item"].dropna().astype(str).unique().tolist())
    if not item:
        return None
    y = x[(x["item"] == item) & parsed[0].mask(x["date"]) & ~x["cancelled"]]
    t = _unit_table(shop)
    k = norm_key(item)
    q_in = (y["qty"] * y["unit"].map(lambda u: t.get((k, norm_key(u)), 1.0))).sum()
    p = _clip(parsed[0], today)
    v = q_in - shop.qty(p, item)
    return _ans("number", int(v) if float(v).is_integer() else float(v), "in_minus_out")


def _lots(shop):
    return _once(shop, "_lots", _make_lots)


def _make_lots(shop):
    """Lô còn tồn: item, lot, expiry, left; None nếu shop không ghi lô."""
    if shop.biz is None:
        return None
    lot_t = sold_t = None
    for t, df in shop.biz.tables.items():
        c = _cols(df)
        lot = next((h for h in ("lo", "malo", "ma_lo") if h in c), None)
        exp = next((h for h in ("han_su_dung", "hansudung", "hsd") if h in c), None)
        qty_in = next((h for h in ("so_luong_dvt_co_ban", "slnhapquydoi", "so_luong_nhap", "sl_nhap_dvt_co_ban", "sl_nhap") if h in c), None)
        if lot and exp and qty_in and re.search(r"lo", norm_col(t).lower()) or (lot and exp and qty_in and "nhap" in norm_col(t).lower() and lot_t is None):
            lot_t = (df, c[lot], c[exp], c[qty_in], c)
        back = next((h for h in ("so_luong_tra_dvt_co_ban", "sltraquydoi", "so_luong_tra", "sl_tra_dvt_co_ban") if h in c), None)
        if lot and back:
            out = next((h for h in ("so_luong_dvt_co_ban", "slquydoi", "so_luong", "sl_quy_ra_le") if h in c), None)
            order = next((c[h] for h in ("ma_hoa_don", "mahoadon", "ma_don_hang") if h in c), None)
            if out and order:
                sold_t = (df, c[lot], c[out], c[back], order)
    if lot_t is None or sold_t is None:
        return None
    df, lot, exp, qty_in, c = lot_t
    code = next((c[h] for h in ("ma_hang", "mahanghoa", "ma_sku", "ma_mon", "ma_hh") if h in c), None)
    if code is None:
        return None
    s, slot, sout, sback, sorder = sold_t
    bad = set(shop.lines.loc[shop.lines["status"] == "huy", "order"].astype(str))
    s = s[~s[sorder].astype(str).isin(bad)]
    used = pd.to_numeric(s[sout], errors="coerce").fillna(0).groupby(s[slot]).sum() - pd.to_numeric(s[sback], errors="coerce").fillna(0).groupby(s[slot]).sum()
    g = pd.DataFrame({"lot": df[lot], "code": df[code].astype(str), "expiry": pd.to_datetime(df[exp], errors="coerce"), "in": pd.to_numeric(df[qty_in], errors="coerce").fillna(0)})
    g = g.groupby("lot", as_index=False).agg({"code": "first", "expiry": "first", "in": "sum"})
    g["left"] = g["in"] - g["lot"].map(used).fillna(0)
    if "item_code" in shop.lines:
        names = shop.lines.dropna(subset=["item_code"]).assign(k=lambda d: d["item_code"].astype(str)).drop_duplicates("k").set_index("k")["item"]
        g["item"] = g["code"].map(names)
    else:
        return None
    return g


def _lot_expiry(q, question, today, shop):
    if not re.search(r"\b(lo|hsd|han dung|het han|can date)\b", q):
        return None
    g = _lots(shop)
    if g is None:
        return None
    live = g[(g["left"] > 0) & (g["expiry"].dt.date >= today)]
    if re.search(r"\bgan het han nhat\b|\bcan date\b|\bhet han ngay\b|\bngay (?:may|nao)\b", q):
        item = find_name(q, shop.items)
        y = live[live["item"] == item] if item else None
        if y is None or y.empty:
            return None
        return _ans("date", y["expiry"].min().strftime("%Y-%m-%d"), "lot_nearest")
    return None


_FAMILIES[:0] = [_unanswerable, _stock_big, _stock_value, _net_profit, _in_minus_out, _lot_expiry]
