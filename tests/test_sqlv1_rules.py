"""Hồi quy oracle btc_v1 (eval/sqlv1/world.py) theo rules_spec: mỗi quy ước một trường hợp dựng tay.
Phần SQL (phân loại trạng thái trong SELECT ánh xạ) cần duckdb: chạy bằng venv upstream, venv cũ thì bỏ qua."""
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval")]
from sqlv1 import world as WB  # noqa: E402

TODAY = "2025-06-20"


def base_rec(by_code=True):
    items = [{"item": "A01", "name": "Bia lon", "group": "Đồ uống", "base": "lon", "price": 10000},
             {"item": "A02", "name": "Mực nướng", "group": "Món nhậu", "base": "đĩa", "price": 50000},
             {"item": "A03", "name": "Nước ngọt", "group": "Đồ uống", "base": "lon", "price": 8000}]
    units = [{"item": "A01", "unit": "lon", "factor": 1}, {"item": "A01", "unit": "thùng", "factor": 24},
             {"item": "A02", "unit": "đĩa", "factor": 1}, {"item": "A03", "unit": "lon", "factor": 1}]
    customers = [{"cust": "KH001", "name": "Lan", "kind": "Khách quen"}, {"cust": "KH002", "name": "Hùng", "kind": "Khách quen"},
                 {"cust": "KH003", "name": "Lan", "kind": "Công ty"}]
    return {"industry": "quan_nhau", "shop_name": "Quán thử", "seed": 1, "start": "2025-05-01", "today": TODAY, "data_end": TODAY,
            "items": items, "units": units, "customers": customers, "orders": [], "lines": [], "returns": [], "payments": [], "purchases": [],
            "lots": [], "features": {"debt": True, "lots": False, "purchases": True}, "opts": {"codes_on_orders": by_code},
            "status_text": {k: [v[0]] for k, v in WB.TEXT.items()}, "walkin_text": ["Khách lẻ"]}


def add(rec, oid, at, status="done", cust=None, lines=(("A01", "lon", 2, 10000),), payment="Tiền mặt", channel="Tại quán", ship=0, voucher=0, discount=0, kind="item"):
    rec["orders"].append({"order": oid, "at": at, "status": status, "status_text": WB.TEXT[status][0], "channel": channel, "cust": cust,
                          "walkin_text": "Khách lẻ" if cust is None else None, "payment": payment, "ship_fee": ship, "voucher": voucher, "note": ""})
    for j, (it, u, qn, pr) in enumerate(lines, 1):
        rec["lines"].append({"order": oid, "line": j, "item": it, "unit": u, "qty": qn, "price": pr, "discount": discount if j == 1 else 0,
                             "amount": qn * pr - (discount if j == 1 else 0), "kind": kind})


def g(rec, slot, period=None, **params):  # noqa: D103
    spec = {"slot": slot, "period": period or {"kind": "month", "y": 2025, "m": 6}, "base_period": None, "params": params, "as_of": TODAY}
    return WB.gold(rec, spec)["answer"]


def test_in_transit_counts_as_normal_order():
    r = base_rec()
    add(r, "D1", "2025-06-01 10:00:00", "done")
    add(r, "D2", "2025-06-02 10:00:00", "transit")
    assert g(r, "revenue") == 40000 and g(r, "n_orders") == 2


def test_partial_return_excludes_whole_order_and_is_not_cancel():
    r = base_rec()
    add(r, "D1", "2025-06-01 10:00:00", "done")
    add(r, "D2", "2025-06-02 10:00:00", "returned", lines=(("A01", "lon", 5, 10000), ("A02", "đĩa", 1, 50000)))
    r["returns"].append({"ret": "T1", "order": "D2", "at": "2025-06-25 09:00:00", "item": "A01", "unit": "lon", "qty": 1, "amount": 10000})  # phiếu trả sau hôm nay
    assert g(r, "revenue") == 20000 and g(r, "n_orders") == 1 and g(r, "qty_base", item="A01") == 2
    assert g(r, "n_cancelled") == 0 and g(r, "n_returned") == 1


def test_undeliverable_is_cancel_and_cancel_rate_counts_all_statuses():
    r = base_rec()
    add(r, "D1", "2025-06-01 10:00:00", "done")
    add(r, "D2", "2025-06-02 10:00:00", "cancel")
    r["orders"][-1]["status_text"] = "Không giao được"
    add(r, "D3", "2025-06-03 10:00:00", "returned")
    add(r, "D4", "2025-06-04 10:00:00", "transit")
    assert g(r, "n_cancelled") == 1 and g(r, "cancel_rate") == 25.0


def test_ties_broken_by_code_and_walkins_excluded():
    r = base_rec()
    add(r, "D1", "2025-06-01 10:00:00", cust="KH002", lines=(("A03", "lon", 4, 8000),))
    add(r, "D2", "2025-06-01 11:00:00", cust="KH001", lines=(("A01", "lon", 4, 8000),))
    add(r, "D3", "2025-06-01 12:00:00", cust=None, lines=(("A02", "đĩa", 9, 50000),))  # khách lẻ mua nhiều nhất
    assert g(r, "top_items", k=2) == ["Mực nướng", "Bia lon"]  # A01 = A03 = 4 lon: mã A01 trước
    assert g(r, "top_customer") == "Lan"  # KH001 = KH002 = 32000: mã nhỏ hơn
    assert g(r, "n_customers") == 2


def test_same_name_two_codes_are_two_customers_but_one_by_name():
    r = base_rec(by_code=True)
    add(r, "D1", "2025-06-01 10:00:00", cust="KH001")
    add(r, "D2", "2025-06-02 10:00:00", cust="KH003")
    assert g(r, "n_customers") == 2
    r2 = copy.deepcopy(r)
    r2["opts"]["codes_on_orders"] = False
    assert g(r2, "n_customers") == 1


def test_name_tie_without_codes_is_ambiguous():
    r = base_rec(by_code=False)
    add(r, "D1", "2025-06-01 10:00:00", cust="KH001")
    add(r, "D2", "2025-06-02 10:00:00", cust="KH002")
    with pytest.raises(WB.Degenerate):
        g(r, "top_customer")


def test_line_discount_voucher_ship_and_service_fee():
    r = base_rec()
    add(r, "D1", "2025-06-01 10:00:00", lines=(("A02", "đĩa", 2, 50000),), discount=10000, voucher=20000, ship=15000)
    add(r, "D1S", "2025-06-01 10:00:00")
    r["orders"].pop()
    r["lines"].append({"order": "D1", "line": 2, "item": "PV", "unit": "lần", "qty": 1, "price": 5000, "amount": 5000, "discount": 0, "kind": "service"})
    assert g(r, "revenue") == 90000 + 5000  # trừ chiết khấu dòng, không trừ voucher, không cộng phí giao, có phí phục vụ
    assert g(r, "top_items", k=1) == ["Mực nướng"] and g(r, "share_group", group="Món nhậu") == round(100 * 90000 / 95000, 2)


def test_units_convert_to_base():
    r = base_rec()
    add(r, "D1", "2025-06-01 10:00:00", lines=(("A01", "thùng", 2, 230000), ("A01", "lon", 3, 10000)))
    assert g(r, "qty_base", item="A01") == 51


def test_debt_order_level_with_ship_partial_payments_and_cod():
    r = base_rec()
    add(r, "D1", "2025-06-01 10:00:00", cust="KH001", payment="Ghi nợ", lines=(("A02", "đĩa", 2, 50000),), ship=20000, voucher=10000)
    r["payments"] += [{"payment": "P1", "order": "D1", "at": "2025-06-01 10:05:00", "amount": 30000},
                      {"payment": "P2", "order": "D1", "at": "2025-06-10 09:00:00", "amount": 40000}]
    add(r, "D2", "2025-06-03 10:00:00", cust="KH002", payment="COD", lines=(("A02", "đĩa", 1, 50000),))
    add(r, "D3", "2025-06-04 10:00:00", cust="KH002", payment="Ghi nợ", status="returned", lines=(("A01", "lon", 3, 10000),))
    spec = lambda slot, kind, d=None: {"slot": slot, "period": {"kind": kind, **({"d": d} if d else {})}, "base_period": None, "params": {}, "as_of": TODAY}  # noqa: E731
    assert WB.gold(r, spec("debt_total", "until", "2025-06-05"))["answer"] == 110000 - 30000 + 30000  # D1 còn 80000, D3 (trả hàng) vẫn nợ 30000
    assert WB.gold(r, spec("debt_total", "until_today"))["answer"] == 40000 + 30000
    assert WB.gold(r, {**spec("debt_new", "month"), "period": {"kind": "month", "y": 2025, "m": 6}})["answer"] == 80000 + 30000
    assert WB.gold(r, {**spec("debt_collected", "month"), "period": {"kind": "month", "y": 2025, "m": 6}})["answer"] == 40000


def test_unsold_needs_sales_before_period_and_future_rows_ignored():
    r = base_rec()
    add(r, "D1", "2025-05-10 10:00:00", lines=(("A01", "lon", 1, 10000), ("A02", "đĩa", 1, 50000)))
    add(r, "D2", "2025-06-05 10:00:00", lines=(("A01", "lon", 1, 10000),))
    add(r, "D3", "2025-06-21 10:00:00", lines=(("A02", "đĩa", 9, 50000),))  # sau hôm nay: chưa xảy ra
    r["data_end"] = "2025-06-22"
    assert g(r, "unsold_items", period={"kind": "this_month"}) == 1 and g(r, "revenue", period={"kind": "this_month"}) == 10000


def test_named_period_past_today_with_later_data_is_ambiguous():
    r = base_rec()
    add(r, "D1", "2025-06-05 10:00:00")
    r["data_end"] = "2025-06-22"
    with pytest.raises(WB.Degenerate):
        WB.gold(r, {"slot": "revenue", "period": {"kind": "month", "y": 2025, "m": 6}, "base_period": None, "params": {}, "as_of": "2025-06-10"})


def test_period_outside_data_is_unanswerable():
    r = base_rec()
    add(r, "D1", "2025-06-05 10:00:00")
    assert WB.gold(r, {"slot": "revenue", "period": {"kind": "month", "y": 2024, "m": 3}, "base_period": None, "params": {}, "as_of": TODAY}) == {"type": "unanswerable"}


def test_shift_boundaries():
    r = base_rec()
    add(r, "D1", "2025-06-01 13:59:59")
    add(r, "D2", "2025-06-01 14:00:00", lines=(("A02", "đĩa", 1, 50000),))
    add(r, "D3", "2025-06-01 22:30:00", lines=(("A03", "lon", 1, 8000),))
    assert g(r, "shift_rev", shift="sang") == 20000 and g(r, "shift_rev", shift="chieu") == 50000


def test_mapping_status_classifier_word_boundaries():
    duckdb = pytest.importorskip("duckdb")
    from sqlv1 import render as RD
    db = duckdb.connect()
    cases = {"Đang vận chuyển": (False, False), "Không giao được": (True, False), "Khách hủy": (True, False), "huỷ": (True, False),
             "Đổi trả": (False, True), "khách trả 1 phần": (False, True), "Trả lại hàng": (False, True), "Đã thanh toán": (False, False),
             "Hoàn thành": (False, False), "giao buổi sáng": (False, False)}
    for text, want in cases.items():
        got = db.execute(f"SELECT {RD.is_cancel('x')}, {RD.is_return('x')} FROM (SELECT ? AS x)", [text]).fetchone()
        assert tuple(got) == want, text


def _q(slot, typ="number", question="Doanh thu tháng 6/2025 bao nhiêu?", period=None):
    return {"spec": {"slot": slot, "period": period or {"kind": "month", "y": 2025, "m": 6}, "as_of": TODAY, "params": {}}, "gold": {"type": typ, "answer": 1},
            "question": question}


def test_semantic_checks_catch_coincidental_convention_errors():
    pytest.importorskip("duckdb")
    from sqlv1 import verify as V
    roles = {"c_rx": "Kê đơn", "l_item": "Mã hàng", "c_code": "Mã hàng"}
    rev = "SELECT ROUND(SUM(amount)) FROM v_sales WHERE line_kind = 'item' AND order_date >= DATE '2025-06-01' AND order_date < DATE '2025-07-01'"
    assert "service_lines_dropped" in V.semantic_check(rev, [], _q("revenue"), roles, True)
    share = "SELECT ROUND(100.0 * SUM(amount) FILTER (WHERE line_kind = 'item' AND product_group = 'Bia') / SUM(amount), 2) FROM v_sales"
    assert V.semantic_check(share, [], _q("share_group", "percent"), roles, True) == []
    rx = "SELECT ROUND(SUM(amount)) FROM v_sales WHERE product_group = 'Thuốc kê đơn'"
    assert "rx_not_from_catalog_flag" in V.semantic_check(rx, [], _q("rx_rev"), roles, True)
    top = "SELECT product_name FROM v_sales GROUP BY product_name ORDER BY SUM(qty) DESC, product_name LIMIT 3"
    assert "item_tie_not_by_code" in V.semantic_check(top, [], _q("top_items", "list", "Top 3 món bán chạy tháng 6/2025?"), roles, True)
    uns = "SELECT COUNT(*) FROM cat WHERE code NOT IN (SELECT product_code FROM v_sales WHERE order_date >= DATE '2025-06-01')"
    assert "unsold_without_prior_sales" in V.semantic_check(uns, [], _q("unsold_items"), roles, True)
    assert "notes_code_not_in_sql" in V.semantic_check("SELECT 1", ["mã hàng XD005"], _q("revenue"), roles, True)
    assert "constant_case" in V.semantic_check("SELECT SUM(qty * CASE WHEN unit = 'thùng' THEN 1.0 ELSE 1.0 END) FROM v_sales", [], _q("qty_base"), roles, True)
    assert "price_from_valid_orders_only" in V.semantic_check("SELECT DISTINCT amount / qty FROM v_sales WHERE product_name = 'Bia lon'", [], _q("price_on_day"), roles, True)
    assert V.question_ok(_q("top_customers", "list", "3 khách mua nhiều nhất tháng 6/2025 là ai, dthu bn?")) == ["name_question_asks_amount"]
    assert V.question_ok(_q("top_customers", "list", "3 khách mua nhiều nhất tháng 6/2025 là ai?")) == []


def test_unsafe_status_substring_predicate_flagged():
    """QA của bundle SFT v2: mẫu LIKE một từ trạng thái ('%huy%' khớp 'vận chuyển') bị loại; cụm nhiều từ theo luật thì không."""
    pytest.importorskip("duckdb")  # sqlv1.bundle nạp upstream (venv upstream), như các test cần upstream khác
    from sqlv1.bundle import unsafe_status_predicate
    assert unsafe_status_predicate("WHERE NOT strip_accents(lower(st)) LIKE '%huy%'") == ["%huy%"]
    assert unsafe_status_predicate("WHERE lower(x) ILIKE '%Hủy%' OR y LIKE '%tra%'") == ["%Hủy%", "%tra%"]
    assert unsafe_status_predicate("WHERE s LIKE '%khong giao duoc%' OR s LIKE '%tra hang%' OR regexp_matches(k, '(^|_)huy(_|$)')") == []
