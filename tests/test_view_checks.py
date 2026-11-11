"""Data checks on a drafted view (status flags, discount taken twice) and the loader's order-date pass for return lines."""
import datetime
import re

import duckdb
import pytest

from datamate import ingest, views

LINES = [  # order, status, list price, qty, promo, line total after promo
    *[(f"O{i:03d}", "đã lấy" if i % 3 else "Thành công", 50000.0, 2.0, 0.0 if i % 4 else 5000.0, 100000.0 - (0.0 if i % 4 else 5000.0)) for i in range(60)],
    *[(f"C{i:03d}", "hủy" if i % 2 else "Hủy", 30000.0, 1.0, 0.0, 30000.0) for i in range(12)],
    *[(f"R{i:03d}", "trả lại", 20000.0, 1.0, 0.0, 20000.0) for i in range(8)],
]


def raw(table="sales_book", oid="Mã phiếu", status="Ghi chú TT"):
    db = duckdb.connect()
    db.execute(f'CREATE TABLE "{table}" ("{oid}" VARCHAR, "Ngày" DATE, "{status}" VARCHAR, "Giá" DOUBLE, "SL" DOUBLE, "KM" DOUBLE, "Thành tiền" DOUBLE, "_huy" BOOLEAN, "_tra" BOOLEAN)')
    for o, st, p, q, km, a in LINES:
        k = ingest.norm(st)
        db.execute(f'INSERT INTO "{table}" VALUES (?, DATE \'2025-03-01\', ?, ?, ?, ?, ?, ?, ?)',
                   [o, st, p, q, km, a, bool(re.search(ingest.CANCELLED, k)), bool(re.search(ingest.RETURNED, k))])
    return db


def mapping(table="sales_book", oid="Mã phiếu", status="Ghi chú TT", cancel=("hủy", "Hủy"), ret=("trả lại",), amount='"Thành tiền"'):
    lst = lambda xs: ", ".join("'" + x + "'" for x in xs)
    return f"""SELECT "{oid}" AS order_id, "Ngày" AS order_date, NULL AS order_ts, NULL AS product_code, NULL AS product_group, 'Hàng A' AS product_name,
        "SL" AS qty, NULL AS unit, {amount} AS amount, NULL AS customer_id, NULL AS customer_name, NULL AS staff, NULL AS channel, NULL AS payment,
        "{status}" IN ({lst(cancel)}) AS is_cancelled, "{status}" IN ({lst(ret)}) AS is_returned, 'item' AS line_kind FROM "{table}" """


def run(db, sql):
    views.build(db, sql, "_cand_")
    views.status_test(db, "_cand_", sql)
    views.discount_test(db, "_cand_", sql)


def test_right_view_passes():
    db = raw()
    run(db, mapping())


def test_done_status_flagged_returned_is_caught():
    db = raw()
    with pytest.raises(ValueError, match="'đã lấy' is a done/delivering status"):
        run(db, mapping(ret=("trả lại", "đã lấy")))


def test_missed_cancel_spelling_is_caught():
    db = raw()
    with pytest.raises(ValueError, match="'Hủy' is a cancel status"):
        run(db, mapping(cancel=("hủy",)))


def test_same_rules_under_other_names():
    db = raw(table="hoa_don", oid="Order ID", status="Trạng thái")
    run(db, mapping(table="hoa_don", oid="Order ID", status="Trạng thái"))
    with pytest.raises(ValueError, match="'Thành công' is a done"):
        run(db, mapping(table="hoa_don", oid="Order ID", status="Trạng thái", cancel=("hủy", "Hủy", "Thành công")))


def test_own_order_id_is_skipped():
    db = raw()
    run(db, mapping(ret=("trả lại", "đã lấy")).replace('"Mã phiếu" AS order_id', "'X-' || \"Mã phiếu\" AS order_id"))


def test_discount_taken_twice_is_caught_and_once_passes():
    db = raw()
    with pytest.raises(ValueError, match="counts the discount twice"):
        run(db, mapping(amount='("Thành tiền" - coalesce("KM", 0))'))
    db.execute('UPDATE sales_book SET "Thành tiền" = "Giá" * "SL"')  # now a gross total: taking the promo off is right
    run(db, mapping(amount='("Thành tiền" - "KM")'))


def test_return_line_dated_later_takes_the_order_date():
    db = duckdb.connect()
    cols = [("Mã phiếu", "VARCHAR"), ("Ngày lập", "TIMESTAMP"), ("Hạn dùng", "TIMESTAMP"), ("_huy", "BOOLEAN"), ("_tra", "BOOLEAN")]
    for t in ("book_t1", "book_t2"):
        db.execute(f'CREATE TABLE {t} (' + ", ".join(f'"{c}" {ty}' for c, ty in cols) + ")")
    db.execute("INSERT INTO book_t1 VALUES ('A', TIMESTAMP '2025-01-30', TIMESTAMP '2026-01-01', false, true), "
               "('B', TIMESTAMP '2025-01-10', NULL, false, false), ('B', TIMESTAMP '2025-01-12', NULL, false, false)")
    db.execute("INSERT INTO book_t2 VALUES ('A', TIMESTAMP '2025-02-02', TIMESTAMP '2026-05-01', false, true)")
    assert ingest.order_dates(db, ["book_t1", "book_t2"], cols) == 1
    assert db.execute('SELECT CAST("Ngày lập" AS DATE), CAST("Hạn dùng" AS DATE) FROM book_t2').fetchone() == (
        datetime.date(2025, 1, 30), datetime.date(2026, 5, 1))  # expiry untouched
    assert db.execute("""SELECT count(DISTINCT "Ngày lập") FROM book_t1 WHERE "Mã phiếu" = 'B'""").fetchone()[0] == 2  # a valid order is kept as recorded
    assert ingest.order_dates(db, ["book_t1", "book_t2"], cols) == 0
