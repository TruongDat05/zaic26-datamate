"""Synthetic-book regressions for shop normalization, base units and sale costs."""
import re

import duckdb
import pandas as pd
import pytest

from datamate import ingest, views


def classify(text):
    k = ingest.norm(text)
    return bool(re.search(ingest.CANCELLED, k)), bool(re.search(ingest.RETURNED, k))


def test_a_check_note_is_not_a_return():
    assert classify("kiểm tra lại tuần sau") == (False, False)
    assert classify("Kiểm tra hàng") == (False, False)
    for returned in ("trả lại hàng", "khách trả lại", "Trả hàng", "hàng trả về", "đổi trả", "khách mang trả"):
        assert classify(returned) == (False, True), returned
    assert views.status_kind("kiểm tra lại tuần sau") is None


def shop(tmp_path, files, readme="Sổ bán hàng."):
    folder = tmp_path / "data" / "shop"
    folder.mkdir(parents=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    (folder / "README.md").write_text(readme, encoding="utf-8")
    return tmp_path


def coffee_csv():
    rows = ["Mã đơn,Ngày,Tên sản phẩm,Số lượng,Đơn giá,Thành tiền,Đơn vị"]
    for i in range(40):
        rows.append(f"D{i:03d},0{1 + i % 9}/03/2025,Trà sữa (M),1,40000,40000,ly")
        rows.append(f"D{i:03d},0{1 + i % 9}/03/2025,Topping Trân châu,1,7000,7000,phần")
    for i in range(12):  # the drink and its toppings in one cell; the topping money is in the line amount
        rows.append(f"T{i:03d},10/03/2025,Trà Sữa (m) + 2 Trân châu (14.000đ),2,40000,94000,ly")
    rows.append("T100,11/03/2025,Trà sữa (M) + Trân châu (7.000đ) + Kem (10k),1,40000,57000,ly")
    return "\n".join(rows) + "\n"


def test_toppings_in_the_cell_become_lines_of_their_own(tmp_path):
    db, _ = ingest.load_business(shop(tmp_path, {"ban_hang.csv": coffee_csv()}), "shop")
    rows = db.execute("""SELECT "Tên sản phẩm", sum("Số lượng"), sum("Thành tiền"), any_value("Đơn vị") FROM ban_hang
                         WHERE "Mã đơn" LIKE 'T%' GROUP BY 1 ORDER BY 1""").fetchall()
    assert rows == [("Kem", 1, 10000, "ly"),  # no line of its own: its name as written, the row's unit
                    ("Topping Trân châu", 25, 175000, "phần"),  # 12 x 2 + 1 portions, 12 x 14000 + 7000, the topping's own unit
                    ("Trà sữa (M)", 25, 12 * 80000 + 40000, "ly")]  # the drink keeps the line's qty; its amount is the line's less the toppings
    total = db.execute('SELECT sum("Thành tiền"), count(*) FROM ban_hang').fetchone()
    assert total == (40 * 47000 + 12 * 94000 + 57000, 80 + 24 + 3)  # no money gained or lost


def test_names_with_a_plus_and_no_price_stay():
    df = pd.DataFrame({"Hoạt chất": ["Paracetamol + Cafein"] * 30, "SL": [1.0] * 30, "Giá": [5.0] * 30, "Tiền": [5.0] * 30})
    df["Hoạt chất"] = df["Hoạt chất"].astype("string")
    assert ingest.split_toppings(df).equals(df)


def test_call_names_map_to_the_staff_list(tmp_path):
    book = "Ngày,Xe,Thợ,Tiền\n" + "".join(f"0{1 + i % 9}/07/2025,Wave {i},{t},100000\n"
                                            for i, t in enumerate(["Phúc", "a Phúc", "A.Phúc", "a Đức", "Đức", "A.Đức", "An"] * 5))
    root = shop(tmp_path, {"so_sua_xe.csv": book, "danh_sach_tho.csv": "Họ tên,Hoa hồng\nĐoàn Bảo Phúc,25\nLương Chi Đức,30\nĐàm Lan An,28\n"})
    db, _ = ingest.load_business(root, "shop", canon=True)
    got = dict(db.execute('SELECT "Thợ", count(*) FROM so_sua_xe GROUP BY 1').fetchall())
    assert got == {"Đoàn Bảo Phúc": 15, "Lương Chi Đức": 15, "Đàm Lan An": 5}
    customers = ingest.people_map({"anh Minh": 2, "Minh": 3, "Anh Thư": 1}, [])
    assert customers == {"anh Minh": "Minh"}  # no roster: only the form of address goes; the given name 'Anh Thư' stays


def phone_csv():
    """Order cells (customer, discount, paid) on the first line of an order only, as a hand-kept book writes them."""
    rows = ["Ngày,STT,Khách,Hàng,SL,Giá,Thành tiền,Bớt,Khách trả,Ghi chú"]
    for o in range(1, 41):
        note = "khách trả lại" if o % 10 == 0 else ""
        rows.append(f"0{1 + o % 9}/06/2025,{o},Khách {o},Máy A{o % 3},1,1000000,1000000,10000,{1000000 + 50000 * (o % 4) - 10000},{note}")
        for k in range(o % 4):
            rows.append(f"0{1 + o % 9}/06/2025,,,Ốp {k},1,50000,50000,,,")
    return "\n".join(rows) + "\n"


def phone_shop(tmp_path):
    return ingest.load_business(shop(tmp_path, {"so_ban.csv": phone_csv()}), "shop")


def test_order_level_amounts_get_an_order_table_and_a_note(tmp_path):
    db, meta = phone_shop(tmp_path)
    t = next(t for t in meta["tables"] if t["name"] == "so_ban")
    assert t["order_level"] == {"id": "STT", "cols": ["Bớt", "Khách trả"], "table": "so_ban_don"}
    assert "ORDER-LEVEL" in ingest.schema_text(meta)
    paid = sum(1000000 + 50000 * (o % 4) - 10000 for o in range(1, 41) if o % 10)
    assert db.execute('SELECT sum("Khách trả") FROM so_ban_don WHERE NOT "_don_tra"').fetchone()[0] == paid
    assert db.execute("SELECT count(*) FROM so_ban_don").fetchone()[0] == 40


def test_loan_payments_of_equal_size_are_not_order_level(tmp_path):
    rows = ["STT,Ngày góp,Khách,Đã góp"] + [f"{o},{1 + k}/0{1 + k % 9}/2025,Khách {o},500000" for o in range(1, 41) for k in range(3)]
    db, meta = ingest.load_business(shop(tmp_path, {"so_gop.csv": "\n".join(rows) + "\n"}), "shop")
    assert not any(t.get("order_level") for t in meta["tables"])  # every row is a payment: no line of an order differs in an item


def pharmacy_db():
    db = duckdb.connect()
    db.execute('CREATE TABLE quy_doi ("Tên sản phẩm" VARCHAR, "Đơn vị quy đổi" VARCHAR, "Tỉ lệ quy đổi" DOUBLE, '
               '"Tỉ lệ quy đổi (đơn vị ghi kèm)" VARCHAR, "Đơn vị cơ bản" VARCHAR)')
    db.execute("INSERT INTO quy_doi VALUES ('Thuốc A', 'vỉ', 4, NULL, 'viên'), ('Thuốc A', 'hộp', 25, 'vỉ', 'viên'), ('Siro B', 'thùng', 24, NULL, 'chai')")
    db.execute('CREATE TABLE ban ("Mã đơn" VARCHAR, "Ngày" DATE, "Tên" VARCHAR, "SL" DOUBLE, "ĐVT" VARCHAR, "Tiền" DOUBLE, "Giá vốn" DOUBLE)')
    lines = [("viên", 1, 1200, 900), ("vỉ", 1, 4800, 3600), ("hộp", 1, 130000, 90000)]
    k = 0
    for i in range(30):
        for u, q, money, cost in lines:
            db.execute("INSERT INTO ban VALUES (?, DATE '2025-03-01' + ?, 'Thuốc A', ?, ?, ?, ?)", [f"O{k}", i, q, u, money, cost])
            k += 1
        for u, q, money, cost in (("chai", 2, 40000, 30000), ("thùng", 1, 450000, 340000)):
            db.execute("INSERT INTO ban VALUES (?, DATE '2025-03-01' + ?, 'Siro B', ?, ?, ?, ?)", [f"O{k}", i, q, u, money, cost])
            k += 1
    return db


def mapping(cost=True):
    return ('SELECT "Mã đơn" AS order_id, "Ngày" AS order_date, NULL AS order_ts, NULL AS product_code, NULL AS product_group, "Tên" AS product_name, '
            '"SL" AS qty, "ĐVT" AS unit, "Tiền" AS amount, ' + ('"SL" * "Giá vốn" AS cost, ' if cost else "")
            + "NULL AS customer_id, NULL AS customer_name, NULL AS staff, NULL AS channel, NULL AS payment, false AS is_cancelled, "
            "false AS is_returned, 'item' AS line_kind FROM ban")


def test_items_in_several_units_fail_the_draft_and_convert_through_the_table():
    db = pharmacy_db()
    views.build(db, mapping(), "_cand_")
    with pytest.raises(ValueError, match="BASE unit"):
        views.unit_test(db, "_cand_")
    fixed = views.qty_by_table(db, mapping())
    views.build(db, fixed)
    assert views.multi_unit_items(db, "v_sales_all") == []
    got = dict(db.execute("SELECT product_name || '/' || unit, sum(qty) FROM v_sales GROUP BY 1").fetchall())
    assert got == {"Thuốc A/viên": 30 * (1 + 4 + 100), "Siro B/chai": 30 * (2 + 24)}  # hộp = 25 vỉ x 4 viên
    views.unit_test(db, "v_")


def test_a_converted_line_is_not_converted_again():
    db = pharmacy_db()
    db.execute("""UPDATE ban SET "SL" = "SL" * 100, "ĐVT" = 'hộp' WHERE "ĐVT" = 'hộp'""")  # qty already in viên, unit still the line's
    views.build(db, views.qty_by_table(db, mapping()))
    assert db.execute("SELECT sum(qty) FROM v_sales WHERE product_name = 'Thuốc A'").fetchone()[0] == 30 * 105


def test_a_small_factor_is_not_applied_twice():
    db = pharmacy_db()
    db.execute("""UPDATE quy_doi SET "Tỉ lệ quy đổi" = 2 WHERE "Đơn vị quy đổi" = 'thùng'""")  # a pack of 2 bottles
    db.execute("""UPDATE ban SET "Tiền" = 40000 WHERE "ĐVT" = 'thùng'""")  # priced as two bottles (20000 each)
    views.build(db, views.qty_by_table(db, mapping()))
    assert db.execute("SELECT sum(qty) FROM v_sales WHERE product_name = 'Siro B'").fetchone()[0] == 30 * (2 + 2)  # 1 pack -> 2 bottles
    db.execute("""UPDATE ban SET "SL" = 2 WHERE "ĐVT" = 'thùng'""")  # a draft that already wrote the pack in bottles
    views.build(db, views.qty_by_table(db, mapping()))
    assert db.execute("SELECT sum(qty) FROM v_sales WHERE product_name = 'Siro B'").fetchone()[0] == 30 * (2 + 2)  # not 2 x 2


def test_cost_column_and_old_mappings_without_it():
    db = pharmacy_db()
    views.build(db, mapping(cost=False))  # a mapping saved before the cost column still builds
    assert db.execute("SELECT count(cost) FROM v_sales").fetchone()[0] == 0
    views.build(db, mapping())
    assert "Gross profit (lãi gộp) = SUM(amount) - SUM(cost)" in views.describe(db, mapping(), 8, 8)
    bad = mapping().replace('"SL" * "Giá vốn" AS cost', '"SL" * "Giá vốn" * 100 AS cost')
    views.build(db, bad, "_cand_")
    with pytest.raises(ValueError, match="SALE unit"):
        views.cost_test(db, "_cand_")


def test_describe_names_the_unit_filter_and_groups():
    db = pharmacy_db()
    views.build(db, views.qty_by_table(db, mapping()))
    text = views.describe(db, mapping(), 8, 8)
    assert "WHERE unit = '<unit>'" in text and "'chai', 'viên'" in text
    assert "never inferred from other columns" in text  # product_group is always NULL here
