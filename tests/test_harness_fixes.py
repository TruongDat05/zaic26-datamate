"""Synthetic-book regressions for shop normalization and order-level tables."""
import re

import pandas as pd

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
