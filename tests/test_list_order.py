"""Rulebook list order for items listed by expiry: the query's final ORDER BY must start with the expiry, ascending.
Each case runs on two shops with different table/column names and row order (metamorphic: the rule reads query structure only)."""
import random

import duckdb
import pytest

from datamate.sqlexec import execute, expiry_order_error

LOTS = [("L1", "P03", "2026-08-01", 5), ("L2", "P01", "2026-07-01", 4), ("L3", "P02", "2026-06-10", 2),
        ("L4", "P01", "2026-06-20", 0), ("L5", "P04", "2027-01-01", 3), ("L6", "P02", "2026-09-01", 1)]
ITEMS = [("P01", "Zeta siro"), ("P02", "Alpha viên"), ("P03", "Beta gel"), ("P04", "Gamma kem")]
# in stock (qty > 0) with expiry in [2026-06-01, 2026-09-30]: P02 (06-10), P01 (07-01; its 06-20 lot is empty), P03 (08-01)
GOLD = ["Alpha viên", "Zeta siro", "Beta gel"]

SHOPS = {
    "vi": {"lot": "lo_hang", "item": "hang_hoa", "lot_id": "Lô", "code": "Mã hàng", "exp": "Hạn dùng", "qty": "SL tồn", "name": "Tên hàng"},
    "en": {"lot": "batches", "item": "products", "lot_id": "batch_no", "code": "sku", "exp": "expiry_date", "qty": "on_hand", "name": "product_name"},
}
Q = "những sản phẩm nào còn hàng có hạn dùng từ 01/06/2026 đến hết 30/09/2026"


@pytest.fixture(params=sorted(SHOPS))
def shop(request):
    c = SHOPS[request.param]
    con = duckdb.connect()
    con.execute(f'CREATE TABLE "{c["item"]}" ("{c["name"]}" VARCHAR, "{c["code"]}" VARCHAR)')
    con.execute(f'CREATE TABLE "{c["lot"]}" ("{c["exp"]}" DATE, "{c["qty"]}" INTEGER, "{c["code"]}" VARCHAR, "{c["lot_id"]}" VARCHAR)')
    rnd = random.Random(request.param)
    lots, items = LOTS[:], ITEMS[:]
    rnd.shuffle(lots), rnd.shuffle(items)
    con.executemany(f'INSERT INTO "{c["item"]}" VALUES (?, ?)', [(n, k) for k, n in items])
    con.executemany(f'INSERT INTO "{c["lot"]}" VALUES (?, ?, ?, ?)', [(e, q, k, l) for l, k, e, q in lots])
    return con, c


def query(c, order):
    return (f'TYPE: list\n```sql\nWITH kept AS (SELECT "{c["code"]}" AS code, "{c["exp"]}" AS exp_day FROM "{c["lot"]}"\n'
            f'  WHERE "{c["qty"]}" > 0 AND "{c["exp"]}" >= DATE \'2026-06-01\' AND "{c["exp"]}" <= DATE \'2026-09-30\')\n'
            f'SELECT i."{c["name"]}" FROM kept k JOIN "{c["item"]}" i ON i."{c["code"]}" = k.code\n'
            f'GROUP BY i."{c["code"]}", i."{c["name"]}"\n{order}\n```')


def test_earliest_expiry_order_passes_and_is_the_rulebook_order(shop):
    con, c = shop
    out = execute(con, query(c, f'ORDER BY MIN(k.exp_day), i."{c["code"]}"'), Q)
    assert out == {"type": "list", "answer": GOLD}


@pytest.mark.parametrize("order", ['ORDER BY i."{name}"', 'ORDER BY i."{code}"', "", 'ORDER BY MIN(k.exp_day) DESC',
                                   'ORDER BY i."{code}", MIN(k.exp_day)'])
def test_other_orders_go_to_repair(shop, order):
    con, c = shop
    with pytest.raises(ValueError, match="LIST ORDER"):
        execute(con, query(c, order.format(**c)), Q)


def test_order_hidden_in_a_comment_or_window_does_not_count(shop):
    con, c = shop
    with pytest.raises(ValueError, match="LIST ORDER"):
        execute(con, query(c, f'ORDER BY i."{c["name"]}" -- ORDER BY MIN(k.exp_day)'), Q)
    sql = (f'TYPE: list\n```sql\nSELECT n FROM (SELECT i."{c["name"]}" AS n, ROW_NUMBER() OVER (ORDER BY l."{c["exp"]}") AS r\n'
           f'  FROM "{c["lot"]}" l JOIN "{c["item"]}" i ON i."{c["code"]}" = l."{c["code"]}" WHERE l."{c["qty"]}" > 0) ORDER BY n\n```')
    with pytest.raises(ValueError, match="LIST ORDER"):
        execute(con, sql, Q)


@pytest.mark.parametrize("question", [
    "top 2 sản phẩm có hạn dùng gần nhất",  # ranking question: its own order
    "doanh thu tháng 6/2026 là bao nhiêu",  # not a lot/expiry question
    "những sản phẩm có hạn dùng trong tháng 9/2026, sắp xếp theo tên",  # the question names its own order
])
def test_other_questions_are_not_linted(shop, question):
    con, c = shop
    out = execute(con, query(c, f'ORDER BY i."{c["name"]}"'), question)
    assert out["type"] == "list" and sorted(out["answer"]) == sorted(GOLD)


def test_shapes_outside_the_rule():
    sql = 'SELECT name FROM lots WHERE "HSD" > DATE \'2026-01-01\' ORDER BY name'
    assert expiry_order_error(Q, sql, ["name"], [["a"]]) is None  # one item has no order
    assert expiry_order_error(Q, sql, ["n"], [[1], [2]]) is None  # numbers, not item names
    assert expiry_order_error(Q, 'SELECT name FROM lots ORDER BY name', ["name"], [["a"], ["b"]]) is None  # reads no expiry
    assert expiry_order_error(Q, 'SELECT name FROM debts WHERE "Hạn trả" > DATE \'2026-01-01\' ORDER BY name', ["name"], [["a"], ["b"]]) is None
    assert expiry_order_error(Q, sql, ["name"], [["a"], ["b"]]) is not None


def test_alias_chain_and_cast_count_as_expiry():
    sql = ('WITH a AS (SELECT item, CAST("HSD" AS DATE) AS d FROM lots), b AS (SELECT item, MIN(d) AS first_day FROM a GROUP BY item) '
           'SELECT item FROM b ORDER BY first_day, item LIMIT 50')
    assert expiry_order_error(Q, sql, ["item"], [["a"], ["b"]]) is None
    sql2 = 'WITH v AS (SELECT * FROM lots WHERE "HSD" >= DATE \'2026-05-01\') SELECT DISTINCT p."Món" AS product_name FROM v ORDER BY product_name'
    assert expiry_order_error(Q, sql2, ["product_name"], [["a"], ["b"]]) is not None  # a name aliased after an expiry CTE is still a name
