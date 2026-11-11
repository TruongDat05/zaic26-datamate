"""Executing a sample on a tiny in-memory shop."""
import duckdb
import pytest

from datamate.sqlexec import execute, parse, to_answer


@pytest.fixture()
def db():
    con = duckdb.connect()
    con.execute("CREATE TABLE ban_hang (ngay DATE, mon VARCHAR, tien BIGINT)")
    con.execute("INSERT INTO ban_hang VALUES ('2026-03-01', 'Trà đào', 30000), ('2026-03-02', 'Bạc xỉu', 25000), ('2026-04-01', 'Trà đào', 30000)")
    return con


def test_parse_reads_type_and_last_sql_block():
    assert parse("notes\nTYPE: number\n```sql\nSELECT 1\n```") == ("number", "SELECT 1")
    assert parse("<think>still thinking") == (None, None)  # a cut-off thought has no query


def test_number_answer(db):
    out = execute(db, 'TYPE: number\n```sql\nSELECT SUM("tien") FROM "ban_hang" WHERE "ngay" < DATE \'2026-04-01\'\n```')
    assert out == {"type": "number", "answer": 55000}


def test_month_string_becomes_a_date(db):
    out = execute(db, "TYPE: string\n```sql\nSELECT 'tháng 3/2026'\n```")
    assert out == {"type": "date", "answer": "2026-03"}


def test_writes_are_refused(db):
    with pytest.raises(ValueError, match="read-only"):
        execute(db, "TYPE: number\n```sql\nDELETE FROM ban_hang\n```")


def test_percent_change_needs_a_subtraction(db):
    with pytest.raises(ValueError, match="change in %"):
        execute(db, "TYPE: percent\n```sql\nSELECT ROUND(100.0 * 3 / 2, 2)\n```", question="Doanh thu tăng bao nhiêu % so với tháng trước?")


def test_shapes():
    assert to_answer("list", ["mon"], [["Trà đào"], ["Bạc xỉu"]]) == ["Trà đào", "Bạc xỉu"]
    assert to_answer("table", ["mon", "tien"], [["Trà đào", 1]]) == {"columns": ["mon", "tien"], "rows": [["Trà đào", 1]]}
    assert to_answer("number", ["x"], [[1234.5]]) == 1235          # money rounds half up to whole đồng
    with pytest.raises(ValueError, match="one cell"):
        to_answer("number", ["a", "b"], [[1, 2]])


@pytest.mark.parametrize("line", ["TYPE: <number>", "TYPE <number>", "TYPE: [number]", "TYPE: `number`", "**TYPE:** <number>", "TYPE: **number**"])
def test_placeholder_type_line_is_sent_to_repair_not_accepted(db, line):
    # the bare form is the only one parse(), the scorer and the SFT validator read; a placeholder copy gets its own repair message
    with pytest.raises(ValueError, match="^invalid TYPE line"):
        execute(db, f'{line}\n```sql\nSELECT SUM("tien") FROM "ban_hang"\n```')
    assert execute(db, 'TYPE: number\n```sql\nSELECT SUM("tien") FROM "ban_hang"\n```') == {"type": "number", "answer": 85000}


def test_no_type_line_keeps_the_type_fill_route(db):
    with pytest.raises(ValueError, match="^missing or invalid TYPE line"):
        execute(db, 'Notes: sum the amounts\n```sql\nSELECT SUM("tien") FROM "ban_hang"\n```')
    with pytest.raises(ValueError, match="^missing or invalid TYPE line"):  # bare 'TYPE number': the TYPE fill completes it
        execute(db, 'TYPE number\n```sql\nSELECT SUM("tien") FROM "ban_hang"\n```')
    with pytest.raises(ValueError, match="^missing or invalid TYPE line"):  # a TYPE word outside the list is not a placeholder
        execute(db, 'TYPE: <numeric>\n```sql\nSELECT SUM("tien") FROM "ban_hang"\n```')


def test_unanswerable_label_on_a_query_that_reads_tables_is_rejected(db):
    with pytest.raises(ValueError, match="TYPE unanswerable does not fit"):
        execute(db, 'TYPE: unanswerable\n```sql\nSELECT SUM("tien") FROM "ban_hang"\n```')
    assert execute(db, "TYPE: unanswerable\n```sql\nSELECT 1\n```") == {"type": "unanswerable"}
    assert execute(db, "TYPE: unanswerable") == {"type": "unanswerable"}


@pytest.mark.parametrize("order", ["DESC", "ASC"])
def test_one_item_list_after_limit_1_is_the_one_object_string(db, order):
    sql = f'SELECT "mon" FROM "ban_hang" GROUP BY "mon" ORDER BY SUM("tien") {order}, "mon" LIMIT 1'
    want = "Trà đào" if order == "DESC" else "Bạc xỉu"
    assert execute(db, f"TYPE: list\n```sql\n{sql}\n```") == {"type": "string", "answer": want}
    assert execute(db, f"TYPE: string\n```sql\n{sql}\n```") == {"type": "string", "answer": want}


def test_lists_without_limit_1_stay_lists(db):
    # a filter that matches one item is still a list ('một phần tử vẫn là mảng'); so is a top-N and a list of numbers
    one = execute(db, 'TYPE: list\n```sql\nSELECT DISTINCT "mon" FROM "ban_hang" WHERE "ngay" >= DATE \'2026-04-01\'\n```')
    assert one == {"type": "list", "answer": ["Trà đào"]}
    top = execute(db, 'TYPE: list\n```sql\nSELECT "mon" FROM "ban_hang" GROUP BY "mon" ORDER BY SUM("tien") DESC LIMIT 2\n```')
    assert top == {"type": "list", "answer": ["Trà đào", "Bạc xỉu"]}
    assert execute(db, 'TYPE: list\n```sql\nSELECT SUM("tien") FROM "ban_hang" LIMIT 1\n```') == {"type": "list", "answer": [85000]}


def test_select_1_under_another_label_is_the_unanswerable_placeholder(db):
    assert execute(db, "TYPE: number\n```sql\nSELECT 1\n```") == {"type": "unanswerable"}
    assert execute(db, 'TYPE: number\n```sql\nSELECT 1 FROM "ban_hang" LIMIT 1\n```') == {"type": "number", "answer": 1}


# ---- a COUNT test that the LAG/LEAD null filter makes impossible ('doanh thu có tăng liên tục qua 3 tháng không') ----------------------

def _trend_db(table, day, amount, rows):
    con = duckdb.connect()
    con.execute(f'CREATE TABLE "{table}" ("{day}" DATE, "mặt hàng" VARCHAR, "{amount}" BIGINT)')
    con.executemany(f'INSERT INTO "{table}" VALUES (?, ?, ?)', rows)
    return con


UP = [("2025-05-03", "A", 100), ("2025-06-03", "A", 200), ("2025-07-03", "A", 300), ("2025-05-04", "B", 10), ("2025-06-04", "B", 20), ("2025-07-04", "B", 5)]
DOWN = [(d, i, 1000 - a) for d, i, a in UP]


def _degenerate(table="sales", day="day", amount="amount", shift="LAG(rev) OVER (ORDER BY m)", test="COUNT(*) = 3", lo="2025-05-01", hi="2025-08-01"):
    return (f'WITH mr AS (SELECT strftime("{day}", \'%Y-%m\') AS m, SUM("{amount}") AS rev FROM "{table}" '
            f"WHERE \"{day}\" >= DATE '{lo}' AND \"{day}\" < DATE '{hi}' GROUP BY 1), "
            f"c AS (SELECT m, rev, {shift} AS prev FROM mr) "
            f"SELECT CASE WHEN {test} AND MIN(CASE WHEN rev > prev THEN 1 ELSE 0 END) = 1 THEN TRUE ELSE FALSE END FROM c WHERE prev IS NOT NULL")


@pytest.mark.parametrize("rows", [UP, DOWN, list(reversed(UP))])  # the answer is FALSE whatever the trend and the row order
@pytest.mark.parametrize("names", [("sales", "day", "amount"), ("Sổ bán hàng", "Ngày lập", "Thành tiền")])
def test_count_over_lag_filtered_rows_that_can_never_pass_goes_to_repair(rows, names):
    con = _trend_db(*names, rows)
    with pytest.raises(ValueError, match=r"^COUNT = 3 can never hold here: .*prev IS NOT NULL.*2 period-to-period pairs"):
        execute(con, f"TYPE: boolean\n```sql\n{_degenerate(*names)}\n```", question="Doanh thu 3 tháng có tăng liên tục không?")


@pytest.mark.parametrize("shift,test", [
    ("LEAD(rev) OVER (ORDER BY m DESC)", "COUNT(*) = 3"),     # LEAD over the reversed order drops the first month too
    ("LAG(rev, 1) OVER (ORDER BY m)", "3 = COUNT(*)"),        # literal on the left
    ("LAG(rev) OVER (ORDER BY m)", "COUNT(DISTINCT m) = 3"),
])
def test_same_impossible_test_in_other_spellings(shift, test):
    con = _trend_db("sales", "day", "amount", UP)
    with pytest.raises(ValueError, match="can never hold"):
        execute(con, f"TYPE: boolean\n```sql\n{_degenerate(shift=shift, test=test)}\n```")


def test_trend_queries_that_can_pass_are_left_alone():
    con = _trend_db("sales", "day", "amount", UP)
    run = lambda sql: execute(con, f"TYPE: boolean\n```sql\n{sql}\n```")["answer"]
    assert run(_degenerate(test="COUNT(*) = 2")) is True                             # 3 months, 2 pairs
    with pytest.raises(ValueError, match="can never hold"):                          # the window opens a month earlier, but that month has no row
        run(_degenerate(test="COUNT(*) = 3", lo="2025-04-01"))
    con.execute("INSERT INTO sales VALUES ('2025-04-03', 'A', 50)")
    assert run(_degenerate(test="COUNT(*) = 3", lo="2025-04-01")) is True             # now it has 4 months, 3 pairs: the test can pass
    # the trend itself in WHERE: the count is the number of rises, below k when a month fell. Not the null filter's doing
    con.execute("INSERT INTO sales VALUES ('2025-06-05', 'B', -400)")
    rises = ("WITH mr AS (SELECT strftime(day, '%Y-%m') AS m, SUM(amount) AS rev FROM sales WHERE day >= DATE '2025-05-01' AND day < DATE '2025-08-01' "
             "GROUP BY 1), c AS (SELECT m, rev, LAG(rev) OVER (ORDER BY m) AS prev FROM mr) SELECT COUNT(*) = 2 FROM c WHERE rev > prev AND prev IS NOT NULL")
    assert run(rises) is False
    # a shift filled with COALESCE has no NULL row; a comment that names LAG / IS NOT NULL is not the query
    filled = _degenerate(shift="COALESCE(LAG(rev) OVER (ORDER BY m), 0)").replace("WHERE prev IS NOT NULL", "")
    assert run(filled) in (True, False)
    commented = "SELECT -- LAG(rev) OVER (ORDER BY m) AS prev ... WHERE prev IS NOT NULL\nCOUNT(*) = 3 FROM (SELECT DISTINCT strftime(day, '%Y-%m') FROM sales WHERE day < DATE '2025-06-01')"  # April, May
    assert run(commented) is False


def test_grouped_count_per_item_is_flagged_only_when_no_item_can_pass():
    con = _trend_db("sales", "day", "amount", UP)
    per_item = lambda k: (
        "WITH mr AS (SELECT \"mặt hàng\" AS item, strftime(day, '%Y-%m') AS m, SUM(amount) AS rev FROM sales GROUP BY 1, 2), "
        "c AS (SELECT item, rev, LAG(rev) OVER (PARTITION BY item ORDER BY m) AS prev FROM mr) "
        f"SELECT item FROM c WHERE prev IS NOT NULL GROUP BY item HAVING COUNT(*) = {k} AND MIN(CASE WHEN rev > prev THEN 1 ELSE 0 END) = 1 ORDER BY item")
    with pytest.raises(ValueError, match="COUNT = 3 can never hold"):
        execute(con, f"TYPE: list\n```sql\n{per_item(3)}\n```", question="Những mặt hàng nào tăng liên tục 3 tháng?")
    assert execute(con, f"TYPE: list\n```sql\n{per_item(2)}\n```") == {"type": "list", "answer": ["A"]}
