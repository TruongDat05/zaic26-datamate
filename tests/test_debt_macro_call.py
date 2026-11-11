"""A debt question on a shop with the debt layer must read a debt macro: the check reads the query as code (comments, strings,
CTEs and aliases named after a macro are no call), the vote pool uses the same check, and the repair turn gets the DEBTS recipes."""
import hashlib

import duckdb
import pytest

from datamate import debts
from datamate.prompts import repair_hint
from datamate.sqlexec import DEBT_LINT, calls_debt_macro, code_only, execute
from datamate.voting import debt_pool

ORDERS = 'SELECT "ma" AS order_id, "ngay" AS order_date, NULL AS customer_id, "khach" AS customer_name, "tong" AS total_due, false AS is_cancelled, false AS no_debt, "tra_bang" = \'no\' AS is_credit FROM "{o}"'
PAYS = 'SELECT "ma_don" AS order_id, "ngay_thu" AS pay_date, "tien" AS amount FROM "{p}"'


def shop(o="don", p="thu"):
    db = duckdb.connect()
    db.execute(f'CREATE TABLE "{o}" ("ma" VARCHAR, "ngay" DATE, "khach" VARCHAR, "tong" BIGINT, "tra_bang" VARCHAR)')
    db.execute(f"""INSERT INTO "{o}" VALUES ('D1', '2026-01-02', 'An', 100, 'no'), ('D2', '2026-01-03', 'Binh', 300, 'no'),
                   ('D3', '2026-01-04', 'An', 50, 'tien_mat'), ('D4', '2026-01-05', 'Chi', 80, 'no')""")
    db.execute(f'CREATE TABLE "{p}" ("ma_don" VARCHAR, "ngay_thu" DATE, "tien" BIGINT)')
    db.execute(f"""INSERT INTO "{p}" VALUES ('D1', '2026-01-10', 40), ('D2', '2026-01-03', 300), ('D3', '2026-01-04', 50), ('D4', '2026-01-20', 10)""")
    debts.build(db, (ORDERS.format(o=o), PAYS.format(p=p)))
    return db


RAW = '''SELECT customer_name, ROUND(SUM(debt)) AS so_tien_no FROM (
  {c}
  SELECT o."khach" AS customer_name, o."tong" - COALESCE(SUM(t."tien"), 0) AS debt FROM "don" o LEFT JOIN "thu" t ON t."ma_don" = o."ma" GROUP BY o."ma", o."khach", o."tong"
) GROUP BY 1 ORDER BY 2 DESC LIMIT 2'''

CALLS = [
    "SELECT ROUND(SUM(debt)) FROM debt_at(DATE '2026-01-31')",
    "select round(sum(debt)) from DEBT_AT (date '2026-01-31') d where d.age_days > 7",
    "SELECT ROUND(SUM(new_debt)) FROM debt_new(DATE '2026-01-01', DATE '2026-02-01')",
    "SELECT (SELECT ROUND(SUM(new_debt)) FROM debt_new(DATE '2026-01-01', DATE '2026-02-01') WHERE customer_name = 'An'),\n"
    "       (SELECT ROUND(SUM(amount)) FROM\n  debt_repaid(DATE '2026-01-01', DATE '2026-02-01') WHERE customer_name = 'An')",
    "SELECT d.customer_name FROM \"khach\" k JOIN debt_at(DATE '2026-01-31') d ON d.customer_name = k.\"ten\"",
    "SELECT COUNT(*) FROM \"khach\" k, debt_at(DATE '2026-01-31') d",
    "-- what is owed\nSELECT ROUND(SUM(debt)) FROM debt_at(DATE '2026-01-31') -- debt_at, today",
]
NO_CALLS = [
    RAW.format(c="/* Implementation of debt_at(DATE '2026-01-31') */"),
    RAW.format(c="-- debt_at(DATE '2026-01-31')"),
    "/* Rulebook: debt_at(DATE 'X') = one row per order */\n" + RAW.format(c=""),
    "SELECT ROUND(SUM(amount)) FROM (SELECT t.\"tien\" AS amount FROM \"thu\" t) AS debt_repaid",
    "WITH debt_at AS (SELECT 1 AS debt) SELECT SUM(debt) FROM debt_at",
    "WITH debt_at(order_id, debt) AS (SELECT 'D1', 5) SELECT SUM(debt) FROM debt_at(DATE '2026-01-31')",
    "WITH x AS MATERIALIZED (SELECT 1), debt_new AS NOT MATERIALIZED (SELECT 1 AS new_debt) SELECT * FROM debt_new(DATE '2026-01-01', DATE '2026-02-01')",
    "SELECT 'debt_at(DATE ''2026-01-31'')' AS note, SUM(\"tong\") FROM \"don\"",
    "SELECT SUM(\"Còn nợ\") FROM \"don_hang\"",
    "",
    None,
]


@pytest.mark.parametrize("sql", CALLS)
def test_real_calls(sql):
    assert calls_debt_macro(sql)


@pytest.mark.parametrize("sql", NO_CALLS)
def test_no_call(sql):
    assert not calls_debt_macro(sql)


def test_code_only_keeps_quoted_identifiers_and_doubled_quotes():
    sql = 'SELECT "Ghi chú -- /* x */" FROM debt_at(DATE \'2026-01-31\') WHERE customer_name = \'O\'\'Neil -- debt_at(\''
    code = code_only(sql)
    assert '"Ghi chú -- /* x */"' in code and "Neil" not in code and code.count("debt_at(") == 1
    assert calls_debt_macro(sql)
    assert not calls_debt_macro('SELECT "debt_at(" FROM "don"')  # a column name is no call either


def test_verdict_does_not_depend_on_table_names_or_layout():
    # metamorphic: other raw table/column names, other dates, other whitespace and case give the same verdict
    for sql in CALLS + [s for s in NO_CALLS if s]:
        renamed = sql.replace('"don"', '"sales_2"').replace('"thu"', '"receipts"').replace("2026-01-31", "2025-12-31")
        spaced = renamed.replace(" ", "  ").replace("FROM", "FROM\n  ")
        assert calls_debt_macro(renamed) == calls_debt_macro(sql) == calls_debt_macro(spaced)


def lint_error(db, sql, question="5 khách nào đang nợ nhiều nhất?"):
    with pytest.raises(ValueError) as e:
        execute(db, f"TYPE: table\n```sql\n{sql}\n```", question)
    return str(e.value)


@pytest.mark.parametrize("names", [("don", "thu"), ("hoa_don_2", "phieu_thu")])
def test_execute_rejects_a_raw_rebuild_named_in_a_comment(names):
    db = shop(*names)
    raw = RAW.format(c="/* Implementation of debt_at(DATE '2026-01-31') */").replace('"don"', f'"{names[0]}"').replace('"thu"', f'"{names[1]}"')
    err = lint_error(db, raw)
    assert err.startswith(DEBT_LINT) and len(err) <= 400 and " | " not in err  # the repair turn reads the last ' | ' part, cut at 400
    shadow = "WITH debt_at AS (SELECT 'x' AS customer_name, 1 AS debt) SELECT customer_name, SUM(debt) FROM debt_at GROUP BY 1"
    assert lint_error(db, shadow).startswith(DEBT_LINT)
    ok = execute(db, "TYPE: table\n```sql\nSELECT customer_name, ROUND(SUM(debt)) AS so_tien_no FROM debt_at(DATE '2026-01-31') "
                     "WHERE customer_key IS NOT NULL GROUP BY customer_key, customer_name ORDER BY 2 DESC, customer_key LIMIT 2\n```",
                 "5 khách nào đang nợ nhiều nhất?")
    assert ok == {"type": "table", "answer": {"columns": ["customer_name", "so_tien_no"], "rows": [["Chi", 70], ["An", 60]]}}


def test_lint_only_on_debt_questions_of_shops_with_the_layer():
    db = shop()
    raw = RAW.format(c="")
    assert execute(db, f"TYPE: table\n```sql\n{raw}\n```", "Top 2 khách mua nhiều nhất?")["type"] == "table"  # not a debt question
    plain = duckdb.connect()
    plain.execute('CREATE TABLE "don" ("ma" VARCHAR, "khach" VARCHAR, "tong" BIGINT)')
    plain.execute('CREATE TABLE "thu" ("ma_don" VARCHAR, "tien" BIGINT)')
    plain.execute("INSERT INTO \"don\" VALUES ('D1', 'An', 100)")
    assert execute(plain, f"TYPE: table\n```sql\n{raw}\n```", "5 khách nào đang nợ nhiều nhất?")["type"] == "table"  # no layer, no lint


def test_repair_hint_gives_the_recipes_only_for_the_debt_lint():
    err = lint_error(shop(), RAW.format(c="-- debt_at(DATE '2026-01-31')"))
    hint = repair_hint(err[:400])
    assert debts.RECIPES in hint and "FROM debt_at(DATE 'X')" in hint and hint.endswith("\n")
    assert debts.RECIPES not in repair_hint("Binder Error: Referenced column \"x\" not found in FROM clause!")
    assert repair_hint("result is NULL (check filters/values)").startswith("No row matched")


def test_debts_block_of_the_prompt_is_unchanged():
    # RECIPES was split out of describe(): the DEBTS block every sample prompt carries must stay byte-identical
    want = {(4, 4): "0b0dfdfe0d9616245b255b77c0e12a88b3b36c15e3ca085d7f46eee036b49767",
            (3, 8): "83c4ad800a4925b7ead92a135c88e7d481a614a276bcc6bd4eb103795522952c",
            (1, 4): "8c72f822401ae2210a04ef55bf8a210e81de8ad6b44829a8da084010a7fe38ac"}
    for (n, k), h in want.items():
        assert hashlib.sha256(debts.describe(n, k).encode()).hexdigest() == h
    assert debts.describe(4, 4).endswith(debts.RECIPES)


def test_debt_pool_ignores_a_macro_named_only_in_a_comment():
    macro = "TYPE: number\n```sql\nSELECT ROUND(SUM(debt)) FROM debt_at(DATE '2026-01-31')\n```"
    commented = "TYPE: table\n```sql\n" + RAW.format(c="/* debt_at(DATE '2026-01-31') */") + "\n```"
    raw = "TYPE: number\n```sql\nSELECT SUM(\"tong\") FROM \"don\"\n```"
    assert debt_pool([commented, macro, raw], 3, debt_q=True) == [1]
    assert debt_pool([commented, raw, raw], 3, debt_q=True) == [0, 1, 2]  # nobody calls a macro: every sample votes, as before
    assert debt_pool([raw, macro + "\n#REPAIR\n" + commented], 2, debt_q=True) == [0, 1]  # the last repair is what ran
