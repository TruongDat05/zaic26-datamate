"""SQL mẫu tất định theo slot (rules_spec btc_v1), viết trên view chuẩn của upstream (v_sales, v_orders, *_all) và macro nợ.

Dùng cho: (1) kiểm chứng quy ước trên dữ liệu BTC train (btc_align.py); (2) lưới an toàn khi không model nào giải đúng
(`metadata.solution_source = "template"`). Mỗi hàm trả (type, sql, notes); notes ≤ 6 dòng ngắn, đúng khuôn completion.
Kỳ [s, e] (kể cả hai đầu) viết thành cửa sổ nửa mở `>= DATE 's' AND < DATE 'e+1'` như prompts.SYSTEM.
Phần cần bảng thô (nhập hàng, lô, giá trên dòng, giá vốn, tồn) nhận `raw` = mô tả bảng/cột của shop (render.raw_info).
"""
from __future__ import annotations

import datetime as dt


def q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def lit(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _d(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x)[:10])


def win(col: str, s, e) -> str:
    s, e = _d(s), _d(e)
    return f"{col} >= DATE '{s}' AND {col} < DATE '{e + dt.timedelta(days=1)}'"


def _fmt(s, e) -> str:
    s, e = _d(s), _d(e)
    return f"[{s}, {e + dt.timedelta(days=1)})"


class Unsupported(ValueError):
    """Slot không có SQL mẫu cho shop này (thiếu bảng/cột)."""


# ------------------------------------------------------------------ đo trên view
def revenue(s, e, item=None, group=None, channel=None, payment=None, shift=None, staff=None):
    w = [win("order_date", s, e)]
    notes = [f"revenue = SUM(amount) of v_sales (valid orders only), order_date in {_fmt(s, e)}"]
    if item is not None:
        w.append(f"product_name = {lit(item)}")
        notes.append(f"item filter: product_name = '{item}'")
    if group is not None:
        w.append(f"product_group = {lit(group)}")
        notes.append(f"group filter: product_group = '{group}'")
    if channel is not None:
        w.append(f"channel = {lit(channel)}")
        notes.append(f"channel recorded on the order = '{channel}'")
    if payment is not None:
        w.append(f"payment = {lit(payment)}")
        notes.append(f"payment method recorded on the order = '{payment}'")
    if shift is not None:
        lo, hi = {"sang": (6, 13), "chieu": (14, 21)}[shift]
        w.append(f"hour(order_ts) BETWEEN {lo} AND {hi}")
        notes.append(f"shift by order time: {lo:02d}:00-{hi:02d}:59")
    if staff is not None:
        w.append(f"staff = {lit(staff)}")
        notes.append(f"staff = '{staff}'")
    return "number", f"SELECT ROUND(COALESCE(SUM(amount), 0)) FROM v_sales WHERE {' AND '.join(w)}", notes


def n_orders(s, e, item=None, group=None, channel=None, customer_key=None):
    notes = [f"count valid orders (v_orders / v_sales), order_date in {_fmt(s, e)}", "COUNT(DISTINCT order_id), not lines"]
    if item is None and group is None:
        w = [win("order_date", s, e)]
        if channel is not None:
            w.append(f"channel = {lit(channel)}")
        return "number", f"SELECT COUNT(*) FROM v_orders WHERE {' AND '.join(w)}", notes
    w = [win("order_date", s, e)]
    if item is not None:
        w.append(f"product_name = {lit(item)}")
    if group is not None:
        w.append(f"product_group = {lit(group)}")
    return "number", f"SELECT COUNT(DISTINCT order_id) FROM v_sales WHERE {' AND '.join(w)}", notes


def n_cancelled(s, e):
    return "number", f"SELECT COUNT(*) FROM v_orders_all WHERE is_cancelled AND {win('order_date', s, e)}", [
        f"cancelled orders (is_cancelled) placed in {_fmt(s, e)}", "returned orders are not cancelled"]


def cancel_rate(s, e):
    return "percent", (f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE is_cancelled) / COUNT(*), 2) FROM v_orders_all "
                       f"WHERE {win('order_date', s, e)}"), [
        f"denominator: every order placed in {_fmt(s, e)}, any status (v_orders_all)", "percent = ROUND(100.0 * cancelled / all, 2)"]


def avg_order(s, e):
    return "number", f"SELECT ROUND(SUM(revenue) / COUNT(*)) FROM v_orders WHERE {win('order_date', s, e)}", [
        f"average order value = revenue / number of valid orders, {_fmt(s, e)}", "money: round to whole VND"]


def qty(s, e, item, unit_base=None):
    return "number", f"SELECT COALESCE(SUM(qty), 0) FROM v_sales WHERE product_name = {lit(item)} AND {win('order_date', s, e)}", [
        f"qty in v_sales is already in base units{f' ({unit_base})' if unit_base else ''}: SUM(qty), no extra conversion",
        f"valid orders only, order_date in {_fmt(s, e)}"]


def top_items(s, e, k, group=None, as_string=False, code=True):
    w = [win("order_date", s, e), "line_kind = 'item'"] + ([f"product_group = {lit(group)}"] if group else [])
    tie = "product_code" if code else "product_name"
    sql = (f"SELECT product_name FROM v_sales WHERE {' AND '.join(w)} GROUP BY product_code, product_name "
           f"ORDER BY SUM(qty) DESC, {tie} LIMIT {k}")
    return ("string" if as_string else "list"), sql, [
        "best sellers by base-unit quantity SUM(qty), not by lines or money; service lines excluded",
        f"valid orders, {_fmt(s, e)}; ties by item code"]


def best_item_rev(s, e, code=True):
    tie = "product_code" if code else "product_name"
    return "string", (f"SELECT product_name FROM v_sales WHERE {win('order_date', s, e)} AND line_kind = 'item' "
                      f"GROUP BY product_code, product_name ORDER BY SUM(amount) DESC, {tie} LIMIT 1"), [
        f"item with the highest revenue in {_fmt(s, e)}", "ties by item code"]


def best_day(s, e):
    return "date", (f"SELECT strftime(order_date, '%Y-%m-%d') FROM v_sales WHERE {win('order_date', s, e)} "
                    "GROUP BY order_date ORDER BY SUM(amount) DESC, order_date LIMIT 1"), [
        f"daily revenue of valid orders in {_fmt(s, e)}", "answer: the top day, earliest on ties"]


def best_month(s, e, lowest=False):
    o = "ASC" if lowest else "DESC"
    return "date", (f"SELECT strftime(order_date, '%Y-%m') FROM v_sales WHERE {win('order_date', s, e)} "
                    f"GROUP BY 1 ORDER BY SUM(amount) {o}, 1 LIMIT 1"), [
        f"monthly revenue of valid orders in {_fmt(s, e)}", f"answer YYYY-MM of the {'lowest' if lowest else 'highest'} month, earliest on ties"]


def month_rev_low(s, e):
    """Tháng doanh thu thấp nhất: tính cả tháng không bán (0)."""
    s, e = _d(s), _d(e)
    return "date", (f"SELECT strftime(m.d, '%Y-%m') FROM range(DATE '{s.replace(day=1)}', DATE '{e + dt.timedelta(days=1)}', INTERVAL 1 MONTH) m(d) "
                    f"LEFT JOIN v_sales v ON date_trunc('month', v.order_date) = m.d AND {win('v.order_date', s, e)} "
                    "GROUP BY m.d ORDER BY COALESCE(SUM(v.amount), 0), m.d LIMIT 1"), [
        f"revenue of each month in {_fmt(s, e)} (a month without sales counts 0)", "answer YYYY-MM of the lowest month, earliest on ties"]


def top_customers(s, e, k, as_string=False, by="money"):
    measure = {"money": "SUM(amount)", "orders": "COUNT(DISTINCT order_id)", "days": "COUNT(DISTINCT order_date)"}[by]
    what = {"money": "revenue", "orders": "number of valid orders", "days": "number of days with a valid order"}[by]
    sql = (f"SELECT any_value(customer_name) FROM v_sales WHERE customer_key IS NOT NULL AND {win('order_date', s, e)} "
           f"GROUP BY customer_key ORDER BY {measure} DESC, customer_key LIMIT {k}")
    return ("string" if as_string else "list"), sql, [
        f"customers ranked by {what} of their valid orders; walk-ins (customer_key NULL) excluded",
        f"{_fmt(s, e)}; ties by customer key"]


def n_customers(s, e):
    return "number", f"SELECT COUNT(DISTINCT customer_key) FROM v_orders WHERE {win('order_date', s, e)}", [
        f"customers with at least one valid order in {_fmt(s, e)}", "COUNT(DISTINCT customer_key) ignores walk-ins (NULL)"]


def cust_ge_n(s, e, n):
    return "number", (f"SELECT COUNT(*) FROM (SELECT customer_key FROM v_orders WHERE customer_key IS NOT NULL AND {win('order_date', s, e)} "
                      f"GROUP BY customer_key HAVING COUNT(*) >= {n})"), [
        f"customers with >= {n} valid orders in {_fmt(s, e)}", "walk-ins are not customers"]


def exists_item(s, e, item):
    return "boolean", f"SELECT COUNT(*) > 0 FROM v_sales WHERE product_name = {lit(item)} AND {win('order_date', s, e)}", [
        f"any valid order line of '{item}' in {_fmt(s, e)}"]


def group_table(s, e):
    return "table", (f'SELECT product_group AS "Nhóm hàng", ROUND(SUM(amount)) AS "Doanh thu" FROM v_sales '
                     f"WHERE product_group IS NOT NULL AND {win('order_date', s, e)} GROUP BY product_group ORDER BY 2 DESC, 1"), [
        f"revenue per product group, valid orders, {_fmt(s, e)}", "table: group name first, then revenue"]


def unsold_items(s, e):
    return "number", (f"SELECT COUNT(*) FROM (SELECT DISTINCT product_code FROM v_sales WHERE line_kind = 'item' AND order_date < DATE '{_d(s)}' "
                      f"EXCEPT SELECT DISTINCT product_code FROM v_sales WHERE line_kind = 'item' AND {win('order_date', s, e)})"), [
        "unsold (ế) = sold in a valid order before the period, none in the period", f"period {_fmt(s, e)}"]


def rev_change_pct(s, e, bs, be, group=None):
    g = f" AND product_group = {lit(group)}" if group else ""
    return "percent", (f"WITH n AS (SELECT SUM(amount) v FROM v_sales WHERE {win('order_date', s, e)}{g}), "
                       f"o AS (SELECT SUM(amount) v FROM v_sales WHERE {win('order_date', bs, be)}{g}) "
                       "SELECT ROUND(100.0 * (n.v - o.v) / o.v, 2) FROM n, o"), [
        f"new period {_fmt(s, e)}, base period {_fmt(bs, be)}" + (f", group '{group}'" if group else ""),
        "change % = ROUND(100.0 * (new - old) / old, 2), negative when it fell"]


def rev_change(s, e, bs, be):
    return "number", (f"WITH n AS (SELECT COALESCE(SUM(amount), 0) v FROM v_sales WHERE {win('order_date', s, e)}), "
                      f"o AS (SELECT COALESCE(SUM(amount), 0) v FROM v_sales WHERE {win('order_date', bs, be)}) SELECT ROUND(n.v - o.v) FROM n, o"), [
        f"difference = revenue {_fmt(s, e)} - revenue {_fmt(bs, be)} (negative if lower)"]


def share_group(s, e, group):
    return "percent", (f"SELECT ROUND(100.0 * SUM(amount) FILTER (WHERE product_group = {lit(group)}) / SUM(amount), 2) FROM v_sales "
                       f"WHERE {win('order_date', s, e)}"), [
        f"share = group revenue / whole-shop revenue (service lines included), {_fmt(s, e)}"]


def share_where(s, e, cond: str, note: str):
    return "percent", (f"SELECT ROUND(100.0 * SUM(amount) FILTER (WHERE {cond}) / SUM(amount), 2) FROM v_sales WHERE {win('order_date', s, e)}"), [
        note, f"denominator: all revenue in {_fmt(s, e)}"]


def days_over(s, e, threshold, strict=False):
    op = ">" if strict else ">="
    return "number", (f"SELECT COUNT(*) FROM (SELECT order_date FROM v_sales WHERE {win('order_date', s, e)} "
                      f"GROUP BY order_date HAVING SUM(amount) {op} {int(threshold)})"), [
        f"count DAYS whose revenue {op} {int(threshold)} in {_fmt(s, e)}, not orders"]


def orders_over(s, e, threshold, strict=False):
    op = ">" if strict else ">="
    return "number", f"SELECT COUNT(*) FROM v_orders WHERE {win('order_date', s, e)} AND revenue {op} {int(threshold)}", [
        f"valid orders whose revenue {op} {int(threshold)}, {_fmt(s, e)}"]


def items_ge(s, e, n, group=None):
    g = f" AND product_group = {lit(group)}" if group else ""
    return "number", (f"SELECT COUNT(*) FROM (SELECT product_code FROM v_sales WHERE line_kind = 'item' AND {win('order_date', s, e)}{g} "
                      f"GROUP BY product_code HAVING SUM(qty) >= {n})"), [
        f"items whose base-unit qty >= {n} in {_fmt(s, e)}" + (f", group '{group}'" if group else "")]


def rev_list(s, e):
    s, e = _d(s), _d(e)
    return "list", (f"SELECT COALESCE(ROUND(SUM(v.amount)), 0) FROM range(DATE '{s.replace(day=1)}', DATE '{e + dt.timedelta(days=1)}', INTERVAL 1 MONTH) m(d) "
                    f"LEFT JOIN v_sales v ON date_trunc('month', v.order_date) = m.d AND {win('v.order_date', s, e)} GROUP BY m.d ORDER BY m.d"), [
        f"one revenue per month of {_fmt(s, e)}, in month order (0 for a month without sales)"]


def avg_daily_rev(s, e):
    n = (_d(e) - _d(s)).days + 1
    return "number", f"SELECT ROUND(COALESCE(SUM(amount), 0) / {n}) FROM v_sales WHERE {win('order_date', s, e)}", [
        f"revenue / {n} days of {_fmt(s, e)} (days without sales count)"]


def staff_best(s, e):
    return "string", (f"SELECT staff FROM v_sales WHERE staff IS NOT NULL AND {win('order_date', s, e)} GROUP BY staff "
                      "ORDER BY SUM(amount) DESC, staff LIMIT 1"), [f"staff ranked by revenue of valid orders, {_fmt(s, e)}"]


# ------------------------------------------------------------------ công nợ (macro)
def debt_total(x):
    return "number", f"SELECT ROUND(COALESCE(SUM(debt), 0)) FROM debt_at(DATE '{_d(x)}')", [
        f"debt owed at the end of {_d(x)}: debt_at macro (credit orders, payments up to that day)"]


def debt_n_cust(x):
    return "number", f"SELECT COUNT(DISTINCT customer_key) FROM debt_at(DATE '{_d(x)}')", [
        f"customers still owing at the end of {_d(x)} (walk-ins have no key)"]


def top_debtor(x, k=1):
    sql = (f"SELECT customer_name FROM debt_at(DATE '{_d(x)}') WHERE customer_key IS NOT NULL GROUP BY customer_key, customer_name "
           f"ORDER BY SUM(debt) DESC, customer_key LIMIT {k}")
    return ("string" if k == 1 else "list"), sql, [f"customers by debt owed at the end of {_d(x)}, ties by customer key"]


def debt_cust(x, name):
    return "number", f"SELECT ROUND(COALESCE(SUM(debt), 0)) FROM debt_at(DATE '{_d(x)}') WHERE customer_name = {lit(name)}", [
        f"debt of '{name}' at the end of {_d(x)}"]


def debt_collected(s, e):
    return "number", f"SELECT ROUND(COALESCE(SUM(amount), 0)) FROM debt_repaid(DATE '{_d(s)}', DATE '{_d(e) + dt.timedelta(days=1)}')", [
        f"repayments after the purchase day, paid in {_fmt(s, e)} (debt_repaid)"]


def debt_new(s, e):
    return "number", f"SELECT ROUND(COALESCE(SUM(new_debt), 0)) FROM debt_new(DATE '{_d(s)}', DATE '{_d(e) + dt.timedelta(days=1)}')", [
        f"new credit debt of orders placed in {_fmt(s, e)} (debt_new)"]


def debt_older(x, days):
    return "number", f"SELECT ROUND(COALESCE(SUM(debt), 0)) FROM debt_at(DATE '{_d(x)}') WHERE age_days > {int(days)}", [
        f"debt at the end of {_d(x)} on orders older than {int(days)} days (age from order date)"]


def completion(kind: str, sql: str, notes: list[str]) -> str:
    """Chuỗi trả lời đúng khuôn sqlexec.parse đọc (Notes ≤ 6 dòng)."""
    lines = [n.strip() for n in notes if n and n.strip()][:6]
    return "Notes:\n" + "\n".join("- " + x for x in lines) + f"\nTYPE: {kind}\n```sql\n{sql}\n```"


# ------------------------------------------------------------------ bảng thô (nhập hàng, lô, giá trên dòng, giá vốn, tồn)
def _status_ok(col):
    return f"NOT regexp_matches(replace(strip_accents(lower(coalesce({col}, ''))), 'đ', 'd'), '(^|[^a-z])huy([^a-z]|$)')"


def _fac_join(raw, alias, item_expr, unit_expr):
    """LEFT JOIN hệ số quy đổi (bảng quy đổi một bậc hoặc theo bậc) → cột {alias}.ff."""
    if not raw.get("units"):
        return f"LEFT JOIN (SELECT CAST(NULL AS VARCHAR) fi, CAST(NULL AS VARCHAR) fu, CAST(NULL AS DOUBLE) ff WHERE false) {alias} ON {alias}.fi = {item_expr} AND {alias}.fu = {unit_expr}"
    U = raw["units"]
    if raw.get("stepwise"):
        sub = (f"(SELECT a.fi, a.fu, a.ff * coalesce(b.ff * coalesce(c.ff, 1), 1) AS ff FROM (SELECT trim({q(raw['u_item'])}) fi, trim({q(raw['u_unit'])}) fu, "
               f"TRY_CAST({q(raw['u_factor'])} AS DOUBLE) ff, trim({q(raw['u_via'])}) fv FROM {U}) a LEFT JOIN (SELECT trim({q(raw['u_item'])}) fi, trim({q(raw['u_unit'])}) fu, "
               f"TRY_CAST({q(raw['u_factor'])} AS DOUBLE) ff, trim({q(raw['u_via'])}) fv FROM {U}) b ON b.fi = a.fi AND b.fu = a.fv LEFT JOIN (SELECT trim({q(raw['u_item'])}) fi, "
               f"trim({q(raw['u_unit'])}) fu, TRY_CAST({q(raw['u_factor'])} AS DOUBLE) ff, trim({q(raw['u_via'])}) fv FROM {U}) c ON c.fi = b.fi AND c.fu = b.fv)")
    else:
        sub = f"(SELECT trim(CAST({q(raw['u_item'])} AS VARCHAR)) fi, trim({q(raw['u_unit'])}) fu, any_value(TRY_CAST({q(raw['u_factor'])} AS DOUBLE)) ff FROM {U} GROUP BY 1, 2)"
    return f"LEFT JOIN {sub} {alias} ON {alias}.fi = {item_expr} AND {alias}.fu = {unit_expr}"


def _item_in(raw, expr, name):
    """expr (an item-code expression of a raw table) is the item the question NAMES: its code is looked up by that name, written as
    the prompt shows it (VALUES MATCHING lists the name; nothing in the prompt pairs a name with its code, so a code literal would
    be a guess), in the catalog when the shop has one, else in the canonical sales lines (v_sales_all, every order)."""
    if raw.get("catalog") and raw.get("c_code") and raw.get("c_name"):
        src = f"SELECT trim(CAST({q(raw['c_code'])} AS VARCHAR)) FROM {raw['catalog']} WHERE lower(trim({q(raw['c_name'])})) = lower({lit(name)})"
    else:
        src = f"SELECT DISTINCT trim(CAST(product_code AS VARCHAR)) FROM v_sales_all WHERE lower(trim(product_name)) = lower({lit(name)})"
    return f"{expr} IN ({src})"


def purchase_money(raw, s, e):
    P = raw["purchases"]
    return "number", (f"SELECT ROUND(COALESCE(SUM(TRY_CAST({q(raw['pu_amount'])} AS DOUBLE)), 0)) FROM {P} "
                      f"WHERE {win('CAST(' + q(raw['pu_date']) + ' AS DATE)', s, e)} AND {_status_ok(q(raw['pu_status']))}"), [
        f"purchase receipts dated in {_fmt(s, e)}, cancelled receipts excluded", "sum of the receipt line amounts"]


def avg_cost(raw, s, e, name, unit_base):
    P = raw["purchases"]
    fj = _fac_join(raw, "f", f"trim(CAST(p.{q(raw['pu_item'])} AS VARCHAR))", f"trim(p.{q(raw['pu_unit'])})")
    return "number", (f"SELECT ROUND(SUM(TRY_CAST(p.{q(raw['pu_amount'])} AS DOUBLE)) / SUM(TRY_CAST(p.{q(raw['pu_qty'])} AS DOUBLE) * coalesce(f.ff, 1))) "
                      f"FROM {P} p {fj} WHERE {_item_in(raw, f"trim(CAST(p.{q(raw['pu_item'])} AS VARCHAR))", name)} AND {win('CAST(p.' + q(raw['pu_date']) + ' AS DATE)', s, e)} "
                      f"AND {_status_ok('p.' + q(raw['pu_status']))}"), [
        f"weighted average purchase cost per {unit_base}: total amount / total quantity in base units", f"receipts in {_fmt(s, e)}, not cancelled"]


def lots_expiring(raw, today, n):
    t = _d(today)
    L = raw["lots"]
    return "number", (f"SELECT COUNT(*) FROM {L} WHERE TRY_CAST({q(raw['lo_qty'])} AS DOUBLE) > 0 AND CAST({q(raw['lo_exp'])} AS DATE) >= DATE '{t}' "
                      f"AND CAST({q(raw['lo_exp'])} AS DATE) <= DATE '{t + dt.timedelta(days=int(n))}'"), [
        f"lots still in stock (qty > 0) expiring from today {t} to {t + dt.timedelta(days=int(n))} inclusive"]


def _line_rel(raw):
    """(FROM ... của dòng bán có ngày đặt, biểu thức mã đơn, mã hàng, đơn vị, SL, đơn giá, giá vốn) theo họ."""
    k = raw["kind"]
    L = raw["lines"]
    if k == "header_lines":
        O = raw["orders"]
        frm = (f"{L} l JOIN {O} o ON trim(CAST(o.{q(raw['o_id'])} AS VARCHAR)) = trim(CAST(l.{q(raw['l_id'])} AS VARCHAR))"
               + (f" AND l.{q(raw['l_event'])} = 'sale'" if raw.get("l_event") else ""))
        dte = f"CAST(o.{q(raw['o_ts'])} AS DATE)"
    else:
        frm = f"{L} l" + (f" WHERE l.{q(raw['l_type'])} = 'Bán'" if k == "neg" else "")
        if k == "neg":
            frm = f"(SELECT * FROM {L} WHERE {q(raw['l_type'])} = 'Bán') l"
        dte = f"CAST(l.{q(raw['l_ts'])} AS DATE)"
    oid = f"trim(CAST(l.{q(raw['l_id'])} AS VARCHAR))"
    return frm, dte, oid


def price_on_day(raw, day, name, unit_base):
    frm, dte, _ = _line_rel(raw)
    unit = f"trim(l.{q(raw['l_unit'])})"
    fj = _fac_join(raw, "f", f"trim(CAST(l.{q(raw['l_item'])} AS VARCHAR))", unit)
    return "number", (f"SELECT DISTINCT TRY_CAST(l.{q(raw['l_price'])} AS DOUBLE) / coalesce(f.ff, 1) FROM {frm} {fj} "
                      f"WHERE {_item_in(raw, f"trim(CAST(l.{q(raw['l_item'])} AS VARCHAR))", name)} AND {dte} = DATE '{_d(day)}'"), [
        f"unit price on the sale lines of orders placed on {_d(day)}, per base unit ({unit_base}), before line discount",
        "all orders of that day count, cancelled or returned included"]


def price_change_pct(raw, d1, d2, name, unit_base):
    _, s1, _ = price_on_day(raw, d1, name, unit_base)
    _, s2, _ = price_on_day(raw, d2, name, unit_base)
    return "percent", f"WITH a AS ({s1}), b AS ({s2}) SELECT ROUND(100.0 * (b.p - a.p) / a.p, 2) FROM (SELECT * FROM a) a(p), (SELECT * FROM b) b(p)", [
        f"unit price per {unit_base} on {_d(d1)} and on {_d(d2)} (sale lines of that day, before line discount)", "change % = ROUND(100.0 * (new - old) / old, 2)"]


def profit_gross(raw, s, e):
    frm, dte, oid = _line_rel(raw)
    cost = (f"WITH r AS (SELECT COALESCE(SUM(amount), 0) v FROM v_sales WHERE {win('order_date', s, e)}), "
            f"c AS (SELECT COALESCE(SUM(TRY_CAST(l.{q(raw['l_qty'])} AS DOUBLE) * TRY_CAST(l.{q(raw['l_cost'])} AS DOUBLE)), 0) v FROM {frm} "
            f"JOIN v_orders vo ON vo.order_id = {oid} WHERE {win('vo.order_date', s, e)}) SELECT ROUND(r.v - c.v) FROM r, c")
    return "number", cost, ["gross profit = revenue of valid orders - SUM(qty in the sale unit x cost written on the sale line)", f"period {_fmt(s, e)}"]


def stock_qty(raw, x, name, unit_base):
    x = _d(x)
    P, S = raw["purchases"], raw["stock"]
    fj = _fac_join(raw, "f", f"trim(CAST(p.{q(raw['pu_item'])} AS VARCHAR))", f"trim(p.{q(raw['pu_unit'])})")
    return "number", (f"SELECT (SELECT COALESCE(SUM(TRY_CAST({q(raw['st_qty'])} AS DOUBLE)), 0) FROM {S} WHERE {_item_in(raw, f"trim(CAST({q(raw['st_item'])} AS VARCHAR))", name)}) "
                      f"+ (SELECT COALESCE(SUM(TRY_CAST(p.{q(raw['pu_qty'])} AS DOUBLE) * coalesce(f.ff, 1)), 0) FROM {P} p {fj} WHERE {_item_in(raw, f"trim(CAST(p.{q(raw['pu_item'])} AS VARCHAR))", name)} "
                      f"AND CAST(p.{q(raw['pu_date'])} AS DATE) <= DATE '{x}' AND {_status_ok('p.' + q(raw['pu_status']))}) "
                      f"- (SELECT COALESCE(SUM(qty), 0) FROM v_sales WHERE {_item_in(raw, 'trim(CAST(product_code AS VARCHAR))', name)} AND order_date <= DATE '{x}')"), [
        f"stock of the item at the end of {x} in {unit_base}: opening stock + purchased base qty (not cancelled) - base qty sold in valid orders"]


def rx_rev(raw, s, e, share=False):
    C = raw["catalog"]
    rx = f"trim(CAST(product_code AS VARCHAR)) IN (SELECT trim(CAST({q(raw['c_code'])} AS VARCHAR)) FROM {C} WHERE lower(trim({q(raw['c_rx'])})) = 'có')"
    if share:
        return share_where(s, e, rx, "prescription share = revenue of items flagged prescription in the catalog / all revenue")
    return "number", f"SELECT ROUND(COALESCE(SUM(amount), 0)) FROM v_sales WHERE {rx} AND {win('order_date', s, e)}", [
        f"revenue of prescription items (catalog flag), valid orders, {_fmt(s, e)}"]


# ------------------------------------------------------------------ điều phối theo đặc tả
def for_spec(spec: dict, rec: dict, raw: dict):
    """(type, sql, notes) cho đặc tả, hoặc raise Unsupported. rec để đổi mã ↔ tên (tham số đặc tả ghi mã hàng)."""
    import synth_world as W
    slot, p = spec["slot"], spec.get("params") or {}
    if slot == "unanswerable":
        return "unanswerable", "SELECT 1", ["the shop records no data for what is asked (rulebook: unanswerable)"]
    s, e = W.resolve(spec["period"], spec["as_of"]) if spec.get("period") else (None, None)
    T = _d(spec["as_of"])
    items = {it["item"]: it for it in rec["items"]}
    nm = lambda c: items[c]["name"]  # noqa: E731
    base = lambda c: items[c]["base"]  # noqa: E731
    if slot == "revenue":
        return revenue(s, e)
    if slot == "revenue_item":
        return revenue(s, e, item=nm(p["item"]))
    if slot == "revenue_group":
        return revenue(s, e, group=p["group"])
    if slot == "revenue_channel":
        return revenue(s, e, channel=p["channel"])
    if slot == "revenue_payment":
        return revenue(s, e, payment=p["payment"])
    if slot == "n_orders":
        return n_orders(s, e, channel=p.get("channel"))
    if slot == "n_orders_item":
        return n_orders(s, e, item=nm(p["item"]))
    if slot == "n_cancelled":
        return n_cancelled(s, e)
    if slot == "n_returned":
        return "number", f"SELECT COUNT(*) FROM v_orders_all WHERE is_returned AND NOT is_cancelled AND {win('order_date', s, e)}", [
            f"returned orders (any returned line, even partial) placed in {_fmt(s, e)}"]
    if slot == "cancel_rate":
        return cancel_rate(s, e)
    if slot == "avg_order":
        return avg_order(s, e)
    if slot == "qty_base":
        return qty(s, e, nm(p["item"]), base(p["item"]))
    if slot in ("top_items", "top_item"):
        return top_items(s, e, p.get("k", 1), as_string=slot == "top_item")
    if slot == "best_item_rev":
        return best_item_rev(s, e)
    if slot == "best_day":
        return best_day(s, e)
    if slot == "best_month":
        return best_month(s, e)
    if slot == "min_month":
        return month_rev_low(s, min(e, T))
    if slot in ("top_customer", "top_customers"):
        return top_customers(s, e, p.get("k", 1), as_string=slot == "top_customer")
    if slot == "top_customer_orders":
        return top_customers(s, e, 1, as_string=True, by="orders")
    if slot == "n_customers":
        return n_customers(s, e)
    if slot == "cust_ge_n":
        return cust_ge_n(s, e, p["n"])
    if slot == "exists_item_day":
        return exists_item(s, e, nm(p["item"]))
    if slot == "group_table":
        return group_table(s, e)
    if slot == "unsold_items":
        return unsold_items(s, e)
    if slot in ("rev_change_pct", "rev_change", "g_rev_change"):
        bs, be = W.resolve(spec["base_period"], spec["as_of"])
        if slot == "rev_change":
            return rev_change(s, e, bs, be)
        return rev_change_pct(s, e, bs, be, group=p.get("group") if slot == "g_rev_change" else None)
    if slot == "share_group":
        return share_group(s, e, p["group"])
    if slot == "share_channel":
        return share_where(s, e, f"channel = {lit(p['channel'])}", f"share of channel '{p['channel']}' in revenue")
    if slot == "shift_share":
        lo, hi = {"sang": (6, 13), "chieu": (14, 21)}[p["shift"]]
        return share_where(s, e, f"hour(order_ts) BETWEEN {lo} AND {hi}", f"shift by order time {lo:02d}:00-{hi:02d}:59")
    if slot == "days_over":
        return days_over(s, e, p["threshold"])
    if slot == "orders_over":
        return orders_over(s, e, p["threshold"])
    if slot == "items_ge":
        return items_ge(s, e, p["n"], p.get("group"))
    if slot == "rev_list":
        return rev_list(s, e)
    if slot == "avg_daily_rev":
        return avg_daily_rev(s, min(e, T))
    if slot == "shift_rev":
        return revenue(s, e, shift=p["shift"])
    if slot == "shift_orders":
        lo, hi = {"sang": (6, 13), "chieu": (14, 21)}[p["shift"]]
        return "number", f"SELECT COUNT(*) FROM v_orders WHERE {win('order_date', s, e)} AND hour(order_ts) BETWEEN {lo} AND {hi}", [
            f"valid orders created {lo:02d}:00-{hi:02d}:59, {_fmt(s, e)}"]
    if slot == "staff_rev":
        return revenue(s, e, staff=p["staff"])
    if slot == "staff_best":
        return staff_best(s, e)
    if slot == "debt_total":
        return debt_total(min(e, T))
    if slot == "debt_n_cust":
        return debt_n_cust(min(e, T))
    if slot == "top_debtor":
        return top_debtor(min(e, T))
    if slot == "debt_cust":
        return debt_cust(min(e, T), p["customer"])
    if slot == "debt_older":
        return debt_older(min(e, T), p["days"])
    if slot == "debt_collected":
        return debt_collected(s, min(e, T))
    if slot == "debt_new":
        return debt_new(s, min(e, T))
    need = {"purchase_money": "purchases", "avg_cost": "purchases", "stock_qty": "stock", "lots_expiring": "lots", "price_on_day": "lines",
            "price_change_pct": "lines", "profit_gross": "lines", "rx_rev": "catalog", "rx_share": "catalog"}
    if slot in need and not raw.get(need[slot]):
        raise Unsupported(slot)
    if slot == "purchase_money":
        return purchase_money(raw, s, min(e, T))
    if slot == "avg_cost":
        return avg_cost(raw, s, min(e, T), nm(p["item"]), base(p["item"]))
    if slot == "lots_expiring":
        return lots_expiring(raw, T, p["n"])
    if slot in ("price_on_day", "price_change_pct"):
        if not raw.get("l_price"):
            raise Unsupported(slot)
        if slot == "price_on_day":
            return price_on_day(raw, s, nm(p["item"]), base(p["item"]))
        return price_change_pct(raw, s, e, nm(p["item"]), base(p["item"]))
    if slot == "profit_gross":
        if not raw.get("l_cost"):
            raise Unsupported(slot)
        return profit_gross(raw, s, e)
    if slot == "stock_qty":
        return stock_qty(raw, min(e, T), nm(p["item"]), base(p["item"]))
    if slot in ("rx_rev", "rx_share"):
        if not raw.get("c_rx"):
            raise Unsupported(slot)
        return rx_rev(raw, s, e, share=slot == "rx_share")
    raise Unsupported(slot)
