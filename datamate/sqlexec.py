"""Executing one sample: parse its TYPE line and SQL, run the query read-only with a timeout, turn the result into a typed
answer, and reject results the rulebook makes implausible (the error goes back to the model in a repair turn)."""
import math
import re
import threading
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP

from .hints import DEBT_WORDS, LIST_WORDS, LOT
from .ingest import ckey, norm

DEBT_CALL = re.compile(r"debt_(at|new|repaid)\s*\(")
# A call of a debt table macro where a table goes: FROM / JOIN / a FROM-list comma, then debt_x( ... ). Read on the query
# with its comments and string literals blanked: SEARCH traces had 17 executed debt samples whose only 'debt_at(' sat in a
# comment ('/* Implementation of debt_at(DATE ...) */') above a query rebuilt from the raw tables; all 17 were wrong, while
# the samples really calling a macro were right 63/82.
DEBT_TABLE_CALL = re.compile(r"(?i)(?:\bfrom|\bjoin|,)\s*debt_(?:at|new|repaid)\s*\(")
# A CTE named after a macro ('WITH debt_at AS (...)', 'debt_at(a, b) AS (...)'): the query defines its own debt_at, it does not call the layer's
DEBT_SHADOW = re.compile(r"(?i)\bdebt_(?:at|new|repaid)\s*(?:\([^()]*\))?\s+as\s+(?:not\s+)?(?:materialized\s+)?\(")
DEBT_LINT = "this shop has the debt layer:"  # prefix of the lint error below (repair_hint keys on it)


def code_only(sql):
    """The query with comments and string literals blanked (quoted identifiers kept): what the database reads as code."""
    out, i, n = [], 0, len(sql)
    while i < n:
        c = sql[i]
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
            out.append(" ")
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        elif c == "'":
            j = i + 1
            while j < n and not (sql[j] == "'" and not sql.startswith("''", j)):
                j += 2 if sql.startswith("''", j) else 1
            i = j + 1
            out.append("''")
        elif c == '"':
            j = sql.find('"', i + 1)
            j = n - 1 if j < 0 else j
            out.append(sql[i:j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def calls_debt_macro(sql):
    """True when the query itself reads a debt macro (debt_at / debt_new / debt_repaid) as a table. A macro name in a comment or a
    string, a CTE that redefines it, or a subquery aliased with its name ('FROM (SELECT ... FROM "thanh_toan") AS debt_repaid')
    is no call: such a query computes the debt from the raw tables."""
    code = code_only(sql or "")
    return bool(DEBT_TABLE_CALL.search(code)) and not DEBT_SHADOW.search(code)
TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable")
# A TYPE line that copies the prompt's format placeholder ('TYPE: <number>', 'TYPE <percent>', 'TYPE: [list]', 'TYPE: `date`'):
# the type word is there but not in the one form parse(), the scorer and the SFT validator read ('TYPE: number').
# A bare 'TYPE percent' (no colon, no marks) is not one: the TYPE fill completes it as before.
PLACEHOLDER_TYPE = re.compile(r"(?m)^[ \t*#>-]*TYPE\b[ \t]*(?:\*\*)?[ \t]*:?[ \t]*(?:\*\*)?[ \t]*[<\[(`'\"*]+[ \t]*(%s)[ \t]*[>\])`'\"*]+" % "|".join(TYPES))
PLACEHOLDER_ERROR = ("invalid TYPE line: the type word is wrapped in brackets or marks (a copy of the format placeholder). "
                     "Write it bare on its own line, e.g. TYPE: number, then the ```sql block")
READS_TABLE = re.compile(r"(?is)\bfrom\b")


def parse(text):
    if "<think>" in text:
        if "</think>" not in text:
            return None, None
        text = text.split("</think>")[-1]
    t = re.search(r"TYPE:\s*([a-z]+)", text)
    s = re.findall(r"```(?:sql)?\s*(.*?)```", text, re.S)
    return (t.group(1) if t else None), (s[-1].strip() if s else None)


def placeholder_type(text):
    """The type word of a placeholder-style TYPE line when parse() finds no TYPE (None otherwise). Only used to route the sample
    to a repair turn that asks for the bare form; the sample itself is never accepted in the placeholder form."""
    if "<think>" in text and "</think>" not in text:  # a cut-off thought has no answer part
        return None
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if re.search(r"TYPE:\s*[a-z]+", text):
        return None
    m = PLACEHOLDER_TYPE.search(text)
    return m.group(1) if m else None


def cell(v):
    if isinstance(v, (list, tuple)):
        return [cell(x) for x in v]
    if isinstance(v, dict):
        return {str(k): cell(x) for k, x in v.items()}
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite number")
        return int(v) if v.is_integer() else round(v, 4)
    if isinstance(v, (datetime, date)):
        return v.isoformat()[:10]
    return v


def run_sql(db, sql, timeout=20):
    timer = threading.Timer(timeout, db.interrupt)
    timer.start()
    try:
        cur = db.execute(sql)
        names = [c[0] for c in cur.description]
        rows = cur.fetchmany(2001)
    finally:
        timer.cancel()
    if len(rows) > 2000:
        raise ValueError("result too large")
    return names, [[cell(v) for v in r] for r in rows]


def to_answer(kind, names, rows):
    if kind == "table":
        return {"columns": names, "rows": rows}
    if kind == "list":
        if len(names) == 1:
            out = [r[0] for r in rows]
        elif len(rows) == 1:
            out = rows[0]
        else:
            raise ValueError(f"list needs one column, got {names}")
        if any(x is None for x in out):  # a filter matched nothing. The shape message made the repair reshape it (holdout: 50 -> 17 samples)
            raise ValueError("a list item is NULL (check filters/values)")
        if any(isinstance(x, (bool, list, dict)) for x in out):  # the format check rejects such a list
            raise ValueError("every list item must be one name or number (one row per item, one column)")
        return out
    if len(rows) != 1 or len(names) != 1:
        raise ValueError(f"expected one cell, got {len(rows)} rows x {names}")
    v = rows[0][0]
    if v is None:
        raise ValueError("result is NULL (check filters/values)")
    if kind in ("number", "percent"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("expected a number")
        # rulebook: money (incl. money averages) rounds to whole đồng, half up. Other averages to 2 decimals.
        # shortcut: |v| >= 1000 taken as VND money. A non-money average that large would be wrongly rounded.
        v = float(Decimal(str(v)).quantize(Decimal("1" if kind == "number" and abs(v) >= 1000 else ".01"), rounding=ROUND_HALF_UP))
        return int(v) if v.is_integer() else v
    if kind == "boolean":
        if not isinstance(v, bool):
            raise ValueError("expected boolean")
        return v
    v = str(v)
    if kind == "date" and not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v):
        raise ValueError("date must be YYYY-MM-DD or YYYY-MM")
    return v


CHANGE = re.compile(r"(?:^|_)(?:tang|giam(?!_(?:dau|gia|can|beo|mo|sot))|thay_doi|hon_hay_kem|cao_hay_thap|len_xuong|tang_truong|chenh_lech)(?:_|$)")  # giảm đau / giảm giá name an item or a discount, not a change


GROSS = re.compile(r"(?:^|_)(?:lai_gop|loi_nhuan_gop|lai_go)(?:_|$)")


LIMIT_ONE = re.compile(r"(?is)\blimit\s+1\s*;?\s*$")  # the query's last clause keeps one row
SELECT_ONE = re.compile(r"(?is)\s*select\s+1\s*;?\s*")  # the prompt's placeholder query for an unanswerable question


def shape_answer(p, sql):
    """Answer shape as the rulebook and scorer read it, from what the query itself says it computes.
    A one-object question ('... nào nhất', 'là ai') has a string answer; a solver that labels it TYPE list and keeps only the top row
    with LIMIT 1 says by its own query that it answers one object (SEARCH traces: 32 such one-item lists, all after LIMIT 1, every gold
    a string). A list question with several items never ends in LIMIT 1. And `SELECT 1` reads no table: it is the prompt's
    unanswerable placeholder under a wrong TYPE label, not the number 1 (12 SEARCH samples, all TYPE number)."""
    if not p or not sql:
        return p
    a = p.get("answer")
    if p["type"] == "list" and isinstance(a, list) and len(a) == 1 and isinstance(a[0], str) and LIMIT_ONE.search(sql):
        return {"type": "string", "answer": a[0]}
    if p["type"] != "unanswerable" and SELECT_ONE.fullmatch(sql):
        return {"type": "unanswerable"}
    return p


# Rulebook answer format: "liệt kê hàng theo hạn dùng thì hạn sớm nhất của mặt hàng trước, cùng hạn thì theo mã", and a list is scored on
# its order. The LIST ORDER hint (hints.lot_hint) says so in the prompt, yet SEARCH traces show both solvers returning the right items of an
# expiry list sorted by name or by item code (a 2-2 order split on both). The lint reads only the query's own structure: when it filters
# on an expiry column, its final ORDER BY must start with an expiry term, ascending.
EXPIRY = re.compile(r"(?:^|_)(?:hsd|han_dung|han_su_dung|het_han|expir[a-z]*|exp|best_before)(?:_|$)")  # not bare 'han' (hạn trả = a due date)
RANKED = re.compile(r"(?:^|_)(?:top|nhat|giam_dan|tang_dan|sap_xep|xep_theo|xep_hang|thu_tu)(?:_|$)")  # the question names its own order
SQL_TYPES = {"date", "timestamp", "time", "varchar", "text", "integer", "int", "bigint", "hugeint", "smallint", "tinyint", "double", "float",
             "real", "decimal", "numeric", "boolean", "interval", "json"}
IDENT = re.compile(r'"((?:[^"]|"")+)"|([^\W\d]\w*)')


def blank_sql(sql):
    """The SQL with comments removed and string literals emptied (same length), quoted identifiers kept."""
    out, i, n = [], 0, len(sql)
    while i < n:
        c = sql[i]
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i)); i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(" " * (j - i)); i = j
        elif c in "'\"":
            j = i + 1
            while j < n and not (sql[j] == c and not sql.startswith(c * 2, j)):
                j += 2 if sql.startswith(c * 2, j) else 1
            j = min(j + 1, n)
            out.append(sql[i:j] if c == '"' else "'" + " " * (j - i - 2) + "'" if j - i >= 2 else sql[i:j]); i = j
        else:
            out.append(c); i += 1
    return "".join(out)


def paren_depths(s):
    """Depth of every character; a parenthesis has the depth of the text around it."""
    d, out = 0, []
    for ch in s:
        if ch == ")":
            d -= 1
        out.append(d)
        if ch == "(":
            d += 1
    return out


def idents(text):
    return [(q.replace('""', '"') if q else w).lower() for q, w in IDENT.findall(text)]


def expiry_names(s, depth):
    """Identifiers of the (blanked) query that hold an expiry: columns named like one, and aliases of expressions over them."""
    names = {x for x in idents(s) if EXPIRY.search(ckey(x))}
    defs = []
    for m in re.finditer(r'(?i)\bas\s+("(?:[^"]|"")+"|[^\W\d]\w*)', s):
        alias = idents(m.group(1))[0]
        if alias in SQL_TYPES:  # CAST(x AS DATE)
            continue
        p, d = m.start(), depth[m.start()]
        k = p - 1
        while k >= 0 and depth[k] >= d and not (depth[k] == d and s[k] == ","):
            k -= 1
        seg = s[k + 1:p]
        cuts = [c.end() for c in re.finditer(r"(?i)\b(?:select|distinct|from|where|join|on|by|having|qualify|union|all)\b", seg) if depth[k + 1 + c.start()] == d]
        defs.append((alias, seg[cuts[-1]:] if cuts else seg))
    for _ in range(4):  # aliases of aliases
        new = {a for a, seg in defs if a not in names and names & set(idents(seg))}
        if not new:
            break
        names |= new
    return names


def final_order_key(s, depth):
    """First key of the query's last top-level ORDER BY (None when it has none)."""
    tops = [m for m in re.finditer(r"(?i)\border\s+by\b", s) if depth[m.start()] == 0]
    if not tops:
        return None
    start = tops[-1].end()
    end = next((m.start() for m in re.finditer(r"(?i)\b(?:limit|offset|fetch)\b", s[start:]) if depth[start + m.start()] == 0), len(s) - start) + start
    key = start
    while key < end and not (s[key] == "," and depth[key] == 0):
        key += 1
    return s[start:key].strip().rstrip(";").strip()


def expiry_order_error(question, sql, names, rows):
    """None, or the repair message for a list of items by expiry whose query does not order by expiry (see EXPIRY above)."""
    t = norm(question)
    if not (LOT.search(t) and LIST_WORDS.search(t)) or RANKED.search(t):
        return None
    if len(names) != 1 or len(rows) < 2 or not all(isinstance(r[0], str) for r in rows):
        return None
    s = blank_sql(sql)
    depth = paren_depths(s)
    exp = expiry_names(s, depth)
    if not exp:  # the query does not read an expiry: not a list by expiry as this query understands it
        return None
    key = final_order_key(s, depth)
    if key and exp & set(idents(key)) and not re.search(r"(?i)\bdesc\b", key):
        return None
    got = "has no final ORDER BY" if not key else f"orders by {key[:60]}" + (" descending" if key and re.search(r"(?i)\bdesc\b", key) else "")
    return ("LIST ORDER (rulebook): items listed by expiry go earliest expiry first (each item's earliest expiry among the lots kept), then by "
            f"item code; not by name or code alone. This query {got}. Keep its filters, one row per item (GROUP BY the item, not DISTINCT), "
            "and end with ORDER BY MIN(<expiry>), <item code>.")


def blank_sql_lines(sql):
    """The SQL with comment text and string-literal contents replaced by spaces, same length (positions map back to the SQL).
    Double-quoted identifiers are kept: they are the column names the structure checks read."""
    out, i, n = list(sql), 0, len(sql)
    while i < n:
        if sql[i] == "'":
            j = i + 1
            while j < n and not (sql[j] == "'" and not sql.startswith("''", j)):
                j += 2 if sql.startswith("''", j) else 1
            out[i + 1:min(j, n)] = " " * (min(j, n) - i - 1)
            i = j + 1
        elif sql[i] == '"':
            j = sql.find('"', i + 1)
            i = n if j < 0 else j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = [c if c == "\n" else " " for c in sql[i:j]]
            i = j
        else:
            i += 1
    return "".join(out)


def _close(s, i):
    """Index just after the parenthesis that closes the one at s[i] (len(s) when it never closes)."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"(": 1, ")": -1}.get(s[j], 0)
        if depth == 0:
            return j + 1
    return len(s)


SHIFT_CALL = re.compile(r"(?i)\b(?:lag|lead)\s*\(")
SHIFT_IDENT = re.compile(r'"[^"]+"|[A-Za-z_]\w*')
ITEM_END = {"from", "where", "group", "order", "having", "qualify", "window", "limit", "union", "select", "on", "join"}
NOT_NAME = {"end", "then", "else", "and", "or", "is", "not", "null", "asc", "desc", "nulls", "first", "last"}


def shift_aliases(blank):
    """Output names of the select items holding a LAG/LEAD value (NULL on the first/last row of each window), from the query's own
    text: `LAG(..) OVER (..) [expr] AS name`, or a bare name right after the OVER clause. The item ends at a comma at its own level
    or at a clause keyword, so COALESCE(LAG(x) OVER w, 0) (a null-filled shift) has no alias here."""
    names = set()
    for m in SHIFT_CALL.finditer(blank):
        k = _close(blank, m.end() - 1)
        o = re.match(r"(?is)\s*(?:(?:ignore|respect)\s+nulls\s*)?over\s*", blank[k:])
        if not o:
            continue
        k += o.end()
        k = _close(blank, k) if blank[k:k + 1] == "(" else k + len(re.match(r"\w*", blank[k:]).group(0))
        j, depth, first = k, 0, True
        while j < len(blank):
            c = blank[j]
            if c.isspace():
                j += 1
                continue
            w = SHIFT_IDENT.match(blank, j)
            if w:
                word = w.group(0).strip('"').lower()
                if word == "as" and not w.group(0).startswith('"'):
                    a = SHIFT_IDENT.match(blank, re.compile(r"\s*").match(blank, w.end()).end())
                    if a:
                        names.add(a.group(0).strip('"').lower())
                    break
                if word in ITEM_END and not w.group(0).startswith('"'):
                    break
                if first and depth == 0 and word not in NOT_NAME and re.match(r"\s*(?:,|\bfrom\b)", blank[w.end():], re.I):
                    names.add(word)  # LAG(x) OVER (...) name
                    break
                j, first = w.end(), False
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "," and depth <= 0:
                break
            j, first = j + 1, False
    return names


COUNT_CALL = re.compile(r"(?i)\bcount\s*\(")
PROBE = "dm_probe_count="


def count_tests(blank):
    """(start, end, count start, count end, k) of each `COUNT(...) = k` / `k = COUNT(...)` test with an integer literal k >= 2."""
    out = []
    for m in COUNT_CALL.finditer(blank):
        e = _close(blank, m.end() - 1)
        after = re.match(r"\s*=\s*(\d+)(?![\w.]|\s*[-+*/%])", blank[e:])
        before = re.search(r"(?<![\w.])(\d+)\s*=\s*$", blank[:m.start()])
        if after and int(after.group(1)) >= 2:
            out.append((m.start(), e + after.end(), m.start(), e, int(after.group(1))))
        elif before and int(before.group(1)) >= 2 and not re.search(r"[-+*/%<>!=]\s*$", blank[:before.start()]):
            out.append((before.start(), e, m.start(), e, int(before.group(1))))
    return out


def _probe(db, sql, edits):
    """The count the PROBE marker reports when the query runs with each (start, end, text) of edits replacing sql[start:end], else None."""
    for start, end, text in sorted(edits, reverse=True):
        sql = sql[:start] + text + sql[end:]
    try:
        run_sql(db, sql)
    except Exception as e:
        m = re.search(PROBE + r"(\d+)", str(e))
        return int(m.group(1)) if m else None
    return None


def shifted_count_error(db, sql):
    """Repair message for a `COUNT(...) = k` test that no group of rows passes only because the rows were filtered by
    `<LAG/LEAD value> IS NOT NULL`: that filter drops the first (or last) period of each window, so k periods leave k - 1 rows
    (k - 1 period-to-period pairs). Such a test is FALSE on any data, so the answer does not depend on the trend asked (SEARCH
    traces: 'tăng liên tục' samples with it answered FALSE on every question and shop, right only where the gold happened to be FALSE).
    Checked on the data by the query itself, with a CASE that raises the count when the test holds: probe 1 runs the query as written
    (the test can pass: no error); probe 2 runs it with those IS NOT NULL filters set to TRUE, and some group then reaches k. A query
    whose other filters (e.g. WHERE revenue > prev) keep the count under k is not flagged. None when the query has no such test."""
    blank = blank_sql_lines(sql)
    if not SHIFT_CALL.search(blank):
        return None
    aliases = shift_aliases(blank)
    nulls = [(m.start(), m.end(), "TRUE") for a in sorted(aliases)
             for m in re.finditer(rf'(?i)(?<![\w".])(?:{SHIFT_IDENT.pattern}\s*\.\s*)?"?{re.escape(a)}"?\s+is\s+not\s+null\b', blank)]
    if not nulls:
        return None
    name = re.sub(r"(?i)\s+is\s+not\s+null$", "", sql[nulls[0][0]:nulls[0][1]])
    for start, end, cs, ce, k in count_tests(blank):
        if any(s < end and start < e for s, e, _ in nulls):
            continue
        cnt = sql[cs:ce]
        test = (start, end, f"(CASE WHEN {cnt} = {k} THEN error('{PROBE}' || CAST({cnt} AS VARCHAR)) ELSE FALSE END)")
        if _probe(db, sql, [test]) is not None:
            continue  # some group has k rows: the test can pass
        if _probe(db, sql, [test, *nulls]) == k:
            return (f"COUNT = {k} can never hold here: the rows it counts keep only {name} IS NOT NULL, and LAG/LEAD gives the first "
                    f"(or last) period no neighbour, so {k} periods leave {k - 1} rows and the answer is fixed whatever the data. {k} periods "
                    f"have {k - 1} period-to-period pairs: test those {k - 1} pairs, or compare the period values directly")
    return None


def has_debt_layer(db):
    try:
        return bool(db.execute("SELECT count(*) FROM duckdb_functions() WHERE function_name = 'debt_at'").fetchone()[0])
    except Exception:
        return False


def execute(db, text, question="", trace=None):
    """trace (optional dict) receives the SQL as it is run against the database (after the rewrites below), for the run logs."""
    kind, sql = parse(text)
    if kind not in TYPES:
        raise ValueError(PLACEHOLDER_ERROR if kind is None and placeholder_type(text) else "missing or invalid TYPE line")
    if kind == "unanswerable":
        if sql and READS_TABLE.search(sql):
            # the prompt's unanswerable form is 'TYPE: unanswerable' + SELECT 1. A query reading the tables computes an answer: in SEARCH
            # traces every such sample came from a 4-token TYPE fill after a full query, and 33 of 35 were on answerable questions
            raise ValueError("TYPE unanswerable does not fit this query: it reads the shop's tables, so write the TYPE of the value it returns "
                             "(unanswerable is only TYPE: unanswerable with SELECT 1, when no table holds the data)")
        return {"type": kind}
    if not sql or not re.match(r"(?is)\s*(with|select|from)\b", sql) or re.search(r"(?i)\b(insert|update|delete|create|drop|alter|copy|attach|install|load|pragma)\b", sql):
        raise ValueError("need one read-only SELECT query in ```sql```")
    if SELECT_ONE.fullmatch(sql):  # the unanswerable placeholder under another TYPE label (see shape_answer), before the lints below
        return {"type": "unanswerable"}
    sql = re.sub(r'"(%[^"]*)"', r"'\1'", sql)  # a double-quoted strftime format is a string, never a column
    if kind == "percent" and CHANGE.search(norm(question)) and "-" not in re.sub(r"'[^']*'", "", sql):
        # 14B often writes 100*new/old after planning 100*(new-old)/old. Lint it into the repair turn
        raise ValueError("a change in % is ROUND(100.0 * (new - old) / old, 2), negative when it fell; this query has no subtraction")
    if DEBT_WORDS.search(norm(question)) and not calls_debt_macro(sql) and has_debt_layer(db):
        # train, shops with a debt layer: macro samples 740/816 right, raw-table samples 21/504, no question only a raw sample solved.
        # SEARCH: after the old wording ('answer ... with its macros') one solver's repairs reached a macro call in 3 of 63 samples
        # hitting this lint: it rebuilt the macro from the raw tables ('Implementation of debt_at'), 17 times passing with the name
        # in a comment. The macros exist: say so, and that the query must read them (prompts.repair_hint adds the DEBTS recipes)
        raise ValueError(DEBT_LINT + " debt_at, debt_new and debt_repaid already exist in this database as table macros that apply the rulebook's debt rules. "
                         "The query must read one of them in FROM, e.g. FROM debt_at(DATE 'X'): do not rebuild them from the raw tables; "
                         "a comment, a CTE or an alias with their name is no call")
    if trace is not None:
        trace["sql"] = sql.rstrip(";")
    names, rows = run_sql(db, sql.rstrip(";"))
    if err := shifted_count_error(db, sql.rstrip(";")):  # before the shape checks: an empty list from this test gets this message
        raise ValueError(err)
    ans = to_answer(kind, names, rows)
    shaped = shape_answer({"type": kind, "answer": ans}, sql)
    if shaped["type"] == "unanswerable":
        return shaped
    kind, ans = shaped["type"], shaped["answer"]
    if kind == "string" and (m := re.fullmatch(r"(?i)\s*th[aá]ng\s*(\d{1,2})\s*[/-]\s*(\d{4})\s*", ans)) and 1 <= int(m[1]) <= 12:
        ans = f"{m[2]}-{int(m[1]):02d}"  # 'tháng 3/2025' and '2025-03' split the vote on public 'tháng nào' questions. Gold is YYYY-MM
    if kind == "string" and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", ans):  # a day/month answer is type date
        kind = "date"
    if kind == "number" and isinstance(ans, (int, float)) and ans < 0 and GROSS.search(norm(question)):  # train: 0 of 7 negative gross-profit samples right
        raise ValueError(f"gross profit {ans} < 0 is implausible: the cost written on a sale line is for ONE SALE UNIT (the unit of the unit price), so multiply it "
                         "by the quantity in that sale unit, not by the base-unit quantity, and do not subtract purchases or expenses")
    if kind == "percent" and isinstance(ans, (int, float)) and abs(ans) > 1000:  # train: 0 of 10 such samples right, gold max 391%
        raise ValueError(f"{ans}% is implausible: the base of the ratio is (almost) empty. If the period it compares with lies outside the shop's data dates, "
                         "the question is unanswerable (TYPE: unanswerable); otherwise check its filters")
    if kind == "list" and (err := expiry_order_error(question, sql, names, rows)):
        raise ValueError(err)
    return {"type": kind, "answer": ans}
