"""Kiểm một lời giải SQL (cục bộ, bằng module upstream): kiểm tĩnh, chạy 2 lần cùng kết quả, score == 1, đối chứng đột biến.

Đột biến: 4 bản mức shop (world.GENERIC) + 1 bản nhắm mọi đặc tả của shop (world.m_target), dựng bằng cùng bộ ghi file và cùng
SELECT ánh xạ view/nợ như shop gốc (view dựng không qua check, vì đột biến có thể cố ý chạm ngưỡng kiểm của views.check).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
from pathlib import Path

from sqlv1.up import debts, fewshot, hints, ingest, scoring, sqlexec, views
from sqlv1 import render as RD, shops as SH, world as WB

BAD_FUNCS = re.compile(r"(?i)\b(current_date|current_timestamp|now\s*\(|today\s*\(|get_current_timestamp|localtimestamp|random\s*\()")
DATE_LIT = re.compile(r"(?i)\bDATE\s*'(\d{4}-\d{2}-\d{2})'")
ANY_DATE = re.compile(r"'(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?'")
DATETIME_LIT = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
STR_LIT = re.compile(r"'((?:[^']|'')*)'")
QIDENT = re.compile(r'"((?:[^"]|"")+)"')


def allowed_dates(question: str, as_of: str, periods: list[str]) -> set[str]:
    """Ngày được phép ghi trong SQL: mọi ngày trong DATE ANCHORS, gợi ý kỳ của câu, và ngày nêu trong câu."""
    txt = hints.date_anchors(as_of) + "\n" + "\n".join(periods)
    out = set(re.findall(r"\d{4}-\d{2}-\d{2}", txt))
    for d, m, y in re.findall(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", question):
        try:
            x = dt.date(int(y), int(m), int(d))
            out |= {x.isoformat(), (x + dt.timedelta(days=1)).isoformat()}
        except ValueError:
            pass
    for m, y in re.findall(r"(?<![\d/])(\d{1,2})[/-](\d{4})", question):  # tháng M/Y
        try:
            a = dt.date(int(y), int(m), 1)
            b = dt.date(int(y) + (int(m) == 12), int(m) % 12 + 1, 1)
            out |= {a.isoformat(), b.isoformat(), (b - dt.timedelta(days=1)).isoformat()}
        except ValueError:
            pass
    base = set(out)  # suy từ mốc: ± 7/14 ngày (tuần trước nữa), ± N nêu trong câu (N ngày trước đó / N ngày tới)
    ns = {7, 14} | {int(x) for x in re.findall(r"(?<![\d/.-])(\d{1,3})(?![\d/.-])", question) if 0 < int(x) <= 400}
    for d in base:
        x = dt.date.fromisoformat(d)
        for n in ns:
            for k in (n, 2 * n, n - 1, 2 * n - 1):
                out |= {(x - dt.timedelta(days=k)).isoformat(), (x + dt.timedelta(days=k)).isoformat()}
    return out


def static_check(sql: str, notes: list[str], question: str, as_of: str, periods: list[str], visible: str, gold: dict) -> list[str]:
    """Lỗi tĩnh (rỗng = qua). visible = văn bản model thấy (ctx + gợi ý + VALUES MATCHING)."""
    errs = []
    body = re.sub(r"--[^\n]*", "", sql)
    if not re.match(r"(?is)\s*(with|select)\b", body) or ";" in body.strip().rstrip(";"):
        errs.append("not_single_select")
    if re.search(r"(?i)\b(insert|update|delete|create|drop|alter|copy|attach|install|load|pragma|export|import|set)\b", re.sub(STR_LIT, "''", body)):
        errs.append("not_read_only")
    if BAD_FUNCS.search(body):
        errs.append("current_date")
    ok_dates = allowed_dates(question, as_of, periods)
    for d in ANY_DATE.findall(body):
        if d not in ok_dates:
            errs.append(f"date_not_anchored:{d}")
    vis = ingest.norm(visible + " " + question)
    for s in STR_LIT.findall(body):
        s = s.replace("''", "'")
        if not s.strip() or DATETIME_LIT.fullmatch(s) or re.fullmatch(r"[%_\s]*|%[a-zA-Z%\-: ]+|\d{4}-\d{2}(-\d{2})?|[%_]*[a-z ]{1,24}[%_]*", s):
            continue  # mẫu LIKE, định dạng strftime, ngày, từ khoá trạng thái không dấu
        k = ingest.norm(s.strip("%_ "))
        if k and k not in vis:
            errs.append(f"literal_not_visible:{s[:30]}")
    if gold.get("type") in ("number", "percent") and isinstance(gold.get("answer"), (int, float)) and abs(gold["answer"]) >= 100:
        g = gold["answer"]
        lits = {float(x) for x in re.findall(r"(?<![\w.'])(-?\d+(?:\.\d+)?)(?![\w.'-])", re.sub(STR_LIT, "''", body))}
        if float(g) in lits and str(int(g)) not in question.replace(".", "").replace(",", ""):
            errs.append("hardcoded_answer")
    if gold.get("type") in ("string",) and isinstance(gold.get("answer"), str) and ingest.norm(gold["answer"]) not in ingest.norm(question):
        if any(ingest.norm(s) == ingest.norm(gold["answer"]) for s in STR_LIT.findall(body)):
            errs.append("hardcoded_answer")
    if len(notes) > 6:
        errs.append("notes_too_long")
    if any(("```" in n) or len(n) > 240 or re.search(r"(?i)\b(todo|fixme|i think|maybe|tôi nghĩ|chắc là)\b|\b(rule|mục)\s+[A-J]\b|§", n) for n in notes):
        errs.append("notes_garbage")
    return errs


def restatic(rec: dict, q: dict, visible_hint_free=True) -> list[str]:
    """Chấm lại phần kiểm tĩnh của một lời giải đã lưu (khi đổi luật kiểm tĩnh), không cần nạp shop: gợi ý kỳ tính từ câu hỏi."""
    periods = hints.period_hints(q["question"], q["as_of_date"], None)
    notes = [ln[2:] for ln in rec["completion"].split("TYPE:")[0].splitlines() if ln.startswith("- ")]
    errs = static_check(rec["sql"], notes, q["question"], q["as_of_date"], periods, "", q["gold"])
    return [e for e in errs if not e.startswith("literal_not_visible")] if visible_hint_free else errs


class ShopEnv:
    """Shop gốc + các bản đột biến đã nạp (DuckDB), cho một tiến trình kiểm."""

    def __init__(self, work: Path, bid: str, specs: list[dict]):
        self.work, self.bid = work, bid
        self.info, self.rec = SH.load_shop(work, bid)
        self.db, self.meta, self.layers = SH.open_db(work, bid, self.info)
        self.shown = self.meta
        self.muts = []  # (tên, bản ghi, db)
        for name, fn in WB.GENERIC:
            self.muts.append((name, fn(self.rec)))
        self.muts.append(("m_target", WB.m_target(self.rec, specs)))
        self.mdb = {}
        for name, mrec in self.muts:
            root = Path(os.environ.get("SQLV1_MUT_ROOT") or work / "mut_root") / name  # a separate root per concurrent job (same shop, same folders)
            folder = root / "data" / bid
            if folder.exists():
                shutil.rmtree(folder)
            (root / "data").mkdir(parents=True, exist_ok=True)
            if not (root / "calendar.csv").exists():
                shutil.copy(work / "shops_root" / "calendar.csv", root / "calendar.csv")
            plan = self.info["plan"]
            written = RD.write_shop(mrec, plan["family"], plan["style"], folder, SH.title_of(plan, mrec))
            try:
                self.mdb[name] = mutant_db(root, bid, written)
            except Exception as e:  # noqa: BLE001
                self.mdb[name] = e

    def close(self):
        self.db.close()
        for v in self.mdb.values():
            if not isinstance(v, Exception):
                v.close()

    def hints(self, q: dict) -> tuple[list[str], list[str]]:
        k = (q["question"], q["as_of_date"])
        if not hasattr(self, "_hint_cache"):
            self._hint_cache = {}
        if k not in self._hint_cache:
            self._hint_cache[k] = (ingest.entity_hits(self.db, self.shown, q["question"]), hints.question_hints(q["question"], q["as_of_date"], self.db, self.shown))
        return self._hint_cache[k]


def mutant_db(root: Path, bid: str, written: dict):
    db, meta = ingest.load_business(root, bid, canon=True)
    db.execute("SET enable_external_access = false")
    db.execute("SET memory_limit = '2GB'")
    Nt = RD.resolve_tables(meta, written["files"])
    N = RD.relations(Nt)
    cols = {k: sorted({c for t in Nt[k] for c in SH._cols(db, t)}) for k in Nt}
    views.build(db, RD.view_sql(written["roles"], N, cols))
    ds = RD.debt_sqls(written["roles"], N)
    if ds:
        try:
            debts.build(db, ds)
        except Exception:  # noqa: BLE001
            pass
    return db


def run(db, text: str, question: str):
    return sqlexec.execute(db, text, question)


def check_solution(env: ShopEnv, q: dict, text: str, min_mutants: int = 2) -> dict:
    """{'ok', 'stage', 'error', 'pred', 'mutants': {...}, 'route', 'skeleton'}."""
    kind, sql = sqlexec.parse(text)
    out = {"ok": False, "stage": None, "error": None, "type": kind}
    if not sql:
        out.update(stage="parse", error="no sql/type")
        return out
    try:
        p1 = run(env.db, text, q["question"])
    except Exception as e:  # noqa: BLE001
        out.update(stage="exec", error=str(e)[:300])
        return out
    try:
        p2 = run(env.db, text, q["question"])
    except Exception as e:  # noqa: BLE001
        out.update(stage="exec2", error=str(e)[:300])
        return out
    out["pred"] = p1
    if p1 != p2:
        out.update(stage="nondeterministic")
        return out
    if scoring.score(p1, q["gold"]) != 1:
        out.update(stage="wrong")
        return out
    res, n_chk, n_changed = {}, 0, 0
    for name, mrec in env.muts:
        mdb = env.mdb.get(name)
        if isinstance(mdb, Exception) or mdb is None:
            res[name] = "mutant_unavailable"
            continue
        try:
            mg = WB.gold(mrec, q["spec"])
        except (WB.Degenerate, ValueError, KeyError, StopIteration, ZeroDivisionError):
            res[name] = "not_checkable"
            continue
        try:
            mp = run(mdb, text, q["question"])
            ok = scoring.score(mp, mg) == 1
        except Exception as e:  # noqa: BLE001
            ok, mp = False, {"error": str(e)[:120]}
        n_chk += 1
        n_changed += mg != q["gold"]
        res[name] = "ok" if ok else "fail"
        if not ok:
            out.update(stage="mutant:" + name, mutants=res)
            return out
    out["mutants"] = res
    out["n_mutants_checked"], out["n_mutants_changed"] = n_chk, n_changed
    if n_chk < min_mutants:
        out.update(stage="too_few_mutants")
        return out
    out.update(ok=True, stage="ok")
    return out


def route(sql: str) -> str:
    return "macro" if re.search(r"debt_(at|new|repaid)\s*\(", sql) else "view" if re.search(r"(?i)\bv_(sales|orders)(_all)?\b", sql) else "raw"


def skeleton(sql: str) -> str:
    return fewshot.skeleton(sql)


# ------------------------------------------------------------------ kiểm ngữ nghĩa tĩnh theo slot (bổ sung sau audit bởi LLM độc lập vòng 1)
REVENUE_SLOTS = {"revenue", "revenue_channel", "revenue_payment", "avg_order", "days_over", "orders_over", "best_day", "best_month", "min_month",
                 "rev_list", "avg_daily_rev", "rev_change_pct", "rev_change", "shift_rev", "staff_rev", "staff_best", "top_customer", "top_customers",
                 "share_group", "share_channel", "shift_share", "rx_share"}
SHARE_SLOTS = {"share_group", "rx_share"}
ITEM_RANK = {"top_items", "top_item", "best_item_rev"}
CUST_RANK = {"top_customer", "top_customers", "top_customer_orders", "top_debtor"}
ITEM_ONLY = re.compile(r"(?i)line_kind\s*(?:=\s*'item'|in\s*\(\s*'item'\s*\))")
NAME_ASKS_AMOUNT = re.compile(r"(?:^|\s)(?:bao nhieu|bao nhiu|bnhiu|bn|bnh)(?:\s|$|\?)")
CODE_TOKEN = re.compile(r"\b[A-Z]{2,3}\d{3,5}\b")


def _order_by(sql: str) -> str:
    i = sql.upper().rfind("ORDER BY")
    return sql[i:] if i >= 0 else ""


def question_ok(q: dict) -> list[str]:
    """Lỗi mức câu hỏi: câu hỏi tên (list/string) mà lại đòi số ('là ai, dthu bn') — mơ hồ giữa list và table."""
    from sqlv1.questions import fold
    if q["gold"]["type"] in ("list", "string") and q["spec"]["slot"] in ITEM_RANK | CUST_RANK | {"staff_best"} and NAME_ASKS_AMOUNT.search(" " + fold(q["question"]) + " "):
        return ["name_question_asks_amount"]
    return []


def semantic_check(sql: str, notes: list[str], q: dict, roles: dict, by_code: bool) -> list[str]:
    slot, errs = q["spec"]["slot"], []
    body = re.sub(r"--[^\n]*", "", sql)
    if slot in REVENUE_SLOTS and "'service'" not in body:
        for m in ITEM_ONLY.finditer(body):
            if slot in SHARE_SLOTS and re.search(r"(?i)filter\s*\(\s*where[^)]*$", body[max(0, m.start() - 120):m.start()]):
                continue
            errs.append("service_lines_dropped")
            break
    if slot in ("rx_rev", "rx_share") and roles.get("c_rx") and f'"{roles["c_rx"]}"' not in body and roles["c_rx"] not in body:
        errs.append("rx_not_from_catalog_flag")
    if slot in ITEM_RANK:
        ob = _order_by(body)
        codes = {"product_code", roles.get("l_item") or "", roles.get("c_code") or ""} - {""}
        if not any(c in ob for c in codes):
            errs.append("item_tie_not_by_code")
    if slot in CUST_RANK and by_code:
        ob = _order_by(body)
        if "customer_key" not in ob and "customer_id" not in ob and (roles.get("o_cust_code") or "#") not in ob:
            errs.append("customer_tie_not_by_code")
    if slot == "unsold_items":
        import synth_world as W
        s, _ = W.resolve(q["spec"]["period"], q["spec"]["as_of"])
        if not re.search(rf"<\s*(?:DATE\s*)?'{s.isoformat()}'", body):
            errs.append("unsold_without_prior_sales")
    if slot in ("price_on_day", "price_change_pct") and re.search(r"(?i)\bv_(?:sales|orders)\b(?!_all)", body):
        errs.append("price_from_valid_orders_only")  # giá bán ngày D tính mọi đơn đặt ngày D, kể cả hủy/trả (R9)
    for n in notes:
        for c in CODE_TOKEN.findall(n):
            if c not in body:
                errs.append("notes_code_not_in_sql")
    for m in re.finditer(r"(?is)\bCASE\b(.*?)\bEND\b", body):
        vals = re.findall(r"(?i)\b(?:THEN|ELSE)\s+([^\s,)]+)", m.group(1))
        if len(vals) >= 2 and len(set(vals)) == 1:
            errs.append("constant_case")
            break
    return sorted(set(errs))
