"""Bước 1.3 (phần 2, venv upstream): kiểm chứng quy ước trên dữ liệu BTC train, CHỈ CỤC BỘ, chỉ in số gộp.

1. Nạp từng shop train bằng ingest.load_business(canon=True) của upstream.
2. Dựng SELECT ánh xạ tất định từ vai trò cột (data/synth_source/shops/index.jsonl), bảng tra trạng thái / khách lẻ / hệ số
   tính bằng Python rồi nhúng thành VALUES; views.check để ghi nhận, views.build để dựng v_sales/v_orders.
3. Mỗi câu thuộc dạng phổ biến (tham số từ btc_capture.py) → SQL mẫu (sqlgen) → sqlexec.execute → scoring.score với answers.jsonl.

    python eval/sqlv1/btc_align.py --params /dev/shm/sqlv1/btc_params.json --out outputs/synth_sql_v1/btc_align.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.up import SYNTH_ROOT, ingest, scoring, sqlexec, views  # noqa: E402
from sqlv1 import sqlgen as G  # noqa: E402

TRAIN = SYNTH_ROOT / "data" / "train"
INDEX = SYNTH_ROOT / "data" / "synth_source" / "shops" / "index.jsonl"


# ------------------------------------------------------------------ chuẩn hoá (bản riêng, cùng ý với datamate.clean cũ)
def key(s) -> str:
    if s is None:
        return ""
    t = unicodedata.normalize("NFKD", str(s).replace("đ", "d").replace("Đ", "D"))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


WALKIN = {"kl", "khach le", "le", "khach vang lai", "vang lai", "khach", "khachle", "kh le", "guest", "walk in"}
STATUS = {
    "cancel": ["huy", "da huy", "cancel", "cancelled", "canceled", "bi huy", "khach huy", "void", "huy don", "shop huy", "khong giao duoc",
               "giao that bai", "giao khong thanh cong", "that bai", "failed", "khach bom hang", "bom hang"],
    "returned": ["tra hang", "da tra hang", "tra", "hoan tra", "return", "returned", "refund", "khach tra", "tra lai", "hang tra ve", "tra ve",
                 "khach tra hang", "hoan hang", "doi tra", "returns"],
    "transit": ["dang giao", "dang van chuyen", "shipping", "delivering", "cho giao", "dang ship", "in transit", "dang giao hang", "dang chuyen",
                "dang xu ly", "cho xu ly", "processing", "pending", "cho xac nhan", "cho lay hang", "dang dong goi", "dang giao dich", "dang gui"],
    "done": ["hoan thanh", "da hoan thanh", "xong", "ok", "done", "completed", "complete", "thanh cong", "da giao", "da thanh toan", "paid",
             "success", "finished", "da xong", "da giao hang", "giao thanh cong", "hoan tat", "da nhan", "delivered", "da lay", "khach da lay"],
}


def status_class(v) -> str:
    k = key(v)
    if not k:
        return ""
    for c, ws in STATUS.items():
        if k in ws:
            return c
    for c in ("cancel", "returned", "transit", "done"):
        if any(w in k for w in STATUS[c] if len(w) > 3):
            return c
    return ""


def is_walkin(v) -> bool:
    k = key(v)
    return k == "" or k in WALKIN or k.startswith("khach le")


ADJUST = re.compile(r"^(voucher|phi ship|phi giao|phi van chuyen|tien ship|ship|giam gia|chiet khau|diem|doi diem|tru diem|dung \d+ diem)\b")
q, lit = G.q, G.lit


def values_cte(name: str, cols: list[str], rows: list[tuple]) -> str:
    if not rows:
        rows = [tuple([None] * len(cols))]
    body = ", ".join("(" + ", ".join("NULL" if v is None else (lit(v) if isinstance(v, str) else repr(float(v))) for v in r) + ")" for r in rows)
    return f"{name}({', '.join(cols)}) AS (SELECT * FROM (VALUES {body}) t({', '.join(cols)}))"


# ------------------------------------------------------------------ dựng SELECT ánh xạ từ vai trò cột
class Mapper:
    def __init__(self, db, meta, roles: dict):
        self.db, self.meta, self.roles = db, meta, roles
        self.cols = {t["name"]: {c for c, *_ in t["columns"]} for t in meta["tables"]}
        self.types = {t["name"]: {c: ty for c, ty, *_ in t["columns"]} for t in meta["tables"]}
        self.rows = {t["name"]: t["rows"] for t in meta["tables"]}
        self.up = {}
        for old, rr in roles.items():
            need = set(rr.values())
            cand = [n for n, cs in self.cols.items() if need <= cs]
            nk = ingest.norm(old)
            best = [n for n in cand if nk in n] or cand
            if best:
                self.up[old] = max(best, key=lambda n: (n.endswith("_all"), self.rows[n]))

    def k(self, table, col, alias=None):
        """Biểu thức khoá nối (VARCHAR, số nguyên không đuôi .0)."""
        ref = (alias + "." if alias else "") + q(col)
        ty = self.types[table].get(col, "VARCHAR")
        if ty in ("DOUBLE", "BIGINT", "INTEGER", "FLOAT", "DECIMAL", "HUGEINT"):
            return f"CAST(CAST({ref} AS BIGINT) AS VARCHAR)"
        return f"trim(CAST({ref} AS VARCHAR))"

    def distinct(self, table, col):
        return [r[0] for r in self.db.execute(f"SELECT DISTINCT {q(col)} FROM {q(table)}").fetchall()]

    def build(self) -> tuple[str, dict]:
        R, info = self.roles, {}
        cand = [t for t, r in R.items() if "amount" in r and ("item" in r or "item_code" in r) and t in self.up
                and not re.search(r"nhap|tra_hang|ton|kho", ingest.norm(t))]
        if not cand:
            raise ValueError("no sales line table")
        sales_old = max(cand, key=lambda t: self.rows[self.up[t]])
        r, S = R[sales_old], self.up[sales_old]
        info["sales"] = S
        ctes = [f"s AS (SELECT *, row_number() OVER () AS _rn FROM {q(S)})"]
        field = {}  # vai trò -> biểu thức SQL trên bảng s / bảng nối
        joins = []
        has_order = "order" in r
        field["order"] = self.k(S, r["order"], "s") if has_order else "CAST(s._rn AS VARCHAR)"
        for role in ("item", "item_code", "date", "status", "qty_base", "qty", "cust", "cust_code", "group", "note", "time", "channel", "pay", "staff", "unit"):
            if role in r:
                field[role] = "s." + q(r[role])
        # đầu đơn ở bảng khác
        for old, rr in R.items():
            if old == sales_old or "order" not in rr or old not in self.up or re.search(r"nhap|thu|quy|tra_hang|giao", ingest.norm(old)):
                continue
            need = [x for x in ("date", "status", "cust", "cust_code", "time", "channel", "pay", "staff") if x in rr and x not in field]
            if not need or not has_order:
                continue
            H = self.up[old]
            a = f"h{len(joins)}"
            sel = ", ".join(f"any_value({q(rr[x])}) AS {q('h_' + x)}" for x in need)
            ctes.append(f"{a} AS (SELECT {self.k(H, rr['order'])} AS _k, {sel} FROM {q(H)} GROUP BY 1)")
            joins.append(f"LEFT JOIN {a} ON {a}._k = {field['order']}")
            for x in need:
                field[x] = f"{a}.{q('h_' + x)}"
        if "date" not in field:
            raise ValueError("no date")
        # danh mục: tên, nhóm theo mã
        if "item_code" in field:
            for old, rr in R.items():
                if old == sales_old or "item_code" not in rr or old not in self.up or re.search(r"nhap|ton", ingest.norm(old)):
                    continue
                need = [x for x in ("item", "group") if x in rr and x not in field]
                if need:
                    C = self.up[old]
                    a = f"c{len(joins)}"
                    sel = ", ".join(f"any_value({q(rr[x])}) AS {q('c_' + x)}" for x in need)
                    ctes.append(f"{a} AS (SELECT {self.k(C, rr['item_code'])} AS _k, {sel} FROM {q(C)} GROUP BY 1)")
                    joins.append(f"LEFT JOIN {a} ON {a}._k = {self.k(S, r['item_code'], 's')}")
                    for x in need:
                        field[x] = f"{a}.{q('c_' + x)}"
        if "group" not in field and "item" in field:
            for old, rr in R.items():
                if old != sales_old and {"item", "group"} <= set(rr) and old in self.up:
                    C = self.up[old]
                    g = {}
                    for a_, b_ in self.db.execute(f"SELECT {q(rr['item'])}, {q(rr['group'])} FROM {q(C)}").fetchall():
                        g.setdefault(key(a_), b_)
                    items = self.distinct(S, r["item"]) if "item" in r else []
                    ctes.append(values_cte("gmap", ["gi", "gg"], [(v, g.get(key(v))) for v in items if v is not None]))
                    joins.append(f"LEFT JOIN gmap ON gmap.gi = {field['item']}")
                    field["group"] = "gmap.gg"
                    break
        if "item" not in field:
            raise ValueError("no item name")
        # khách: tên từ danh mục khách khi đơn chỉ có mã
        if "cust" not in field and "cust_code" in field:
            for old, rr in R.items():
                if old in self.up and "cust_code" in rr and re.search(r"khach", ingest.norm(old)):
                    C = self.up[old]
                    nm = rr.get("cust") or next((c for c in self.cols[C] if ingest.norm(c) in ("ten_goi", "tengoi", "ten_kh", "ten_khach_hang", "ten_khach", "khach_hang_ten", "kh")), None)
                    if nm:
                        ctes.append(f"kh AS (SELECT {self.k(C, rr['cust_code'])} AS _k, any_value({q(nm)}) AS kname FROM {q(C)} GROUP BY 1)")
                        joins.append(f"LEFT JOIN kh ON kh._k = CAST({field['cust_code']} AS VARCHAR)")
                        field["cust"] = "kh.kname"
                        break
        # trạng thái → lớp (bảng tra)
        st_rows, src = [], None
        if "status" in field:
            src = field["status"]
        elif "note" in field:
            src = field["note"]
        if src is not None:
            vals = [x[0] for x in self.db.execute("WITH " + ",\n".join(ctes) + f"\nSELECT DISTINCT CAST({src} AS VARCHAR) FROM s " + " ".join(joins)).fetchall()]
            st_rows = [(v, status_class(v) or "done") for v in vals if v is not None]
            info["status_values"] = collections.Counter(c for _, c in st_rows)
            ctes.append(values_cte("stmap", ["sv", "sc"], st_rows))
            joins.append(f"LEFT JOIN stmap ON stmap.sv = CAST({src} AS VARCHAR)")
            cls = "coalesce(stmap.sc, 'done')"
        else:
            cls = "'done'"
        # trả hàng: số lượng âm, bảng có số lượng trả
        ret_extra = []
        qcol = r.get("qty") or r.get("qty_base")
        if qcol:
            ret_extra.append(f"s.{q(qcol)} < 0")
        for old, rr in R.items():
            if "returned_qty" in rr and "order" in rr and old in self.up:
                T = self.up[old]
                ctes.append(f"rq AS (SELECT DISTINCT {self.k(T, rr['order'])} AS _k FROM {q(T)} WHERE TRY_CAST({q(rr['returned_qty'])} AS DOUBLE) > 0)")
                ret_extra.append(f"{field['order']} IN (SELECT _k FROM rq)")
                break
        # số lượng gốc
        if "qty_base" in r:
            qty = f"TRY_CAST(s.{q(r['qty_base'])} AS DOUBLE)"
        elif "qty" in r:
            ucols = [c for c in (r["qty"] + " (đơn vị ghi kèm)", r.get("unit")) if c and c in self.cols[S]]
            uexpr = ("coalesce(" + ", ".join("CAST(s." + q(c) + " AS VARCHAR)" for c in ucols) + ")") if ucols else None
            fac = self.factors(S, r, ucols)
            if fac:
                ctes.append(values_cte("fmap", ["fi", "fu", "ff"], fac))
                joins.append(f"LEFT JOIN fmap ON fmap.fi = CAST(s.{q(r['item'])} AS VARCHAR) AND fmap.fu IS NOT DISTINCT FROM {uexpr}")
                qty = f"TRY_CAST(s.{q(r['qty'])} AS DOUBLE) * coalesce(fmap.ff, 1)"
            else:
                qty = f"TRY_CAST(s.{q(r['qty'])} AS DOUBLE)"
        else:
            qty = "NULL"
        # khách lẻ
        if "cust" in field:
            cv = [x[0] for x in self.db.execute("WITH " + ",\n".join(ctes) + f"\nSELECT DISTINCT CAST({field['cust']} AS VARCHAR) FROM s " + " ".join(joins)).fetchall()]
            walk = [v for v in cv if v is not None and is_walkin(v)]
            cname = f"CASE WHEN {field['cust']} IS NULL" + (f" OR CAST({field['cust']} AS VARCHAR) IN ({', '.join(lit(v) for v in walk)})" if walk else "") + f" THEN NULL ELSE CAST({field['cust']} AS VARCHAR) END"
        else:
            cname = "NULL"
        if "cust_code" in field:
            cid = f"CASE WHEN {cname} IS NULL THEN NULL ELSE CAST({field['cust_code']} AS VARCHAR) END" if "cust" in field else f"CAST({field['cust_code']} AS VARCHAR)"
        else:
            cid = "NULL"
        items = [x[0] for x in self.db.execute("WITH " + ",\n".join(ctes) + f"\nSELECT DISTINCT CAST({field['item']} AS VARCHAR) FROM s " + " ".join(joins)).fetchall() if x[0] is not None]
        adj = [v for v in items if ADJUST.search(key(v))]
        svc = [v for v in items if key(v) == "phi phuc vu"]
        kind = "CASE" + (f" WHEN {field['item']} IN ({', '.join(lit(v) for v in svc)}) THEN 'service'" if svc else "") + \
               (f" WHEN {field['item']} IN ({', '.join(lit(v) for v in adj)}) THEN 'adjust'" if adj else "") + " ELSE 'item' END"
        if kind == "CASE ELSE 'item' END":
            kind = "'item'"
        date = f"CAST({field['date']} AS DATE)"
        ts = f"CAST({field['date']} AS TIMESTAMP)" if self.types[S].get(r.get("date", ""), "").startswith("TIMESTAMP") or "date" not in r else "NULL"
        if "time" in field:
            tcol = field["time"]
            ts = f"TRY_CAST({tcol} AS TIMESTAMP)"
        ret = f"({cls} = 'returned'" + "".join(f" OR {x}" for x in ret_extra) + ")"
        sel = f"""SELECT {field['order']} AS order_id, min({date}) OVER (PARTITION BY {field['order']}) AS order_date, {ts} AS order_ts,
  {('CAST(' + field['item_code'] + ' AS VARCHAR)') if 'item_code' in field else 'NULL'} AS product_code, {field.get('group', 'NULL')} AS product_group,
  {field['item']} AS product_name, {qty} AS qty, {('s.' + q(r['unit'])) if 'unit' in r else 'NULL'} AS unit,
  TRY_CAST(s.{q(r['amount'])} AS DOUBLE) AS amount, {cid} AS customer_id, {cname} AS customer_name,
  {field.get('staff', 'NULL')} AS staff, {field.get('channel', 'NULL')} AS channel, {field.get('pay', 'NULL')} AS payment,
  ({cls} = 'cancel') AS is_cancelled, {ret} AS is_returned, {kind} AS line_kind
FROM s {' '.join(joins)}
WHERE {field['date']} IS NOT NULL"""
        return "WITH " + ",\n".join(ctes) + "\n" + sel, info

    def factors(self, S, r, ucols):
        if not ucols or "item" not in r:
            return []
        table, via = {}, {}
        for old, rr in self.roles.items():
            if {"item", "big_unit", "factor"} <= set(rr) and old in self.up:
                T = self.up[old]
                dv = rr["factor"] + " (đơn vị ghi kèm)"
                has_dv = dv in self.cols[T]
                cols = [q(rr["item"]), q(rr["big_unit"]), q(rr["factor"])] + ([q(dv)] if has_dv else [])
                for row in self.db.execute(f"SELECT {', '.join(cols)} FROM {q(T)}").fetchall():
                    try:
                        f = float(row[2])
                    except (TypeError, ValueError):
                        continue
                    table[(key(row[0]), key(row[1]))] = f
                    if has_dv and row[3]:
                        via[(key(row[0]), key(row[1]))] = key(row[3])
        for kk, w in via.items():
            table[kk] = table[kk] * table.get((kk[0], w), 1.0)
        if not table:
            return []
        uexpr = "coalesce(" + ", ".join(f"CAST({q(c)} AS VARCHAR)" for c in ucols) + ")"
        pairs = self.db.execute(f"SELECT DISTINCT CAST({q(r['item'])} AS VARCHAR), {uexpr} FROM {q(S)}").fetchall()
        return [(a, u, table[(key(a), key(u))]) for a, u in pairs if a is not None and (key(a), key(u)) in table]


# ------------------------------------------------------------------ dịch lời gọi lớp cũ → SQL mẫu
def to_sql(rec: dict):
    tpl, calls = rec["template"], rec["calls"]
    P = lambda c, i=0: (c[1][i]["start"], c[1][i]["end"])  # noqa: E731
    c0 = calls[0]
    if tpl == "revenue":
        c = next(c for c in calls if c[0] == "revenue")
        s, e = P(c)
        a = c[1] + [None] * 3
        return G.revenue(s, e, item=a[1], group=a[2])
    if tpl == "n_orders":
        s, e = P(c0)
        a = c0[1] + [None] * 3
        return G.n_orders(s, e, item=a[1], group=a[2])
    if tpl == "cancel_rate":
        return G.cancel_rate(*P(c0))
    if tpl == "n_cancelled":
        return G.n_cancelled(*P(c0))
    if tpl == "avg_order":
        return G.avg_order(*P(c0))
    if tpl == "qty":
        return G.qty(*P(c0), item=c0[1][1])
    if tpl == "best_day":
        return G.best_day(*P(c0))
    if tpl == "rev_change_pct":
        (s, e), (bs, be) = P(calls[0]), P(calls[1])
        return G.rev_change_pct(s, e, bs, be)
    if tpl == "share_group":
        return G.share_group(*P(c0), group=c0[1][1])
    if tpl in ("top_items", "top_item"):
        k = len(rec["old_answer"]) if tpl == "top_items" else 1
        return G.top_items(*P(c0), k=k, group=c0[1][1] if len(c0[1]) > 1 else None, as_string=tpl == "top_item")
    if tpl in ("top_customers", "top_customer"):
        k = len(rec["old_answer"]) if tpl == "top_customers" else 1
        return G.top_customers(*P(c0), k=k, as_string=tpl == "top_customer", by=c0[1][1] if len(c0[1]) > 1 and c0[1][1] else "money")
    if tpl == "days_over":
        return G.days_over(*P(c0), threshold=rec["threshold"])
    if tpl == "orders_over":
        return G.orders_over(*P(c0), threshold=rec["threshold"])
    if tpl == "best_month":
        s, e = P(calls[0])[0], P(calls[-1])[1]
        return G.best_month(s, e)
    raise G.Unsupported(tpl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shops", nargs="*")
    a = ap.parse_args()
    params = json.loads(a.params.read_text(encoding="utf-8"))
    gold = {x["id"]: x for x in map(json.loads, open(TRAIN / "answers.jsonl", encoding="utf-8"))}
    qtext = {x["id"]: x["question"] for x in map(json.loads, open(TRAIN / "questions.jsonl", encoding="utf-8"))}
    index = {json.loads(x)["business_id"]: json.loads(x) for x in open(INDEX, encoding="utf-8")}
    by_shop = collections.defaultdict(list)
    for qid, rec in params.items():
        if "calls" in rec:
            by_shop[rec["business_id"]].append((qid, rec))
    res, shops = {}, {}
    for b in sorted(by_shop):
        if a.shops and b not in a.shops:
            continue
        db, meta = ingest.load_business(TRAIN, b, canon=True)
        db.execute("SET enable_external_access = false")
        try:
            sql, info = Mapper(db, meta, index[b]["table_roles"]).build()
            chk = views.check(db, ["```sql\n" + sql + "\n```"])[0]
            fp = views.build(db, sql)
            shops[b] = {"view": True, "check_ok": chk[0] is not None, "check_error": None if chk[0] else chk[1][:200], "fingerprint": list(fp),
                        "status_classes": dict(info.get("status_values") or {})}
        except Exception as e:  # noqa: BLE001
            shops[b] = {"view": False, "error": str(e)[:300]}
            for qid, rec in by_shop[b]:
                res[qid] = {"b": b, "template": rec["template"], "result": "no_view"}
            db.close()
            continue
        for qid, rec in by_shop[b]:
            try:
                kind, s, _ = to_sql(rec)
            except G.Unsupported:
                res[qid] = {"b": b, "template": rec["template"], "result": "unsupported"}
                continue
            text = f"TYPE: {kind}\n```sql\n{s}\n```"
            try:
                pred = sqlexec.execute(db, text, qtext[qid])
                sc = scoring.score(pred, gold[qid])
                res[qid] = {"b": b, "template": rec["template"], "result": "ok" if sc == 1 else "wrong", "check_ok": shops[b]["check_ok"]}
            except Exception as e:  # noqa: BLE001
                res[qid] = {"b": b, "template": rec["template"], "result": "error", "error": str(e)[:160], "check_ok": shops[b]["check_ok"]}
        db.close()
        n = collections.Counter(r["result"] for r in res.values() if r["b"] == b)
        print(f"{b}: view={shops[b]['view']} check={shops[b].get('check_ok')} {dict(n)}", flush=True)
    by_t = collections.defaultdict(collections.Counter)
    for r in res.values():
        by_t[r["template"]][r["result"]] += 1
    appl = [r for r in res.values() if r["result"] in ("ok", "wrong", "error")]
    ok = sum(r["result"] == "ok" for r in appl)
    summary = {"applicable": len(appl), "ok": ok, "match_rate": round(ok / len(appl), 4) if appl else None,
               "by_template": {k: dict(v) for k, v in sorted(by_t.items())}, "shops": shops,
               "not_applicable": dict(collections.Counter(r["result"] for r in res.values() if r["result"] not in ("ok", "wrong", "error")))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "per_question": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "shops"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
