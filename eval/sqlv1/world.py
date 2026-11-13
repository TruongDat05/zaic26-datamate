"""Thế giới shop theo quy ước BTC (rules_spec btc_v1): bản ghi, oracle độc lập, đột biến đối chứng.

Bản ghi gốc lấy từ synth_world.simulate (giữ nguyên 132 hạt cũ), rồi `to_btc` thêm phần BTC có mà thế giới cũ không có:
bốn trạng thái (đang giao = đơn thường), trả hàng ghi trên đơn/dòng/phiếu kể cả trả một phần, chiết khấu dòng, voucher và phí
giao cấp đơn, dòng phí phục vụ, trả nợ theo từng đơn. Oracle là vòng lặp Python + Fraction, không dùng pandas/duckdb/SQL.
"""
from __future__ import annotations

import collections
import copy
import datetime as dt
import random
from fractions import Fraction

import synth_world as W

ORACLE_VERSION = "btc_v1.3"
CORE = ("tap_hoa", "quan_nhau", "ca_phe", "vlxd", "nha_thuoc")

TEXT = {"done": ["Hoàn thành", "hoàn thành", "Xong", "xong", "Đã giao", "Thành công", "Hoàn tất", "Đã lấy", "Đã thanh toán"],
        "cancel": ["Hủy", "huỷ", "Đã hủy", "Khách hủy", "Không giao được", "Hủy đơn", "Đã huỷ"],
        "returned": ["Trả hàng", "Khách trả hàng", "Đổi trả", "Hoàn trả", "Trả lại hàng", "trả hàng"],
        "transit": ["Đang giao", "đang giao", "Đang vận chuyển", "Đang giao hàng", "Chờ giao"]}
NOTE_TEXT = {"done": ["", "", "", "gói quà", "giao buổi sáng", "khách quen", "xuất hoá đơn đỏ"],
             "cancel": ["khách hủy", "không giao được", "hủy đơn"],
             "returned": ["khách trả 1 phần", "trả hàng 1 phần", "trả lại hàng"],
             "transit": ["đang giao", "đang giao hàng"]}
WALKIN = ["Khách lẻ", "khách lẻ", "KL", "kl", "Khách vãng lai", "", "lẻ"]
MARKET = "Sàn TMĐT"
DELIVERY = ("Giao hàng", "Giao tận nhà", "Giao thuốc", "Giao công trình", "Giao hoa", "Giao tận ruộng", "Giao lắp đặt", "GrabFood", "Website",
            "Facebook", "Zalo", MARKET, "Livestream", "Đặt trước")


class Degenerate(ValueError):
    """Đặc tả cho đáp án suy biến hoặc mơ hồ trên bản ghi này (bỏ, lấy tham số khác)."""


_d, _ts, resolve = W._d, W._ts, W.resolve


def money(x) -> int:
    return W._money(Fraction(x))


def pct(num, den) -> float:
    if W.half_tie(num, den):
        raise Degenerate("half tie")
    return W._pct(num, den)


def avg2(num, den) -> float:
    v = Fraction(num) / Fraction(den)
    if (v * 100).denominator == 2:
        raise Degenerate("half tie")
    return W._pct(v, 100)


def name_key(s) -> str:
    """Khoá khách theo tên (không mã): như views.build — bỏ dấu, chữ thường, chỉ giữ chữ số."""
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", str(s or "").replace("đ", "d").replace("Đ", "D"))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]", "", t)


# ------------------------------------------------------------------ bản ghi BTC
def to_btc(old: dict, opts: dict) -> dict:
    """Bản ghi theo quy ước BTC. opts: {codes_on_orders, service_fee, line_discount, ship_fee, voucher, note_status, same_name}."""
    rec = copy.deepcopy(old)
    rng = random.Random(f"btc:{old['seed']}:{old['industry']}")
    rec["rules"], rec["oracle_version"], rec["opts"] = "btc_v1", ORACLE_VERSION, dict(opts)
    items = {it["item"]: it for it in rec["items"]}
    lines = collections.defaultdict(list)
    for ln in rec["lines"]:
        ln["discount"], ln["kind"] = 0, "item"
        lines[ln["order"]].append(ln)
    st_pick = {k: rng.sample(v, min(len(v), 3)) for k, v in TEXT.items()}
    rec["status_text"] = st_pick
    rec["walkin_text"] = rng.sample(WALKIN, 3)
    refunded = collections.defaultdict(list)
    for r in rec["refunds"]:
        refunded[r["order"]].append(r)
    rec["returns"] = []
    for o in rec["orders"]:
        st = {"pending": "transit"}.get(o["status"], o["status"])
        if o["order"] in refunded and st == "done":
            st = "returned"
            for r in refunded[o["order"]]:
                rec["returns"].append({"ret": r["refund"], "order": o["order"], "at": r["at"], "item": r["item"], "unit": r["unit"], "qty": r["qty"],
                                       "amount": r["amount"]})
        o["status"] = st
        o["status_text"] = rng.choice(st_pick[st])
        o["note"] = rng.choice(NOTE_TEXT[st]) if opts.get("note_status") else ""
        if o["cust"] is None:
            o["walkin_text"] = rng.choice(rec["walkin_text"])
        o["ship_fee"], o["voucher"] = 0, 0
        if opts.get("ship_fee") and o["channel"] in DELIVERY and rng.random() < 0.6:
            o["ship_fee"] = rng.choice([10000, 15000, 20000, 25000, 30000])
        if opts.get("voucher") and rng.random() < 0.06:
            o["voucher"] = rng.choice([5000, 10000, 20000, 50000])
        if o["payment"] == "Tiền mặt" and o["channel"] in DELIVERY and o["channel"] != MARKET and rng.random() < 0.35:
            o["payment"] = "COD"
        if o["channel"] == MARKET and o["payment"] == "Ghi nợ":
            o["payment"] = "Chuyển khoản"
    del rec["refunds"]
    if opts.get("line_discount"):
        for ln in rec["lines"]:
            if rng.random() < 0.08 and ln["qty"] * ln["price"] >= 20000:
                ln["discount"] = int(round(ln["qty"] * ln["price"] * rng.choice([0.05, 0.1]), -3))
                ln["amount"] = ln["qty"] * ln["price"] - ln["discount"]
    if opts.get("service_fee"):  # dòng phí phục vụ: tính doanh thu, không phải mặt hàng
        for o in rec["orders"]:
            if rng.random() < 0.35:
                base = sum(ln["amount"] for ln in lines[o["order"]])
                fee = int(round(base * 0.05, -3))
                if fee > 0:
                    ln = {"order": o["order"], "line": len(lines[o["order"]]) + 1, "item": "PV", "unit": "lần", "qty": 1, "price": fee, "amount": fee,
                          "discount": 0, "kind": "service"}
                    rec["lines"].append(ln)
                    lines[o["order"]].append(ln)
    # trả nợ theo từng đơn ghi nợ (thay khoản trả theo khách của thế giới cũ)
    rec["payments"] = []
    end = _d(rec["data_end"])
    for o in rec["orders"]:
        if o["payment"] != "Ghi nợ" or o["status"] == "cancel":
            continue
        tot = order_total(o, lines[o["order"]])
        d0 = _d(o["at"])
        paid = 0
        if rng.random() < 0.35:
            up = int(round(tot * rng.choice([0.2, 0.3, 0.5]), -3))
            if 0 < up < tot:
                rec["payments"].append({"payment": "", "order": o["order"], "at": o["at"], "amount": up})
                paid = up
        if rng.random() < 0.65:
            pd_ = d0 + dt.timedelta(days=rng.randint(2, 40))
            if pd_ <= end:
                amt = tot - paid if rng.random() < 0.55 else int(round((tot - paid) * rng.choice([0.3, 0.5]), -3))
                if amt > 0:
                    rec["payments"].append({"payment": "", "order": o["order"], "at": dt.datetime.combine(pd_, dt.time(rng.randint(8, 19), 0)).isoformat(sep=" "),
                                            "amount": amt})
    rec["payments"].sort(key=lambda p: (p["at"], p["order"]))
    for i, p in enumerate(rec["payments"], 1):
        p["payment"] = f"PT{i:05d}"
    for x in rec["purchases"]:
        x["cancelled"] = False
    for x in rec["purchases"]:
        if rng.random() < 0.03:
            x["cancelled"] = True
    if rec["features"].get("cost"):
        fac = {(u["item"], u["unit"]): u["factor"] for u in rec["units"]}
        for ln in rec["lines"]:
            if ln["kind"] == "item":
                ln["cost"] = items[ln["item"]]["cost"] * fac[(ln["item"], ln["unit"])]
    if opts.get("same_name") and len(rec["customers"]) > 6:  # hai mã khách trùng tên: theo mã là hai khách
        rec["customers"][-1]["name"] = rec["customers"][5]["name"]
    rec["lines"].sort(key=lambda ln: (ln["order"], ln["line"]))
    return rec


def order_total(o, lns) -> int:
    return sum(ln["amount"] for ln in lns) - o.get("voucher", 0) + o.get("ship_fee", 0)


# ------------------------------------------------------------------ chỉ mục cho oracle
class View:
    """Bản ghi nhìn tại hôm nay T: bỏ đơn đặt sau T; lớp đơn theo ghi nhận trên đơn (kể cả phiếu trả sau T)."""

    def __init__(self, rec: dict, today):
        self.rec, self.T = rec, _d(today)
        self.items = {it["item"]: it for it in rec["items"]}
        self.fac = {(u["item"], u["unit"]): u["factor"] for u in rec["units"]}
        self.cust = {c["cust"]: c for c in rec["customers"]}
        self.lines = collections.defaultdict(list)
        for ln in rec["lines"]:
            self.lines[ln["order"]].append(ln)
        self.orders = [o for o in rec["orders"] if _d(o["at"]) <= self.T]
        self.by_code = rec["opts"].get("codes_on_orders", False)

    def valid(self, o) -> bool:
        return o["status"] in ("done", "transit")

    def orders_in(self, s, e, pred=None):
        return [o for o in self.orders if s <= _d(o["at"]) <= e and (pred is None or pred(o))]

    def vlines(self, s, e, kind=None):
        """(đơn, dòng) của đơn valid trong [s, e]; kind = 'item' bỏ dòng phí phục vụ."""
        return [(o, ln) for o in self.orders_in(s, e, self.valid) for ln in self.lines[o["order"]] if kind is None or ln["kind"] == kind]

    def base_qty(self, ln) -> int:
        return ln["qty"] * self.fac[(ln["item"], ln["unit"])]

    def ckey(self, o):
        """Khoá khách của đơn hoặc None (khách lẻ)."""
        if o["cust"] is None:
            return None
        return o["cust"] if self.by_code else name_key(self.cust[o["cust"]]["name"])

    def cname(self, key):
        return self.cust[key]["name"] if self.by_code else next(c["name"] for c in self.cust.values() if name_key(c["name"]) == key)


def _rank(score: dict, k: int, tie_key, ambiguous_ties=False):
    """Top k theo điểm giảm dần, đồng hạng theo tie_key tăng dần. ambiguous_ties: đồng hạng chạm ranh giới k → Degenerate."""
    order = sorted((x for x in score if score[x] > 0), key=lambda x: (-score[x], tie_key(x)))
    if len(order) < k:
        raise Degenerate("not enough objects")
    if ambiguous_ties:
        top = [score[x] for x in order[:k + 1]]
        if len(set(top)) < len(top):
            raise Degenerate("tie without codes")
    return order[:k]


# ------------------------------------------------------------------ oracle
def gold(rec: dict, spec: dict) -> dict:
    v = View(rec, spec["as_of"])
    slot, p = spec["slot"], spec.get("params") or {}
    if slot == "unanswerable":
        return {"type": "unanswerable"}
    s, e = resolve(spec["period"], spec["as_of"]) if spec.get("period") else (None, None)
    data0, data1 = _d(rec["start"]), _d(rec["data_end"])
    if s is not None and spec["period"]["kind"] not in ("until", "until_today", "lots"):
        if e < data0 or s > min(data1, v.T):
            return {"type": "unanswerable"}  # kỳ nằm hoàn toàn ngoài dữ liệu (R10)
        if spec["period"]["kind"] not in W.RELATIVE and e > v.T and data1 > v.T:
            raise Degenerate("named period runs past today while data goes on")
    typ = SLOT_TYPE[slot]
    val = _measure(v, rec, spec, slot, p, s, e)
    return {"type": typ, "answer": val}


def _rev(v, s, e, pred=None, kind=None):
    return sum(ln["amount"] for o, ln in v.vlines(s, e, kind) if pred is None or pred(o, ln))


def _debt_orders(v, x):
    """Đơn ghi nợ không hủy, đặt đến hết x: (đơn, nợ còn tại x)."""
    paid = collections.Counter()
    for pm in v.rec["payments"]:
        if _d(pm["at"]) <= x:
            paid[pm["order"]] += pm["amount"]
    out = []
    for o in v.rec["orders"]:
        if o["payment"] == "Ghi nợ" and o["status"] != "cancel" and o["channel"] != MARKET and _d(o["at"]) <= x:
            debt = order_total(o, v.lines[o["order"]]) - paid[o["order"]]
            if debt > 0:
                out.append((o, debt))
    return out


def _measure(v, rec, spec, slot, p, s, e):
    item = p.get("item")
    if slot == "revenue":
        return _rev(v, s, e)
    if slot == "revenue_item":
        return _rev(v, s, e, lambda o, ln: ln["item"] == item)
    if slot == "revenue_group":
        return _rev(v, s, e, lambda o, ln: ln["kind"] == "item" and v.items[ln["item"]]["group"] == p["group"])
    if slot == "revenue_channel":
        return _rev(v, s, e, lambda o, ln: o["channel"] == p["channel"])
    if slot == "revenue_payment":
        return _rev(v, s, e, lambda o, ln: o["payment"] == p["payment"])
    if slot == "n_orders":
        return len(v.orders_in(s, e, v.valid if "channel" not in p else (lambda o: v.valid(o) and o["channel"] == p["channel"])))
    if slot == "n_orders_item":
        return len({o["order"] for o, ln in v.vlines(s, e) if ln["item"] == item})
    if slot == "n_cancelled":
        return len(v.orders_in(s, e, lambda o: o["status"] == "cancel"))
    if slot == "n_returned":
        return len(v.orders_in(s, e, lambda o: o["status"] == "returned"))
    if slot == "cancel_rate":
        allo = v.orders_in(s, e)
        if not allo:
            raise Degenerate("no orders")
        return pct(sum(o["status"] == "cancel" for o in allo), len(allo))
    if slot == "avg_order":
        n = len(v.orders_in(s, e, v.valid))
        if not n:
            raise Degenerate("no orders")
        return money(Fraction(_rev(v, s, e), n))
    if slot == "qty_base":
        return sum(v.base_qty(ln) for o, ln in v.vlines(s, e, "item") if ln["item"] == item)
    if slot in ("top_items", "top_item", "best_item_rev"):
        sc = collections.Counter()
        for o, ln in v.vlines(s, e, "item"):
            sc[ln["item"]] += v.base_qty(ln) if slot != "best_item_rev" else ln["amount"]
        k = p.get("k", 1) if slot == "top_items" else 1
        r = _rank(sc, k, lambda c: c)
        names = [v.items[c]["name"] for c in r]
        return names if slot == "top_items" else names[0]
    if slot in ("best_day", "best_month", "min_month"):
        dd = collections.Counter()
        for o, ln in v.vlines(s, e):
            d = _d(o["at"])
            dd[d if slot == "best_day" else (d.year, d.month)] += ln["amount"]
        if slot == "min_month":
            y, m = s.year, s.month
            while dt.date(y, m, 1) <= min(e, v.T):
                dd.setdefault((y, m), 0)
                y, m = W._month_add(y, m, 1)
            if len(dd) < 2:
                raise Degenerate("one month")
            y, m = min(dd, key=lambda k: (dd[k], k))
            return f"{y:04d}-{m:02d}"
        if not dd or (slot == "best_month" and len(dd) < 2):
            raise Degenerate("no sales")
        best = min(dd, key=lambda k: (-dd[k], k))
        return best.isoformat() if slot == "best_day" else f"{best[0]:04d}-{best[1]:02d}"
    if slot in ("top_customer", "top_customers", "top_customer_orders"):
        sc = collections.Counter()
        if slot == "top_customer_orders":
            for o in v.orders_in(s, e, v.valid):
                if v.ckey(o):
                    sc[v.ckey(o)] += 1
        else:
            for o, ln in v.vlines(s, e):
                if v.ckey(o):
                    sc[v.ckey(o)] += ln["amount"]
        k = p.get("k", 1) if slot == "top_customers" else 1
        r = _rank(sc, k, lambda c: c, ambiguous_ties=not v.by_code)
        names = [v.cname(c) for c in r]
        if len({name_key(x) for x in names}) < len(names):
            raise Degenerate("same-name customers in answer")
        return names if slot == "top_customers" else names[0]
    if slot == "n_customers":
        return len({v.ckey(o) for o in v.orders_in(s, e, v.valid) if v.ckey(o)})
    if slot == "cust_ge_n":
        c = collections.Counter(v.ckey(o) for o in v.orders_in(s, e, v.valid) if v.ckey(o))
        return sum(1 for x in c.values() if x >= p["n"])
    if slot == "exists_item_day":
        return any(ln["item"] == item for o, ln in v.vlines(s, e))
    if slot == "group_table":
        g = collections.Counter()
        for o, ln in v.vlines(s, e, "item"):
            g[v.items[ln["item"]]["group"]] += ln["amount"]
        rows = [[k, x] for k, x in sorted(g.items()) if x > 0]
        if len(rows) < 2:
            raise Degenerate("table too small")
        return {"columns": ["Nhóm hàng", "Doanh thu"], "rows": rows}
    if slot == "unsold_items":
        before = {ln["item"] for o, ln in v.vlines(dt.date(1900, 1, 1), s - dt.timedelta(days=1), "item")}
        now = {ln["item"] for o, ln in v.vlines(s, e, "item")}
        if not before:
            raise Degenerate("nothing sold before the period")
        return len(before - now)
    if slot in ("rev_change_pct", "rev_change", "g_rev_change"):
        bs, be = resolve(spec["base_period"], spec["as_of"])
        if be < _d(rec["start"]):
            raise Degenerate("base period outside data")
        pred = (lambda o, ln: ln["kind"] == "item" and v.items[ln["item"]]["group"] == p["group"]) if slot == "g_rev_change" else None
        old, new = _rev(v, bs, be, pred), _rev(v, s, e, pred)
        if slot == "rev_change":
            return new - old
        if not old or new == old:
            raise Degenerate("base zero or no change")
        return pct(new - old, old)
    if slot in ("share_group", "shift_share", "rx_share", "share_channel"):
        tot = _rev(v, s, e)
        if slot == "share_group":
            part = _rev(v, s, e, lambda o, ln: ln["kind"] == "item" and v.items[ln["item"]]["group"] == p["group"])
        elif slot == "shift_share":
            part = _rev(v, s, e, lambda o, ln: _shift(o) == p["shift"])
        elif slot == "share_channel":
            part = _rev(v, s, e, lambda o, ln: o["channel"] == p["channel"])
        else:
            part = _rev(v, s, e, lambda o, ln: ln["kind"] == "item" and v.items[ln["item"]].get("rx"))
        if not tot or not part or part == tot:
            raise Degenerate("share degenerate")
        return pct(part, tot)
    if slot == "days_over":
        dd = collections.Counter()
        for o, ln in v.vlines(s, e):
            dd[_d(o["at"])] += ln["amount"]
        return sum(1 for x in dd.values() if x >= p["threshold"])
    if slot == "orders_over":
        tot = collections.Counter()
        for o, ln in v.vlines(s, e):
            tot[o["order"]] += ln["amount"]
        return sum(1 for x in tot.values() if x >= p["threshold"])
    if slot == "items_ge":
        q = collections.Counter()
        for o, ln in v.vlines(s, e, "item"):
            if "group" not in p or v.items[ln["item"]]["group"] == p["group"]:
                q[ln["item"]] += v.base_qty(ln)
        return sum(1 for x in q.values() if x >= p["n"])
    if slot == "rev_list":
        mm = collections.Counter()
        for o, ln in v.vlines(s, e):
            mm[(_d(o["at"]).year, _d(o["at"]).month)] += ln["amount"]
        out, (y, m) = [], (s.year, s.month)
        while dt.date(y, m, 1) <= e:
            out.append(mm[(y, m)])
            y, m = W._month_add(y, m, 1)
        if len(out) < 2:
            raise Degenerate("one month")
        return out
    if slot == "avg_daily_rev":
        return money(Fraction(_rev(v, s, e), (min(e, v.T) - s).days + 1))
    if slot in ("shift_rev", "shift_orders"):
        if slot == "shift_rev":
            return _rev(v, s, e, lambda o, ln: _shift(o) == p["shift"])
        return len(v.orders_in(s, e, lambda o: v.valid(o) and _shift(o) == p["shift"]))
    if slot == "staff_rev":
        return _rev(v, s, e, lambda o, ln: o.get("staff") == p["staff"])
    if slot == "staff_best":
        sc = collections.Counter()
        for o, ln in v.vlines(s, e):
            if o.get("staff"):
                sc[o["staff"]] += ln["amount"]
        return _rank(sc, 1, lambda x: x, ambiguous_ties=True)[0]
    if slot in ("rx_rev",):
        return _rev(v, s, e, lambda o, ln: ln["kind"] == "item" and v.items[ln["item"]].get("rx"))
    if slot in ("debt_total", "debt_n_cust", "top_debtor", "debt_cust", "debt_older"):
        x = min(e, v.T)
        ds = _debt_orders(v, x)
        if slot == "debt_total":
            return sum(d for _, d in ds)
        if slot == "debt_older":
            return sum(d for o, d in ds if (x - _d(o["at"])).days > p["days"])
        by = collections.Counter()
        for o, d in ds:
            if v.ckey(o):
                by[v.ckey(o)] += d
        if slot == "debt_n_cust":
            return len(by)
        if slot == "top_debtor":
            r = _rank(by, 1, lambda c: c, ambiguous_ties=not v.by_code)
            return v.cname(r[0])
        keys = [k for k in by if name_key(v.cname(k)) == name_key(p["customer"])]
        if len({c["cust"] for c in v.cust.values() if name_key(c["name"]) == name_key(p["customer"])}) != 1:
            raise Degenerate("customer name not unique")
        val = sum(by[k] for k in keys)
        if val <= 0:
            raise Degenerate("customer owes nothing")
        return val
    if slot in ("debt_collected", "debt_new"):
        credit = {o["order"]: o for o in v.rec["orders"] if o["payment"] == "Ghi nợ" and o["status"] != "cancel" and o["channel"] != MARKET}
        if slot == "debt_collected":
            return sum(pm["amount"] for pm in v.rec["payments"] if pm["order"] in credit and s <= _d(pm["at"]) <= min(e, v.T)
                       and _d(pm["at"]) > _d(credit[pm["order"]]["at"]))
        tot = 0
        for o in credit.values():
            if s <= _d(o["at"]) <= min(e, v.T):
                same = sum(pm["amount"] for pm in v.rec["payments"] if pm["order"] == o["order"] and _d(pm["at"]) <= _d(o["at"]))
                tot += order_total(o, v.lines[o["order"]]) - same
        return tot
    if slot == "purchase_money":
        return sum(x["amount"] for x in rec["purchases"] if not x.get("cancelled") and s <= _d(x["at"]) <= min(e, v.T))
    if slot == "avg_cost":
        rows = [x for x in rec["purchases"] if not x.get("cancelled") and x["item"] == item and s <= _d(x["at"]) <= min(e, v.T)]
        qb = sum(x["qty"] * v.fac[(x["item"], x["unit"])] for x in rows)
        if not qb:
            raise Degenerate("no purchases")
        return money(Fraction(sum(x["amount"] for x in rows), qb))
    if slot == "lots_expiring":
        return sum(1 for lot in rec["lots"] if lot["qty_left"] > 0 and v.T <= _d(lot["expiry"]) <= v.T + dt.timedelta(days=p["n"]))
    if slot in ("price_on_day", "price_change_pct"):
        def price(day):
            ps = {Fraction(ln["price"], v.fac[(ln["item"], ln["unit"])]) for o in v.rec["orders"] if _d(o["at"]) == day
                  for ln in v.lines[o["order"]] if ln["item"] == item}
            if len(ps) != 1:
                raise Degenerate("no or several prices that day")
            return ps.pop()
        if slot == "price_on_day":
            pr = price(s)
            if pr.denominator != 1:
                raise Degenerate("fractional base price")
            return int(pr)
        a, b = price(s), price(e)
        if a == b:
            raise Degenerate("no price change")
        return pct(b - a, a)
    if slot == "profit_gross":
        return _rev(v, s, e) - sum(ln["qty"] * ln["cost"] for o, ln in v.vlines(s, e, "item"))
    if slot == "stock_qty":
        x = min(e, v.T)
        q = next(o["qty"] for o in rec["opening_stock"] if o["item"] == item)
        q += sum(r["qty"] * v.fac[(r["item"], r["unit"])] for r in rec["purchases"] if not r.get("cancelled") and r["item"] == item and _d(r["at"]) <= x)
        q -= sum(v.base_qty(ln) for o, ln in v.vlines(dt.date(1900, 1, 1), x, "item") if ln["item"] == item)
        return q
    raise ValueError("slot " + slot)


def _shift(o):
    h = _ts(o["at"]).hour
    return "sang" if 6 <= h <= 13 else "chieu" if 14 <= h <= 21 else None


SLOT_TYPE = {"revenue": "number", "revenue_item": "number", "revenue_group": "number", "revenue_channel": "number", "revenue_payment": "number",
             "n_orders": "number", "n_orders_item": "number", "n_cancelled": "number", "n_returned": "number", "cancel_rate": "percent",
             "avg_order": "number", "qty_base": "number", "top_items": "list", "top_item": "string", "best_item_rev": "string", "best_day": "date",
             "best_month": "date", "min_month": "date", "top_customer": "string", "top_customers": "list", "top_customer_orders": "string",
             "n_customers": "number", "cust_ge_n": "number", "exists_item_day": "boolean", "group_table": "table", "unsold_items": "number",
             "rev_change_pct": "percent", "rev_change": "number", "g_rev_change": "percent", "share_group": "percent", "shift_share": "percent",
             "rx_share": "percent", "share_channel": "percent", "days_over": "number", "orders_over": "number", "items_ge": "number", "rev_list": "list",
             "avg_daily_rev": "number", "shift_rev": "number", "shift_orders": "number", "staff_rev": "number", "staff_best": "string",
             "rx_rev": "number", "debt_total": "number", "debt_n_cust": "number", "top_debtor": "string", "debt_cust": "number",
             "debt_older": "number", "debt_collected": "number", "debt_new": "number", "purchase_money": "number", "avg_cost": "number",
             "lots_expiring": "number", "price_on_day": "number", "price_change_pct": "percent", "profit_gross": "number", "stock_qty": "number",
             "unanswerable": "unanswerable"}


# ------------------------------------------------------------------ đột biến đối chứng (mức shop, tất định theo hạt)
def _new_order(d, rng, at, status, cust, item, qty, unit=None, channel=None, payment="Tiền mặt", price=None, walk=None):
    items = {it["item"]: it for it in d["items"]}
    fac = {(u["item"], u["unit"]): u["factor"] for u in d["units"]}
    oid = f"DM{len(d['orders']) + 1:05d}"
    u = unit or items[item]["base"]
    pr = price if price is not None else items[item]["price"] * fac[(item, u)]
    o = {"order": oid, "at": at.isoformat(sep=" "), "status": status, "status_text": rng.choice(d["status_text"][status]), "channel": channel or d["orders"][0]["channel"],
         "cust": cust, "walkin_text": (walk if walk is not None else rng.choice(d["walkin_text"])) if cust is None else None, "payment": payment,
         "note": rng.choice(NOTE_TEXT[status]) if d["opts"].get("note_status") else "", "ship_fee": 0, "voucher": 0}
    if d.get("staff"):
        o["staff"] = d["staff"][0]
    d["orders"].append(o)
    ln = {"order": oid, "line": 1, "item": item, "unit": u, "qty": qty, "price": pr, "amount": qty * pr, "discount": 0, "kind": "item"}
    if d["features"].get("cost"):
        ln["cost"] = items[item]["cost"] * fac[(item, u)]
    d["lines"].append(ln)
    return o


def _finish(d, kind):
    d["orders"].sort(key=lambda o: (o["at"], o["order"]))
    d["lines"].sort(key=lambda ln: (ln["order"], ln["line"]))
    d["returns"].sort(key=lambda r: (r["at"], r["ret"]))
    d["mutation"] = kind
    return d


def m_status(rec):
    """Đổi lớp trạng thái: hoàn thành → trả một phần / hủy (kể cả 'không giao được') / đang giao; ngược lại."""
    d, rng = copy.deepcopy(rec), random.Random(f"{rec['seed']}:m_status")
    lines = collections.defaultdict(list)
    for ln in d["lines"]:
        lines[ln["order"]].append(ln)
    end = _d(d["data_end"])
    for o in d["orders"]:
        x = rng.random()
        if o["status"] == "done" and x < 0.07 and o["payment"] != "Ghi nợ":
            ln = next(ln for ln in lines[o["order"]] if ln["kind"] == "item")
            at = min(end, _d(o["at"]) + dt.timedelta(days=rng.randint(0, 6)))
            d["returns"].append({"ret": f"TM{len(d['returns']) + 1:05d}", "order": o["order"], "at": dt.datetime.combine(at, dt.time(17, 0)).isoformat(sep=" "),
                                 "item": ln["item"], "unit": ln["unit"], "qty": 1, "amount": ln["price"]})
            o["status"] = "returned"
        elif o["status"] == "done" and x < 0.12:
            o["status"] = "cancel"
            d["payments"] = [p for p in d["payments"] if p["order"] != o["order"]]
        elif o["status"] == "done" and x < 0.18:
            o["status"] = "transit"
        elif o["status"] in ("transit", "cancel") and x < 0.3 and o["payment"] != "Ghi nợ":
            o["status"] = "done"
        else:
            continue
        o["status_text"] = rng.choice(d["status_text"][o["status"]])
        if d["opts"].get("note_status"):
            o["note"] = rng.choice([t for t in NOTE_TEXT[o["status"]] if t] or [""])
    return _finish(d, "m_status")


def m_rank_units(rec):
    """Đồng hạng (mặt hàng theo số lượng gốc, khách theo tiền: mã nhỏ hơn đứng trước) + đổi hệ số đơn vị lớn."""
    d, rng = copy.deepcopy(rec), random.Random(f"{rec['seed']}:m_rank_units")
    for u in d["units"]:
        if u["factor"] > 1:
            u["factor"] += 2 if u["factor"] < 20 else 10
    v = View(d, _d(d["data_end"]))
    by_code = d["opts"].get("codes_on_orders")
    for (y, m) in [(x.year, x.month) for x in sorted({_d(o["at"]).replace(day=1) for o in d["orders"]})]:
        s, e = dt.date(y, m, 1), W._month_end(y, m)
        q = collections.Counter()
        for o, ln in v.vlines(s, e, "item"):
            q[ln["item"]] += v.base_qty(ln)
        top = sorted(q, key=lambda c: (-q[c], c))[:2]
        day = min(e, _d(d["data_end"]), s + dt.timedelta(days=14))
        if len(top) == 2 and q[top[0]] > q[top[1]]:
            _new_order(d, rng, dt.datetime.combine(day, dt.time(10, 30)), "done", None, top[1], q[top[0]] - q[top[1]])
        sp = collections.Counter()
        for o, ln in v.vlines(s, e):
            if o["cust"]:
                sp[o["cust"] if by_code else name_key(v.cust[o["cust"]]["name"])] += ln["amount"]
        ct = sorted(sp, key=lambda c: (-sp[c], c))[:2]
        if len(ct) == 2 and sp[ct[0]] > sp[ct[1]]:
            cust = ct[1] if by_code else next(c["cust"] for c in d["customers"] if name_key(c["name"]) == ct[1])
            it = d["items"][0]["item"]
            diff = sp[ct[0]] - sp[ct[1]] + (0 if by_code else 1000)  # theo tên: vượt hẳn (đồng hạng không có mã là mơ hồ)
            _new_order(d, rng, dt.datetime.combine(day, dt.time(11, 0)), "done", cust, it, 1, price=diff)
    return _finish(d, "m_rank_units")


def m_edges(rec):
    """Đơn sát ranh giới kỳ (23:59:59 cuối tháng/Chủ nhật, 00:00:00 đầu tháng/thứ Hai), đơn lớn của khách lẻ, đơn sau hôm nay."""
    d, rng = copy.deepcopy(rec), random.Random(f"{rec['seed']}:m_edges")
    T, end = _d(d["today"]), _d(d["data_end"])
    hot = max(d["items"], key=lambda it: it["price"])["item"]
    named = [c["cust"] for c in d["customers"]]
    y, m = _d(d["start"]).year, _d(d["start"]).month
    while dt.date(y, m, 1) <= end:
        me = W._month_end(y, m)
        if me <= end:
            _new_order(d, rng, dt.datetime.combine(me, dt.time(23, 59, 59)), "done", named[2 % len(named)], hot, 3)
        if dt.date(y, m, 1) > _d(d["start"]):
            _new_order(d, rng, dt.datetime.combine(dt.date(y, m, 1), dt.time(0, 0, 0)), "transit", None, hot, 2, walk="")
        mid = min(end, dt.date(y, m, min(16, me.day)))
        _new_order(d, rng, dt.datetime.combine(mid, dt.time(15, 20)), "done", None, hot, 25, walk=rng.choice(d["walkin_text"]))
        y, m = W._month_add(y, m, 1)
    mon = T - dt.timedelta(days=T.weekday())
    if mon - dt.timedelta(days=1) >= _d(d["start"]):
        _new_order(d, rng, dt.datetime.combine(mon - dt.timedelta(days=1), dt.time(23, 59, 59)), "done", named[0], hot, 2)
    _new_order(d, rng, dt.datetime.combine(mon, dt.time(0, 0, 1)), "done", named[1 % len(named)], hot, 2)
    _new_order(d, rng, dt.datetime.combine(T, dt.time(23, 59, 59)), "done", named[0], hot, 4)
    nxt = T + dt.timedelta(days=1)
    _new_order(d, rng, dt.datetime.combine(nxt, dt.time(0, 0, 30)), "done", named[0], hot, 30)
    d["data_end"] = max(end, nxt).isoformat()
    return _finish(d, "m_edges")


def m_amounts(rec):
    """Đổi tiền: giá vài dòng, chiết khấu, phí giao, voucher — bắt đáp án/ngưỡng ghi cứng."""
    d, rng = copy.deepcopy(rec), random.Random(f"{rec['seed']}:m_amounts")
    for ln in d["lines"]:
        if ln["kind"] == "item" and rng.random() < 0.15:
            ln["price"] += 1000
            ln["amount"] = ln["qty"] * ln["price"] - ln["discount"]
    for o in d["orders"]:
        if d["opts"].get("ship_fee") and o.get("ship_fee"):
            o["ship_fee"] += 5000
        if d["opts"].get("voucher") and rng.random() < 0.05:
            o["voucher"] = 10000
    return _finish(d, "m_amounts")


GENERIC = (("m_status", m_status), ("m_rank_units", m_rank_units), ("m_edges", m_edges), ("m_amounts", m_amounts))


def m_target(rec, specs):
    """Một bản sửa nhắm mọi đặc tả của shop: trong kỳ của từng câu thêm đơn hợp lệ đúng đối tượng câu nói (mặt hàng, nhóm, kênh,
    khách, ca, nhân viên), một đơn hủy, một đơn trả một phần của đơn đang hợp lệ, một đơn đang giao."""
    d, rng = copy.deepcopy(rec), random.Random(f"{rec['seed']}:m_target")
    lines = collections.defaultdict(list)
    for ln in d["lines"]:
        lines[ln["order"]].append(ln)
    for sp in specs:
        if sp["slot"] in ("unanswerable", "lots_expiring") or not sp.get("period"):
            continue
        T = _d(sp["as_of"])
        try:
            s, e = resolve(sp["period"], sp["as_of"])
        except ValueError:
            continue
        day = max(_d(d["start"]), min(e, T))
        p = sp.get("params") or {}
        items = [it for it in d["items"]]
        it = p.get("item") or (next((x["item"] for x in items if x["group"] == p.get("group")), None)) or rng.choice(items)["item"]
        cust = None
        if p.get("customer"):
            cust = next((c["cust"] for c in d["customers"] if c["name"] == p["customer"]), None)
        elif rng.random() < 0.5:
            cust = rng.choice(d["customers"])["cust"]
        hour = {"sang": 9, "chieu": 16}.get(p.get("shift"), rng.choice([9, 16]))
        o = _new_order(d, rng, dt.datetime.combine(day, dt.time(hour, 7)), rng.choice(["done", "transit"]), cust, it, rng.randint(1, 3),
                       channel=p.get("channel"), payment=p.get("payment", "Ghi nợ" if (cust and d["features"].get("debt") and rng.random() < 0.5) else "Tiền mặt"))
        if p.get("staff"):
            o["staff"] = p["staff"]
        _new_order(d, rng, dt.datetime.combine(day, dt.time(hour, 9)), "cancel", cust, it, 2, channel=p.get("channel"))
        live = [x for x in d["orders"] if x["status"] == "done" and s <= _d(x["at"]) <= min(e, T) and x["payment"] != "Ghi nợ" and lines.get(x["order"])]
        if live:
            x = rng.choice(live)
            ln = next((y for y in lines[x["order"]] if y["kind"] == "item"), None)
            if ln:
                x["status"], x["status_text"] = "returned", rng.choice(d["status_text"]["returned"])
                if d["opts"].get("note_status"):
                    x["note"] = NOTE_TEXT["returned"][0]
                d["returns"].append({"ret": f"TT{len(d['returns']) + 1:05d}", "order": x["order"], "at": x["at"], "item": ln["item"], "unit": ln["unit"], "qty": 1,
                                     "amount": ln["price"]})
    return _finish(d, "m_target")
