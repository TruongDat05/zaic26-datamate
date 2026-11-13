"""Lấy mẫu đặc tả câu hỏi (slot + kỳ + tham số + as_of) trên bản ghi BTC của một shop; đáp án chỉ do world.gold tính."""
from __future__ import annotations

import collections
import datetime as dt
import random

import synth_world as W
from sqlv1 import world as WB

_d = WB._d

# trọng số slot cho câu mới: bù nhóm còn thiếu so với MINING (giá bán, ngành, so sánh kỳ, phần trăm)
SLOT_W = {"revenue": 10, "revenue_item": 4, "revenue_group": 3, "revenue_channel": 2, "revenue_payment": 2, "n_orders": 5, "n_orders_item": 2,
          "n_cancelled": 3, "n_returned": 1, "cancel_rate": 4, "avg_order": 4, "qty_base": 5, "top_items": 3, "top_item": 3, "best_item_rev": 1,
          "best_day": 3, "best_month": 2, "min_month": 1, "top_customer": 2, "top_customers": 2, "top_customer_orders": 1, "n_customers": 2,
          "cust_ge_n": 1, "exists_item_day": 1, "group_table": 1, "unsold_items": 1, "rev_change_pct": 6, "rev_change": 2, "g_rev_change": 2,
          "share_group": 4, "share_channel": 1, "days_over": 2, "orders_over": 2, "items_ge": 1, "rev_list": 1, "avg_daily_rev": 1,
          "shift_rev": 2, "shift_share": 2, "shift_orders": 1, "staff_rev": 1, "staff_best": 1, "rx_rev": 3, "rx_share": 3, "debt_total": 3,
          "debt_n_cust": 1, "top_debtor": 2, "debt_cust": 2, "debt_older": 1, "debt_collected": 2, "debt_new": 1, "purchase_money": 2,
          "avg_cost": 2, "lots_expiring": 1, "price_on_day": 5, "price_change_pct": 4, "profit_gross": 2, "stock_qty": 2, "unanswerable": 2}
NEEDS = {"rx_rev": "rx", "rx_share": "rx", "debt_total": "debt", "debt_n_cust": "debt", "top_debtor": "debt", "debt_cust": "debt", "debt_older": "debt",
         "debt_collected": "debt", "debt_new": "debt", "purchase_money": "purchases", "avg_cost": "purchases", "lots_expiring": "lots",
         "profit_gross": "cost", "stock_qty": "stock", "staff_rev": "staff", "staff_best": "staff"}
NO_PRICE_FAMILIES = ("neg_lines", "events")  # sổ không ghi đơn giá dòng: không hỏi giá bán theo ngày
PERIOD_W = {"month": 56, "year": 5, "quarter": 6, "range": 4, "day": 4, "today": 2, "yesterday": 3, "this_month": 5, "last_month": 5,
            "this_week": 2, "last_week": 3, "last_n": 3, "this_year": 2}
STABLE = {"price_on_day", "price_change_pct", "exists_item_day", "best_item_rev", "group_table"}  # kỳ là ngày


def months_in(rec, T):
    d0 = _d(rec["start"])
    out, (y, m) = [], (d0.year, d0.month)
    while dt.date(y, m, 1) <= T:
        out.append((y, m))
        y, m = W._month_add(y, m, 1)
    return out


GEN_PERIOD_W = {"month": 76, "year": 6, "quarter": 3, "range": 2, "day": 3, "today": 1, "yesterday": 2, "this_month": 2, "last_month": 2,
                "this_week": 1, "last_week": 1, "last_n": 1, "this_year": 1}  # câu mới bù kho câu cũ (tháng cụ thể chỉ 17%)


def period(r: random.Random, rec, T, kind=None, weights=None):
    d0, d1 = _d(rec["start"]), _d(rec["data_end"])
    w = weights or PERIOD_W
    kind = kind or r.choices(list(w), list(w.values()))[0]
    ms = months_in(rec, T)
    full = [(y, m) for y, m in ms if W._month_end(y, m) <= T or d1 == T]
    if kind == "month":
        y, m = r.choice(full)
        return {"kind": "month", "y": y, "m": m}
    if kind == "quarter":
        qs = sorted({(y, (m - 1) // 3 + 1) for y, m in full})
        y, qn = r.choice(qs)
        return {"kind": "quarter", "y": y, "q": qn}
    if kind == "year":
        ys = sorted({y for y, _ in full})
        return {"kind": "year", "y": r.choice(ys)}
    if kind == "range":
        a = d0 + dt.timedelta(days=r.randint(0, max(0, (T - d0).days - 10)))
        b = min(T, a + dt.timedelta(days=r.randint(3, 25)))
        return {"kind": "range", "d1": a.isoformat(), "d2": b.isoformat()}
    if kind == "day":
        return {"kind": "day", "d": (d0 + dt.timedelta(days=r.randint(0, (T - d0).days))).isoformat()}
    if kind == "last_n":
        return {"kind": "last_n", "n": r.choice([7, 10, 14, 30])}
    return {"kind": kind}


def _threshold(v, r):
    if not v:
        return None
    x = sorted(v)[int(len(v) * r.uniform(0.4, 0.85))]
    mag = 10 ** max(4, len(str(int(x))) - 2)
    return int(round(x / mag) * mag) or mag


MONTH_SLOT_W = {"revenue": 14, "revenue_item": 4, "revenue_group": 3, "revenue_channel": 2, "revenue_payment": 1, "n_orders": 6, "n_orders_item": 2,
                "n_cancelled": 3, "cancel_rate": 4, "avg_order": 4, "qty_base": 4, "top_items": 2, "top_item": 2, "top_customer": 1, "top_customers": 1,
                "n_customers": 1, "rev_change_pct": 3, "rev_change": 1, "share_group": 2, "days_over": 1, "orders_over": 1, "debt_collected": 1,
                "debt_new": 1, "purchase_money": 1, "profit_gross": 1, "rx_rev": 1, "rx_share": 1, "shift_rev": 1}
MONTH_PERIOD_W = {"month": 90, "year": 10}


def sample(rec: dict, family: str, n: int, seed, force_slots=None, period_w=None, slot_w=None) -> list[dict]:
    """n đặc tả có gold không suy biến (thử lại khi Degenerate)."""
    r = random.Random(f"spec:{rec['seed']}:{seed}")
    T0 = _d(rec["today"])
    feats = rec["features"]
    v0 = WB.View(rec, T0)
    sold = collections.Counter(ln["item"] for o, ln in v0.vlines(_d(rec["start"]), T0, "item"))
    items = [c for c, _ in sold.most_common()] or [it["item"] for it in rec["items"]]
    multi = [c for c in items if any(u["item"] == c and u["factor"] > 1 for u in rec["units"])]
    groups = sorted({v0.items[c]["group"] for c in items})
    chans = sorted({o["channel"] for o in rec["orders"]})
    pays = sorted({o["payment"] for o in rec["orders"]})
    custs = [c["name"] for c in rec["customers"]]
    missing = W.INDUSTRIES[rec["industry"]]["missing"]
    W_ = slot_w or SLOT_W
    slots = [s for s in W_ if (NEEDS.get(s) is None or feats.get(NEEDS[s])) and not (s in ("price_on_day", "price_change_pct") and family in NO_PRICE_FAMILIES)]
    if not rec["opts"].get("codes_on_orders"):
        pass
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 30:
        tries += 1
        slot = r.choice(force_slots) if force_slots else r.choices(slots, [W_[s] for s in slots])[0]
        T = T0 if r.random() < 0.75 else T0 - dt.timedelta(days=r.randint(1, 45))
        if T <= _d(rec["start"]) + dt.timedelta(days=40):
            T = T0
        spec = {"slot": slot, "as_of": T.isoformat(), "params": {}, "base_period": None}
        p = spec["params"]
        kind = None
        if slot in STABLE:
            kind = "day" if slot != "group_table" else r.choice(["day", "month", "last_week"])
        if slot in ("debt_total", "debt_n_cust", "top_debtor", "debt_cust", "debt_older", "stock_qty"):
            kind = r.choice(["until", "until_today", "until_today"])
        if slot == "lots_expiring":
            spec["period"] = None
            p["n"] = r.choice([7, 15, 30, 60])
        elif kind == "until":
            spec["period"] = {"kind": "until", "d": (T - dt.timedelta(days=r.randint(0, 40))).isoformat()}
        elif slot == "price_change_pct":
            a = _d(rec["start"]) + dt.timedelta(days=r.randint(0, (T - _d(rec["start"])).days - 5))
            b = min(T, a + dt.timedelta(days=r.randint(10, 90)))
            spec["period"] = {"kind": "range", "d1": a.isoformat(), "d2": b.isoformat()}
        elif slot in ("best_month", "min_month", "rev_list"):
            spec["period"] = period(r, rec, T, r.choice(["quarter", "year", "this_year", "range"]))
            if spec["period"]["kind"] == "range":
                ms = months_in(rec, T)
                i = r.randint(0, max(0, len(ms) - 3))
                (y1, m1), (y2, m2) = ms[i], ms[min(len(ms) - 1, i + r.randint(1, 3))]
                spec["period"] = {"kind": "range", "d1": dt.date(y1, m1, 1).isoformat(), "d2": min(T, W._month_end(y2, m2)).isoformat()}
        elif slot in ("rev_change_pct", "rev_change", "g_rev_change"):
            pair = r.choice([("month", "month"), ("month", "month"), ("this_month", "prev_same_days"), ("last_week", "week_before_last"),
                             ("last_n", "prev_n"), ("quarter", "quarter"), ("last_month", "month_before")])
            if pair[0] == "month":
                ms = months_in(rec, T)[1:]
                y, m = r.choice([x for x in ms if W._month_end(*x) <= T] or ms)
                py, pm = W._month_add(y, m, -1)
                spec["period"], spec["base_period"] = {"kind": "month", "y": y, "m": m}, {"kind": "month", "y": py, "m": pm}
            elif pair[0] == "quarter":
                qs = sorted({(y, (m - 1) // 3 + 1) for y, m in months_in(rec, T) if W._month_end(y, m) <= T})
                if len(qs) < 2:
                    continue
                i = r.randint(1, len(qs) - 1)
                spec["period"], spec["base_period"] = {"kind": "quarter", "y": qs[i][0], "q": qs[i][1]}, {"kind": "quarter", "y": qs[i - 1][0], "q": qs[i - 1][1]}
            elif pair[0] == "last_month":
                y, m = W._month_add(T.year, T.month, -2)
                spec["period"], spec["base_period"] = {"kind": "last_month"}, {"kind": "month", "y": y, "m": m}
            elif pair[0] == "last_n":
                k = r.choice([7, 14, 30])
                spec["period"], spec["base_period"] = {"kind": "last_n", "n": k}, {"kind": "prev_n", "n": k}
            else:
                spec["period"], spec["base_period"] = {"kind": pair[0]}, {"kind": pair[1]}
        else:
            spec["period"] = period(r, rec, T, kind, period_w)
        if slot in ("revenue_item", "qty_base", "n_orders_item", "exists_item_day", "price_on_day", "price_change_pct", "avg_cost", "stock_qty"):
            pool = multi if slot == "qty_base" and multi and r.random() < 0.7 else items
            p["item"] = r.choice(pool[:max(3, len(pool))])
        if slot in ("revenue_group", "share_group", "g_rev_change") or (slot == "items_ge" and r.random() < 0.5):
            p["group"] = r.choice(groups)
        if slot in ("revenue_channel", "share_channel"):
            p["channel"] = r.choice(chans)
        if slot == "revenue_payment":
            p["payment"] = r.choice(pays)
        if slot in ("top_items", "top_customers"):
            p["k"] = r.choice([2, 3, 3, 5])
        if slot in ("shift_rev", "shift_share", "shift_orders"):
            p["shift"] = r.choice(["sang", "chieu"])
        if slot == "staff_rev":
            p["staff"] = r.choice(rec.get("staff") or ["?"])
        if slot == "debt_cust":
            p["customer"] = r.choice(custs)
        if slot == "debt_older":
            p["days"] = r.choice([7, 15, 30])
        if slot == "cust_ge_n":
            p["n"] = r.choice([2, 3, 5])
        if slot == "items_ge":
            p["n"] = r.choice([10, 20, 50, 100])
        if slot == "unanswerable":
            p["missing"] = r.choice(missing)
        if slot in ("days_over", "orders_over"):
            try:
                s_, e_ = W.resolve(spec["period"], T.isoformat())
            except ValueError:
                continue
            vv = WB.View(rec, T)
            agg = collections.Counter()
            for o, ln in vv.vlines(s_, e_):
                agg[_d(o["at"]) if slot == "days_over" else o["order"]] += ln["amount"]
            th = _threshold(list(agg.values()), r)
            if not th:
                continue
            p["threshold"] = th
        key = (slot, repr(spec["period"]), repr(spec.get("base_period")), repr(sorted(p.items())), spec["as_of"])
        if key in seen:
            continue
        try:
            g = WB.gold(rec, spec)
        except (WB.Degenerate, ValueError, KeyError, StopIteration, ZeroDivisionError):
            continue
        if g["type"] == "unanswerable" and slot != "unanswerable":
            if r.random() < 0.85:
                continue  # kỳ ngoài dữ liệu: giữ ít
        if g["type"] in ("number",) and g["answer"] == 0 and slot not in ("n_cancelled", "n_returned", "unsold_items", "days_over", "orders_over", "rev_change") and r.random() < 0.8:
            continue  # đáp án 0 hay do lọc rỗng: giữ ít
        seen.add(key)
        spec["gold"] = g
        out.append(spec)
    return out
