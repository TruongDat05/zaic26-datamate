"""Đột biến nhắm theo câu (m_probe) cho các ứng viên mà 5 bản đột biến chung chưa làm đổi đáp án.

    DATAMATE_UPSTREAM=<snapshot> python eval/sqlv1/probe.py --snapshot <snapshot> [--procs 12]
      → outputs/synth_sql_v1/sft_v2/mutation_report_probe.jsonl (báo cáo đầy đủ: bản cũ + m_probe) + probe_summary.json

Với mỗi câu có ứng viên chưa phân biệt (n_changed_ok = 0), thử lần lượt các biến đổi nhỏ trên bản ghi thế giới, nhắm đúng thứ câu
hỏi đọc (phiếu nhập trong kỳ, giá của mặt hàng ngày D, lô sắp hết hạn, khách mới / đơn lớn của khách khác, đơn ghi nợ, khoản trả nợ,
đơn lớn ngày khác, nhân viên khác…), tính gold mới bằng oracle (world.gold) và giữ biến đổi ĐẦU TIÊN làm đổi đáp án mà không suy
biến. Shop đột biến dựng lại bằng loader của snapshot, SQL chạy bằng guard + scorer cố định như mutation_replay. SQL trượt ở m_probe
→ ứng viên trượt (pass = false). Không tìm được biến đổi đổi đáp án (vd. câu unanswerable) → ghi 'no_change_found', không tính.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import datetime as dt
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--pkg", type=Path, default=controller_root() / "build/synthetic_sql_v1")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--tmp", type=Path, default=Path("/dev/shm/sftv2_probe"))
    ap.add_argument("--limit", type=int)
    return ap.parse_args()


A = _args() if __name__ == "__main__" else None
if A is not None:
    os.environ["DATAMATE_UPSTREAM"] = str(A.snapshot.resolve())
from sqlv1 import mutation_replay as MR  # noqa: E402
from sqlv1 import package as PK, render as RD, shops as SH, world as WB  # noqa: E402

WORK = MR.WORK
OUT = MR.OUT
SURN = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi", "Đỗ", "Ngô"]
GIVEN = ["Thị Hồng Nhung", "Văn Khải", "Minh Tâm", "Quốc Bảo", "Thu Trang", "Gia Huy", "Ngọc Ánh", "Đức Long"]
ORDER = {  # biến đổi ưu tiên theo dạng câu; sau đó thử mọi biến đổi chung
    "purchase_money": ["purchase_new", "purchase_bump"], "avg_cost": ["purchase_new", "purchase_bump"],
    "price_on_day": ["price_shift"], "price_change_pct": ["price_shift"], "lots_expiring": ["lot_new", "lot_drain"],
    "n_customers": ["new_cust_sale"], "cust_ge_n": ["cust_orders", "new_cust_sale"], "top_customer": ["cust_sale"], "top_customers": ["cust_sale"],
    "top_customer_orders": ["cust_orders"], "top_debtor": ["debt_sale"], "debt_n_cust": ["debt_sale"], "debt_collected": ["debt_payment"],
    "debt_new": ["debt_sale"], "debt_at": ["debt_sale", "debt_payment"], "staff_best": ["staff_sale"], "exists_item_day": ["cancel_item_day", "big_sale"],
    "best_month": ["big_amount_months"], "min_month": ["big_amount_months"], "best_day": ["big_amount"], "days_over": ["big_amount"], "orders_over": ["big_amount"],
    "top_items": ["big_qty"], "top_item": ["big_qty"], "items_ge": ["big_qty"], "best_item_rev": ["big_amount_item"], "unsold_items": ["sell_each_item"],
}
ORDER["top_debtor"] = ["debt_big", "debt_sale"]
ORDER["debt_n_cust"] = ["debt_new_cust", "debt_sale"]
ORDER["price_on_day"] = ["price_scale", "price_shift"]
ORDER["price_change_pct"] = ["price_shift", "price_scale"]
GENERIC = ["big_sale", "cust_sale", "new_cust_sale", "cust_orders", "price_shift", "purchase_new", "debt_sale", "staff_sale"]


def window(rec, spec):
    T = WB._d(spec["as_of"])
    start, end = WB._d(rec["start"]), min(WB._d(rec["data_end"]), T)
    if spec.get("period"):
        try:
            s, e = WB.resolve(spec["period"], spec["as_of"])
        except ValueError:
            s, e = start, end
    else:
        s, e = start, end
    s, e = max(s, start), min(e, end)
    return (s, e) if s <= e else (start, end)


def days_of(rec, spec):
    s, e = window(rec, spec)
    p = spec.get("period") or {}
    ds = [WB._d(p[k]) for k in ("d", "d1", "d2") if p.get(k)] or [s, e, s + (e - s) // 2]
    return list(dict.fromkeys(d for d in ds if s <= d <= e)) or [s]


def items_for(rec, spec, rng):
    p = spec.get("params") or {}
    its = [it["item"] for it in rec["items"] if not p.get("group") or it["group"] == p["group"]] or [it["item"] for it in rec["items"]]
    first = [p["item"]] if p.get("item") else []
    rest = [x for x in its if x not in first]
    rng.shuffle(rest)
    return first + rest[:3]


def new_customer(d, rng):
    used = {WB.name_key(c["name"]) for c in d["customers"]}
    for _ in range(50):
        name = f"{rng.choice(SURN)} {rng.choice(GIVEN)}"
        if WB.name_key(name) not in used:
            break
    code = f"KH{max(int(c['cust'][2:]) for c in d['customers'] if c['cust'][2:].isdigit()) + 1:03d}"
    d["customers"].append({"cust": code, "name": name, "kind": rng.choice(d["customers"])["kind"]})
    return code


def variants(rec, spec, kind, rng):
    """(tên, bản ghi đã biến đổi) của một loại biến đổi; vài biến thể (ngày, mặt hàng, khách)."""
    s, e = window(rec, spec)
    days, its = days_of(rec, spec), items_for(rec, spec, rng)
    at = lambda day, h=10, m=15: dt.datetime.combine(day, dt.time(h, m))  # noqa: E731
    debt_ok = rec["features"].get("debt")
    if kind in ("big_amount", "big_amount_item"):
        for day in days[:3]:
            for it in (its[:3] if kind == "big_amount_item" else its[:1]):
                d = copy.deepcopy(rec)
                WB._new_order(d, rng, at(day, 11), "done", None, it, 1, price=9_000_000)
                yield f"{kind}@{day}:{it}", d
    elif kind == "big_amount_months":
        y, m = s.year, s.month
        while dt.date(y, m, 1) <= e:
            day = min(e, max(s, dt.date(y, m, 15)))
            d = copy.deepcopy(rec)
            WB._new_order(d, rng, at(day, 11), "done", None, its[0], 1, price=9_000_000)
            yield f"big_amount_month@{day}", d
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    elif kind == "big_qty":
        for it in its[:4]:
            for day in days[:2]:
                d = copy.deepcopy(rec)
                WB._new_order(d, rng, at(day, 12), "done", None, it, 400)
                yield f"big_qty@{day}:{it}", d
    elif kind == "sell_each_item":
        for it in [x["item"] for x in rec["items"]][:25]:
            d = copy.deepcopy(rec)
            WB._new_order(d, rng, at(days[0], 12), "done", None, it, 1)
            yield f"sell_item@{days[0]}:{it}", d
    elif kind in ("debt_big", "debt_new_cust"):
        if not debt_ok:
            return
        custs = [c["cust"] for c in rec["customers"]]
        rng.shuffle(custs)
        for c in (custs[:3] if kind == "debt_big" else [None, None]):
            d = copy.deepcopy(rec)
            c = c or new_customer(d, rng)
            WB._new_order(d, rng, at(days[0]), "done", c, its[0], 1, payment="Ghi nợ", price=9_000_000)
            yield f"{kind}@{days[0]}:{c}", d
    elif kind == "price_scale":
        for day in days[:2]:
            for it in its[:2]:
                d = copy.deepcopy(rec)
                orders = {o["order"] for o in d["orders"] if WB._d(o["at"]) == day}
                lines = [ln for ln in d["lines"] if ln["order"] in orders and ln["item"] == it and ln["kind"] == "item"]
                if not lines or any(ln["price"] % 2 for ln in lines):
                    continue
                for ln in lines:  # giá × 3/2 trên mọi dòng: giá quy về đơn vị gốc vẫn một mức
                    ln["price"] = ln["price"] * 3 // 2
                    ln["amount"] = ln["qty"] * ln["price"] - ln["discount"]
                yield f"price_scale@{day}:{it}", d
    elif kind == "big_sale":
        for day in days[:3]:
            for it in its[:3]:
                d = copy.deepcopy(rec)
                WB._new_order(d, rng, at(day), "done", None, it, 40)
                yield f"big_sale@{day}:{it}", d
    elif kind in ("cust_sale", "cust_orders", "debt_sale"):
        custs = [c["cust"] for c in rec["customers"]]
        rng.shuffle(custs)
        for c in custs[:3]:
            for day in days[:2]:
                d = copy.deepcopy(rec)
                if kind == "cust_orders":
                    for k in range(12):
                        WB._new_order(d, rng, at(day, 8 + k % 10, 5 * (k // 10)), "done", c, its[0], 1)
                elif kind == "debt_sale":
                    if not debt_ok:
                        return
                    WB._new_order(d, rng, at(day), "done", c, its[0], 30, payment="Ghi nợ")
                else:
                    WB._new_order(d, rng, at(day), "done", c, its[0], 1, price=9_000_000)
                yield f"{kind}@{day}:{c}", d
    elif kind == "new_cust_sale":
        for day in days[:2]:
            d = copy.deepcopy(rec)
            c = new_customer(d, rng)
            WB._new_order(d, rng, at(day), "done", c, its[0], 2)
            yield f"new_cust_sale@{day}:{c}", d
    elif kind == "price_shift":
        for day in days[:2]:
            for it in its[:2]:
                d = copy.deepcopy(rec)
                orders = {o["order"] for o in d["orders"] if WB._d(o["at"]) == day}
                hit = 0
                for ln in d["lines"]:
                    if ln["order"] in orders and ln["item"] == it and ln["kind"] == "item":
                        ln["price"] += max(1000, int(round(ln["price"] * 0.1, -3)))
                        ln["amount"] = ln["qty"] * ln["price"] - ln["discount"]
                        hit += 1
                if hit:
                    yield f"price_shift@{day}:{it}", d
    elif kind in ("purchase_new", "purchase_bump"):
        if not rec["features"].get("purchases"):
            return
        for it in its[:2]:
            d = copy.deepcopy(rec)
            items = {x["item"]: x for x in d["items"]}
            if kind == "purchase_new":
                n = len(d["purchases"]) + 1
                day = days[0]
                d["purchases"].append({"receipt": f"PN{9000 + n}", "at": at(day, 9, 30).isoformat(sep=" "), "item": it, "unit": items[it]["base"],
                                       "qty": 7, "amount": 7 * int((items[it].get("cost") or items[it]["price"] * 0.7) * 1.3) // 100 * 100, "cancelled": False})
            else:
                hit = 0
                for pu in d["purchases"]:
                    if s <= WB._d(pu["at"]) <= e and not pu["cancelled"] and (pu["item"] == it or not (spec.get("params") or {}).get("item")):
                        pu["amount"] += max(1000, int(round(pu["amount"] * 0.1, -3)))
                        hit += 1
                if not hit:
                    continue
            yield f"{kind}:{it}", d
    elif kind in ("lot_new", "lot_drain"):
        if not rec["features"].get("lots"):
            return
        T, n = WB._d(spec["as_of"]), int((spec.get("params") or {}).get("n") or 30)
        if kind == "lot_new":
            for it in its[:2]:
                d = copy.deepcopy(rec)
                d["lots"].append({"lot": f"LO{9000 + len(d['lots'])}", "item": it, "expiry": (T + dt.timedelta(days=max(1, n // 3))).isoformat(), "qty_left": 9})
                yield f"lot_new:{it}", d
        else:
            for k, lot in enumerate(rec["lots"]):
                if T <= WB._d(lot["expiry"]) <= T + dt.timedelta(days=n) and lot["qty_left"] > 0:
                    d = copy.deepcopy(rec)
                    d["lots"][k]["qty_left"] = 0
                    yield f"lot_drain:{lot['lot']}", d
                    return
    elif kind == "debt_payment":
        if not debt_ok:
            return
        paid = collections.Counter()
        for p in rec["payments"]:
            paid[p["order"]] += p["amount"]
        cand = [o for o in rec["orders"] if o["payment"] == "Ghi nợ" and o["status"] != "cancel" and WB._d(o["at"]) < e]
        rng.shuffle(cand)
        for o in cand[:3]:
            d = copy.deepcopy(rec)
            day = max(s, WB._d(o["at"]) + dt.timedelta(days=1))
            if day > e:
                continue
            d["payments"].append({"payment": f"PT{90000 + len(d['payments'])}", "order": o["order"], "at": at(day, 16, 5).isoformat(sep=" "), "amount": 1000})
            yield f"debt_payment@{day}:{o['order']}", d
    elif kind == "staff_sale":
        for st in (rec.get("staff") or [])[:3]:
            d = copy.deepcopy(rec)
            o = WB._new_order(d, rng, at(days[0]), "done", None, its[0], 1, price=9_000_000)
            o["staff"] = st
            yield f"staff_sale@{days[0]}:{st}", d
    elif kind == "cancel_item_day":
        it = its[0]
        for day in days[:1]:
            d = copy.deepcopy(rec)
            orders = {ln["order"] for ln in d["lines"] if ln["item"] == it}
            hit = 0
            for o in d["orders"]:
                if o["order"] in orders and WB._d(o["at"]) == day and o["status"] in ("done", "transit"):
                    o["status"], o["status_text"] = "cancel", rng.choice(d["status_text"]["cancel"])
                    if d["opts"].get("note_status"):  # sổ ghi trạng thái ở cột ghi chú: đổi cả ghi chú, như world.m_status
                        o["note"] = rng.choice([t for t in WB.NOTE_TEXT["cancel"] if t] or [""])
                    d["payments"] = [p for p in d["payments"] if p["order"] != o["order"]]
                    hit += 1
            if hit:
                yield f"cancel_item_day@{day}:{it}", d


def find_probe(rec, spec, base_gold, max_tries=45):
    rng = random.Random(f"{rec['seed']}:probe:{json.dumps(spec, sort_keys=True)}")
    kinds = list(dict.fromkeys(ORDER.get(spec["slot"], []) + GENERIC))
    tries = 0
    def safe(kind):
        try:
            yield from variants(rec, spec, kind, rng)
        except (KeyError, ValueError, IndexError, StopIteration):  # biến đổi không áp được cho shop này
            return
    for kind in kinds:
        for name, d in safe(kind):
            tries += 1
            try:
                g = WB.gold(WB._finish(d, "m_probe"), spec)
            except (WB.Degenerate, ValueError, KeyError, StopIteration, ZeroDivisionError):
                g = None
            if g is not None and {k: v for k, v in g.items() if k != "alias"} != {k: v for k, v in base_gold.items() if k != "alias"}:
                return name, d, g, tries
            if tries >= max_tries:
                return None, None, None, tries
    return None, None, None, tries


def probe_shop(job):
    bid, qs, pkg, tmp = job
    t0 = time.time()
    info, rec = SH.load_shop(WORK, bid)
    plan = info["plan"]
    out = []
    for q, cands in qs:
        spec = q["spec"]
        name, mrec, mg, tries = (None, None, None, 0) if spec["slot"] == "unanswerable" else find_probe(rec, spec, q["gold"])
        if name is None:
            out += [{"candidate_id": MR.candidate_id(c), "m_probe": {"status": "no_change_found", "tries": tries}} for c in cands]
            continue
        root = tmp / bid / f"q{len(out)}"
        folder = root / "data" / bid
        (root / "data").mkdir(parents=True)
        shutil.copy(WORK / "shops_root" / "calendar.csv", root / "calendar.csv")
        try:
            written = RD.write_shop(mrec, plan["family"], plan["style"], folder, SH.title_of(plan, mrec))
            db = MR.build_db(root, bid, MR.mutant_layer(root, bid, written), root / "m.duckdb")
        except Exception as e:  # noqa: BLE001
            out += [{"candidate_id": MR.candidate_id(c), "m_probe": {"status": "unavailable", "perturbation": name, "detail": str(e)[:200]}} for c in cands]
            shutil.rmtree(root, ignore_errors=True)
            continue
        al = PK.alias_of(mrec, mg) if mg["type"] != "unanswerable" else None
        mgold = {**mg, **({"alias": al} if al else {})}
        for c in cands:
            kind, sql = MR.scorer.parse_completion(c["completion"])
            st, det = MR.judge(db, kind, sql, mgold) if sql and kind else ("fail", "no TYPE/SQL")
            out.append({"candidate_id": MR.candidate_id(c), "m_probe": {"status": st, "gold_changed": True, "perturbation": name, "tries": tries,
                                                                          **({"detail": det} if det else {})}})
        db.close()
        shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(tmp / bid, ignore_errors=True)
    return {"bid": bid, "rows": out, "secs": round(time.time() - t0, 1)}


def main(a):
    snap = a.snapshot.resolve()
    rep = {x["candidate_id"]: x for x in map(json.loads, (OUT / "mutation_report.jsonl").read_text(encoding="utf-8").splitlines())}
    assert all(x["source_hash"] == MR.E.source_hash(snap) for x in list(rep.values())[:5]), "report ran on another snapshot"
    cands = MR.load_candidates(a.pkg)
    sel = {x["id"]: x for x in map(json.loads, (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines()) if x.get("id")}
    byq = collections.defaultdict(list)
    for c in cands:
        byq[c["id"]].append(c)
    todo = [qid for qid, cs in byq.items() if any(rep[MR.candidate_id(c)]["n_changed_ok"] == 0 and rep[MR.candidate_id(c)]["pass"] for c in cs)]
    todo = todo[: a.limit] if a.limit else todo
    by_shop = collections.defaultdict(list)
    for qid in todo:
        by_shop[sel[qid]["business_id"]].append((sel[qid], [c for c in byq[qid] if rep[MR.candidate_id(c)]["pass"]]))
    res = {}
    with cf.ProcessPoolExecutor(a.procs) as ex:
        for i, r in enumerate(ex.map(probe_shop, [(b, qs, a.pkg, a.tmp) for b, qs in sorted(by_shop.items())]), 1):
            for x in r["rows"]:
                res[x["candidate_id"]] = x["m_probe"]
            if i % 20 == 0:
                print(f"{i}/{len(by_shop)} shops", flush=True)
    out = []
    for cid, x in sorted(rep.items()):
        x = json.loads(json.dumps(x))
        p = res.get(cid)
        if p:
            x["mutants"]["m_probe"] = p
            if p["status"] in ("ok", "fail", "error"):
                x["n_checked"] += 1
                x["n_changed"] += 1
                x["n_changed_ok"] += p["status"] == "ok"
                if p["status"] != "ok" and x["first_failure"] is None:
                    x["first_failure"] = "m_probe"
                x["pass"] = x["pass"] and p["status"] == "ok"
        out.append(x)
    path = OUT / "mutation_report_probe.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    st = collections.Counter(v["status"] for v in res.values())
    pert = collections.Counter(v.get("perturbation", "").split("@")[0].split(":")[0] for v in res.values() if v["status"] in ("ok", "fail", "error"))
    summ = {"questions_probed": len(todo), "candidates_probed": len(res), "status": dict(st), "perturbations": dict(pert),
            "before": {"pass": sum(x["pass"] for x in rep.values()), "pass_with_changed_gold": sum(x["pass"] and x["n_changed_ok"] > 0 for x in rep.values())},
            "after": {"pass": sum(x["pass"] for x in out), "pass_with_changed_gold": sum(x["pass"] and x["n_changed_ok"] > 0 for x in out)},
            "failed_by_slot": dict(collections.Counter(sel[x["question_id"]]["spec"]["slot"] for x in out if x["first_failure"] == "m_probe")),
            "report_file": str(path), "report_sha256": MR.F.file_hash(path)}
    (OUT / "probe_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(A)
