"""Bộ sinh đa dạng (synth_world, synth_corpus, synth_rubric): oracle độc lập, 3 lỗi đã gặp ở pilot, đối chứng, kiểm tĩnh, split. Không gọi API."""
import collections
import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
import synth_corpus as C  # noqa: E402
import synth_world as W  # noqa: E402

TODAY = "2025-05-16"


@pytest.fixture(scope="module")
def rec():
    return W.simulate("tiem_banh", 777, "2025-02-01", TODAY, after_days=2)


def spec(slot, period, **params):
    return {"slot": slot, "period": period, "params": params, "as_of": TODAY}


def test_refund_counts_by_refund_date_across_periods(rec):
    april = {"kind": "month", "y": 2025, "m": 4}
    s, e = W.resolve(april, TODAY)
    orders = {o["order"]: o for o in rec["orders"]}
    cross_in = [r for r in rec["refunds"] if s <= W._d(r["at"]) <= e and not s <= W._d(orders[r["order"]]["at"]) <= e]
    cross_out = [r for r in rec["refunds"] if s <= W._d(orders[r["order"]]["at"]) <= e and W._d(r["at"]) > e]
    assert cross_in and cross_out, "mô phỏng phải có hoàn chéo kỳ cả hai chiều"
    gross = W.gold(rec, spec("revenue", april))["answer"]
    net = W.gold(rec, spec("revenue_net", april))["answer"]
    by_refund_date = sum(r["amount"] for r in rec["refunds"] if s <= W._d(r["at"]) <= e)
    assert net == gross - by_refund_date
    by_order_date = sum(r["amount"] for r in rec["refunds"] if s <= W._d(orders[r["order"]]["at"]) <= e)
    assert by_order_date != by_refund_date  # lỗi pilot (lọc hoàn theo đơn trong kỳ) cho đáp án khác


def test_today_is_inclusive_and_after_today_ignored(rec):
    p = {"kind": "this_month"}
    late = [o for o in rec["orders"] if o["at"].startswith(TODAY + " 23:5") and o["status"] == "done"]
    assert late, "phải có đơn khuya đúng TODAY"
    g = W.gold(rec, spec("n_orders", p))["answer"]
    s, e = W.resolve(p, TODAY)
    midnight_bug = len({o["order"] for o in rec["orders"] if o["status"] == "done" and s <= W._d(o["at"]) and W._ts(o["at"]) <= dt.datetime.combine(e, dt.time())})
    assert g > midnight_bug  # `<= pd.Timestamp(TODAY)` bỏ mất đơn trong ngày TODAY
    assert W._d(rec["data_end"]) > W._d(TODAY)
    after = W.mutate(rec, "m_after")
    assert W.gold(after, spec("revenue", p)) == W.gold(rec, spec("revenue", p))


def test_period_resolution_matches_runtime_dm():
    from datamate import dm
    t = dt.date(2025, 5, 16)
    pairs = [({"kind": "this_week"}, dm.this_week(t)), ({"kind": "last_week"}, dm.last_week(t)), ({"kind": "last_n", "n": 7}, dm.last_n_days(7, t)),
             ({"kind": "this_month"}, dm.this_month(t)), ({"kind": "last_month"}, dm.last_month(t)), ({"kind": "prev_same_days"}, dm.prev_month_same_days(t)),
             ({"kind": "quarter", "y": 2025, "q": 1}, dm.quarter(2025, 1)), ({"kind": "this_year"}, dm.year_to_date(t))]
    for p, d in pairs:
        assert W.resolve(p, t) == (d.start, d.end), p


@pytest.mark.parametrize("family", W.FAMILIES)
def test_every_family_round_trips_through_loader(tmp_path, family):
    rec = W.simulate("nha_thuoc", 31, "2025-03-01", "2025-05-10", after_days=0)
    fx = W.write_fixture(tmp_path / family, "syn_" + family, rec, family)
    assert W.integrity(rec, family, fx["biz"].tables) == []


def test_runtime_catches_three_pilot_bugs(tmp_path):
    """Code mắc lỗi pilot (hoàn tiền âm bị trừ, `<= Timestamp(TODAY)`, gán .loc index trùng) phải trượt; code đúng phải qua."""
    from synth_tools import Kit
    rec = W.simulate("my_pham", 4242, "2025-02-01", TODAY, after_days=0)
    fx = W.write_fixture(tmp_path / "ev", "syn_ev", rec, "events")
    kit = Kit(data=fx["root"], cache=Path(fx["root"]) / "cache")
    april = spec("revenue_net", {"kind": "month", "y": 2025, "m": 4})
    gold = W.gold(rec, april)
    head = ("h = T['headers']; e = T['events']\nok = h['state'].map(dm.norm_status) == 'hoan_thanh'\np = dm.month(2025, 4)\n"
            "orders = set(h.loc[ok & p.mask(h['ordered_at']), 'order'])\nsale = e[(e['event'] == 'sale') & e['order'].isin(orders)]['amount'].sum()\n")
    good = head + "ref = e[(e['event'] == 'refund') & p.mask(e['event_at'])]['amount'].sum()\nANSWER = dm.money(sale + ref)\n"
    sign_bug = head + "ref = e[(e['event'] == 'refund') & p.mask(e['event_at'])]['amount'].sum()\nANSWER = dm.money(sale - ref)\n"
    scope_bug = head + "ref = e[(e['event'] == 'refund') & e['order'].isin(orders)]['amount'].sum()\nANSWER = dm.money(sale + ref)\n"
    assert C.check_answer(kit.execute("syn_ev", good, TODAY)["answer"], gold)
    assert not C.check_answer(kit.execute("syn_ev", sign_bug, TODAY)["answer"], gold)
    assert not C.check_answer(kit.execute("syn_ev", scope_bug, TODAY)["answer"], gold)
    avg = spec("avg_order", {"kind": "this_month"})
    g2 = W.gold(rec, avg)
    base = "h = T['headers']; e = T['events']\nok = h['state'].map(dm.norm_status) == 'hoan_thanh'\n"
    right = base + "d = h[ok & dm.this_month(TODAY).mask(h['ordered_at'])]\nANSWER = dm.money(e[(e['event']=='sale') & e['order'].isin(d['order'])]['amount'].sum() / len(d))\n"
    today_bug = base + ("start = pd.Timestamp(TODAY.replace(day=1)); end = pd.Timestamp(TODAY)\nd = h[ok & (h['ordered_at'] >= start) & (h['ordered_at'] <= end)]\n"
                        "ANSWER = dm.money(e[(e['event']=='sale') & e['order'].isin(d['order'])]['amount'].sum() / len(d))\n")
    assert C.check_answer(kit.execute("syn_ev", right, TODAY)["answer"], g2)
    assert not C.check_answer(kit.execute("syn_ev", today_bug, TODAY)["answer"], g2)
    index_bug = base + ("x = e[e['event']=='sale'].copy()\nx['d'] = h.set_index('order').loc[x['order'], 'ordered_at'].dt.date\n"
                        "ANSWER = dm.date(x.groupby('d')['amount'].sum().idxmax())\n")
    out = kit.execute("syn_ev", index_bug, TODAY)
    assert not out["ok"] and "reindex" in out["error"]


def test_static_checks_hardcoded_answer_and_today(rec):
    sp = dict(spec("revenue", {"kind": "this_month"}))
    sp["gold"] = W.gold(rec, sp)
    v = sp["gold"]["answer"]
    assert C.code_issues(f"x = T['sales']\nANSWER = dm.money({v})", sp, "doanh thu thang nay") == ["constant_answer"]
    disguised = f"x = T['sales']\ns = x['line_total'].sum() * 0 + {v}\nANSWER = dm.money(s)"
    assert "hardcoded_answer" in C.code_issues(disguised, sp, "doanh thu thang nay")
    lit = "x = T['sales']\np = dm.date_range(dt.date(2025, 5, 1), dt.date(2025, 5, 16))\nANSWER = dm.money(x.loc[p.mask(x['sold_at']), 'line_total'].sum())"
    assert "hardcoded_today" in C.code_issues(lit, sp, "doanh thu thang nay")
    ok = "x = T['sales']\nANSWER = dm.money(x.loc[dm.this_month(TODAY).mask(x['sold_at']), 'line_total'].sum())"
    assert C.code_issues(ok, sp, "doanh thu thang nay") == []


def test_mutations_change_gold_where_intended(rec):
    april = {"kind": "month", "y": 2025, "m": 4}
    assert W.gold(W.mutate(rec, "m_refund"), spec("revenue_net", april)) != W.gold(rec, spec("revenue_net", april))
    assert W.gold(W.mutate(rec, "m_alias"), spec("cancel_rate", april)) == W.gold(rec, spec("cancel_rate", april))
    item = next(u["item"] for u in rec["units"] if u["factor"] > 1)
    assert W.gold(W.mutate(rec, "m_units"), spec("qty_base", april, item=item)) != W.gold(rec, spec("qty_base", april, item=item))
    walk = W.mutate(rec, "m_walkin")
    assert W.gold(walk, spec("top_customer", april)) == W.gold(rec, spec("top_customer", april))
    assert W.gold(walk, spec("revenue", april)) != W.gold(rec, spec("revenue", april))


def test_period_drift_is_detected(rec):
    s = dict(spec("revenue", {"kind": "month", "y": 2025, "m": 4}))
    assert C.period_ok("Doanh thu tháng 4/2025 là bao nhiêu?", s)
    assert not C.period_ok("Doanh thu tháng 3/2025 là bao nhiêu?", s)
    r = dict(spec("revenue", {"kind": "last_week"}))
    assert C.period_ok("Tuần trước tiệm bán được bao nhiêu tiền?", r)
    assert not C.period_ok("Tháng trước tiệm bán được bao nhiêu tiền?", r)


def test_splits_are_disjoint_by_shop_seed_and_family():
    shops = C.shop_plan()
    by = collections.defaultdict(set)
    for s in shops:
        by[s["split"]].add((s["name"], s["seed"], s["family"]))
    fams = {k: {x[2] for x in v} for k, v in by.items()}
    assert fams["train"].isdisjoint(fams["val"]) and fams["train"].isdisjoint(fams["final"]) and fams["val"].isdisjoint(fams["final"])
    seeds = [s["seed"] for s in shops]
    assert len(seeds) == len(set(seeds))
    assert {s["industry"] for s in shops if s["split"] == "final"} & set(W.FINAL_ONLY) == set(W.FINAL_ONLY)
    assert not {s["industry"] for s in shops if s["split"] in ("train", "val")} & set(W.FINAL_ONLY)
    assert {s["family"] for s in shops if s["split"] == "final_b"} == {"monthly"}
    assert len({s["name"] for s in shops}) == len(shops)


def test_rubric_per_sample_flags():
    import synth_rubric as R
    ok = R.per_sample("Doanh thu tháng 4/2025 là bao nhiêu?", {"type": "number", "answer": 1200000}, set(), (5, 36))
    assert all(ok.values())
    bad = R.per_sample("Doanh thu tháng 4/2025 là 1200000 đúng không, không tính khách lẻ?", {"type": "number", "answer": 1200000}, set(), (5, 36))
    assert not bad["no_leak"] and not bad["no_hint"] and not bad["type_consistent"]


def test_target_mutation_changes_gold_for_luck_prone_slots(rec):
    """Đáp án 0, có/không, xếp hạng: đối chứng nhắm đặc tả phải làm đổi đáp án (bắt code đúng do may)."""
    cases = [spec("cancel_rate", {"kind": "yesterday"}), spec("exists_item_day", {"kind": "day", "d": "2025-04-03"}, item=rec["items"][0]["item"]),
             spec("best_day", {"kind": "month", "y": 2025, "m": 4}), spec("top_customer", {"kind": "month", "y": 2025, "m": 4}),
             spec("unsold_items", {"kind": "day", "d": "2025-04-03"}), spec("revenue_net", {"kind": "last_month"})]
    for sp in cases:
        out = W.target_mutation(rec, sp)
        assert out is not None, sp["slot"]
        assert W.gold(out[0], sp) != W.gold(rec, sp), sp["slot"]
    assert W.target_mutation(rec, spec("unanswerable", {"kind": "last_month"}, missing="x")) is None
