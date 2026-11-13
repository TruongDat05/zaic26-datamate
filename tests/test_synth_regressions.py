"""Regression cho các lỗi hệ thống tìm thấy khi audit corpus synthetic: tỷ lệ hủy đếm phiếu trả, công nợ lọc từ ngày cố định,
ghi cứng mã/ngày, code nháp, chứng nhận cũ còn hiệu lực sau khi sửa code, và oracle + đối chứng của slot shop phức tạp."""
import collections
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
import synth_corpus as C  # noqa: E402
import synth_world as W  # noqa: E402

TODAY = "2025-05-16"


def spec(slot, period, **params):
    return {"slot": slot, "period": period, "params": params, "as_of": TODAY}


@pytest.fixture(scope="module")
def cx():
    """Shop phức tạp: lịch sử giá, giá vốn, tồn đầu kỳ, nhân viên/ca, công nợ."""
    return W.simulate("tap_hoa", 9191, "2025-02-01", TODAY, after_days=2, complex_=True)


@pytest.fixture(scope="module")
def rx():
    return W.simulate("nha_thuoc", 9292, "2025-02-01", TODAY, after_days=0, complex_=True)


def test_cancel_rate_return_slips_caught_by_target_mutation(tmp_path):
    """Lỗi [17]: code đếm cả phiếu trả ('Trả') vào mẫu số tỷ lệ hủy, khớp do may khi gold 0%. Đối chứng nhắm đặc tả phải làm nó trượt."""
    from synth_tools import Kit
    rec = W.simulate("dien_gia_dung", 4243, "2025-02-01", TODAY, after_days=0)
    sp = spec("cancel_rate", {"kind": "yesterday"})
    mut, _ = W.target_mutation(rec, sp)
    assert W.gold(mut, sp) != W.gold(rec, sp)
    fx = W.write_fixture(tmp_path / "nl", "syn_nl", mut, "neg_lines")
    kit = Kit(data=fx["root"], cache=Path(fx["root"]) / "cache")
    head = "t = T['transactions']\np = dm.yesterday(TODAY)\n"
    tail = ("st = x['status'].map(dm.norm_status)\ntotal = x['bill'].nunique()\ncanc = x.loc[st == 'huy', 'bill'].nunique()\n"
            "ANSWER = dm.percent(canc / total * 100)\n")
    good = head + "x = t[p.mask(t['time']) & (t['type'] == 'Bán')]\n" + tail
    bug = head + "x = t[p.mask(t['time'])]\n" + tail
    g = W.gold(mut, sp)
    assert C.check_answer(kit.execute("syn_nl", good, TODAY)["answer"], g)
    assert not C.check_answer(kit.execute("syn_nl", bug, TODAY)["answer"], g)


def test_debt_fixed_start_date_is_caught(cx):
    """Công nợ 'tính đến': code đặt mốc đầu cố định (ngày đầu dữ liệu) — bị kiểm tĩnh và đối chứng đơn nợ trước ngày đầu dữ liệu bắt."""
    for slot, params in (("debt_total", {}), ("top_debtor", {})):
        sp = spec(slot, {"kind": "until", "d": "2025-05-10"}, **params)
        out = W.target_mutation(cx, sp)
        assert out is not None, slot
        mut = out[0]
        start = min(W._d(o["at"]) for o in cx["orders"])
        assert any(W._d(o["at"]) < start for o in mut["orders"]), "đơn nợ phải nằm trước ngày đầu dữ liệu"
        clipped = dict(mut, orders=[o for o in mut["orders"] if W._d(o["at"]) >= start])
        assert W.gold(clipped, sp) != W.gold(mut, sp), slot  # code lọc từ ngày đầu dữ liệu cố định cho đáp án khác
    sp = spec("debt_total", {"kind": "until", "d": "2025-05-10"})
    sp["gold"] = W.gold(cx, sp)
    q = "Tính đến hết ngày 10/05/2025 khách còn nợ bao nhiêu?"
    fixed = "o = T['orders']\nm = (o['ordered_at'] >= '2025-02-01') & (o['ordered_at'] < '2025-05-11')\nANSWER = dm.money(o.loc[m, 'total'].sum())"
    assert "fixed_date_until" in C.code_issues(fixed, sp, q)
    ok = "o = T['orders']\nm = o['ordered_at'] < '2025-05-11'\nANSWER = dm.money(o.loc[m, 'total'].sum())"
    assert "fixed_date_until" not in C.code_issues(ok, sp, q)


def test_hardcoded_key_and_draft_comment():
    sp = spec("revenue_item", {"kind": "month", "y": 2025, "m": 4}, item="DG006")
    sp["gold"] = {"type": "number", "answer": 1234567}
    q = "Tháng 4/2025 nồi cơm điện bán được bao nhiêu tiền?"
    code = "x = T['sales']\nm = dm.month(2025, 4).mask(x['sold_at']) & (x['sku'] == 'DG006')\nANSWER = dm.money(x.loc[m, 'amount'].sum())"
    assert "hardcoded_key" in C.code_issues(code, sp, q, codes={"dg006"}, doc="")
    assert "hardcoded_key" not in C.code_issues(code, sp, q + " (mã DG006)", codes={"dg006"}, doc="")
    named = "x = T['sales']\nit = dm.find_item(T, 'nồi cơm điện')\nm = dm.month(2025, 4).mask(x['sold_at']) & (x['sku'] == it)\nANSWER = dm.money(x.loc[m, 'amount'].sum())"
    assert "hardcoded_key" not in C.code_issues(named, sp, q, codes={"dg006"}, doc="")
    draft = "x = T['sales']\n# Sai, cần tổng tiền\nANSWER = dm.money(x.loc[dm.month(2025, 4).mask(x['sold_at']), 'amount'].sum())"
    assert "draft_comment" in C.code_issues(draft, sp, q)


def test_certification_expires_when_code_or_standard_changes():
    s = {"coder": {"code": "ANSWER = dm.money(1)"}, "source_hash": "abc"}
    ver = C.cert_version()
    s["certified"] = {"ok": True, "standard": C.CERT_STANDARD, "version": ver, "code_sha256": C.digest(s["coder"]["code"]), "source_hash": "abc"}
    assert C.cert_current(s, ver)
    assert not C.cert_current(dict(s, coder={"code": "ANSWER = dm.money(2)"}), ver)  # code sửa sau chứng nhận
    assert not C.cert_current(dict(s, source_hash="xyz"), ver)
    assert not C.cert_current(dict(s, certified=dict(s["certified"], standard="corpus_v2+m_target")), ver)
    assert not C.cert_current(dict(s, certified=dict(s["certified"], version="old")), ver)
    assert not C.cert_current(dict(s, certified=dict(s["certified"], ok=False)), ver)
    mv = {"ro_rang": True, "dung_nghia": True, "giu_nghia": True, "code_ok": False, "pass": False, "code_sha256": C.digest("ANSWER = dm.money(1)")}
    assert C.manual_verdict(s, mv) == "fail"
    assert C.manual_verdict(dict(s, coder={"code": "ANSWER = dm.money(2)"}), mv) == "stale"  # code đã sửa: phải audit lại
    assert C.manual_verdict(s, dict(mv, dung_nghia=False, code_sha256="other")) == "fail"  # lỗi nghĩa câu: sửa code không cứu
    assert C.manual_verdict(s, None) == "none"


def test_complex_shop_round_trips_every_family(tmp_path, cx):
    for fam in W.FAMILIES:
        fx = W.write_fixture(tmp_path / fam, "syn_" + fam, cx, fam)
        assert W.integrity(cx, fam, fx["biz"].tables) == [], fam


def test_complex_oracle_invariants(cx):
    """Kiểm độc lập (bất biến nghiệp vụ) cho slot mới: ca và nhân viên chia hết doanh thu, tồn kho = đầu kỳ + nhập − bán, giá theo ngày."""
    april = {"kind": "month", "y": 2025, "m": 4}
    rev = W.gold(cx, spec("revenue", april))["answer"]
    assert sum(W.gold(cx, spec("shift_rev", april, shift=x["shift"]))["answer"] for x in cx["shifts"]) == rev
    assert sum(W.gold(cx, spec("staff_rev", april, staff=n))["answer"] for n in cx["staff"]) == rev
    fac = {(u["item"], u["unit"]): u["factor"] for u in cx["units"]}
    for it in cx["items"][:4]:
        c = it["item"]
        want = next(x["qty"] for x in cx["opening_stock"] if x["item"] == c)
        want += sum(x["qty"] * fac[(c, x["unit"])] for x in cx["purchases"] if x["item"] == c and W._d(x["at"]) <= W._d("2025-04-30"))
        orders = {o["order"]: o for o in cx["orders"]}
        want -= sum(ln["qty"] * fac[(c, ln["unit"])] for ln in cx["lines"] if ln["item"] == c and orders[ln["order"]]["status"] == "done"
                    and W._d(orders[ln["order"]]["at"]) <= W._d("2025-04-30"))
        assert W.gold(cx, spec("stock_qty", {"kind": "until", "d": "2025-04-30"}, item=c))["answer"] == want >= 0
    ch = min(cx["price_history"], key=lambda x: (x["item"], x["from"]))  # lần đổi giá đầu tiên của một mặt hàng
    catalog = next(it["price"] for it in cx["items"] if it["item"] == ch["item"])
    before = (W._d(ch["from"]) - W.dt.timedelta(days=1)).isoformat()
    assert W.gold(cx, spec("price_on_day", {"kind": "day", "d": before}, item=ch["item"]))["answer"] == catalog  # chưa đổi: giá danh mục
    assert W.gold(cx, spec("price_on_day", {"kind": "day", "d": ch["from"]}, item=ch["item"]))["answer"] == ch["price"]  # áp dụng từ đầu ngày
    profit = W.gold(cx, spec("profit_gross", april))["answer"]
    assert 0 < profit < rev


def test_new_slots_have_specs_and_gold_changing_target_mutations(cx, rx):
    shop = {"today": TODAY, "start": "2025-02-01", "after_days": 0, "family": "joined", "name": "syn_test_cx"}
    rng = random.Random(5)
    got = collections.defaultdict(lambda: [0, 0])
    for slot in W.COMPLEX_SLOTS:
        rec = rx if slot.startswith("rx_") else cx
        for kind in C.SLOT_KINDS.get(slot, C.DEFAULT_KINDS)[:3]:
            for _ in range(2):
                sp = C.sample_spec(rec, shop, slot, kind, rng)
                if sp is None:
                    continue
                got[slot][0] += 1
                out = W.target_mutation(rec, sp)
                if out is not None:
                    assert W.gold(out[0], sp) != sp["gold"], slot
                    got[slot][1] += 1
    missing = [s for s in W.COMPLEX_SLOTS if got[s][0] == 0]
    assert not missing, missing
    no_mut = [s for s in W.COMPLEX_SLOTS if got[s][1] == 0]
    assert not no_mut, no_mut


def test_quality_issues_from_phase2_audit():
    """Lỗi hệ thống audit giai đoạn 2 (10/11): chú thích rác/suy đoán, code thừa, gán đè, if/else trùng, năm ghi cứng ở kỳ tương đối."""
    sp = spec("rev_change_pct", {"kind": "this_month"})
    sp["gold"] = {"type": "percent", "answer": 5.1}
    q = "Doanh thu tháng này tăng bao nhiêu phần trăm so với tháng trước?"
    base = "p = dm.this_month(TODAY)\nq = dm.last_month(TODAY)\nx = T['sales']\na = x.loc[p.mask(x['sold_at']), 'amount'].sum()\nb = x.loc[q.mask(x['sold_at']), 'amount'].sum()\n"
    clean = base + "ANSWER = dm.pct_change(a, b)"
    assert not C.quality_issues(clean, sp)
    assert "dead_code" in C.quality_issues(base + "pct = (a - b) / b * 100\nANSWER = dm.pct_change(a, b)", sp)
    assert "dead_code" in C.quality_issues("m = 1\nm = 2\nANSWER = dm.count(m)", sp)
    assert "dead_code" in C.quality_issues("if TODAY.day > 1:\n    k = 1\nelse:\n    k = 1\nANSWER = dm.count(k)", sp)
    assert "junk_comment" in C.quality_issues("# ponytail: assume qty > 0\n" + clean, sp)
    assert "speculative_comment" in C.quality_issues("# Giả sử mọi dòng đều là đơn vị gốc\n" + clean, sp)
    assert "speculative_comment" not in C.quality_issues("# Chi phí thuê kho không có trong dữ liệu\nANSWER = dm.unanswerable('không ghi')", sp)
    assert "year_literal_rel" in C.quality_issues(base + "ANSWER = dm.month_answer(dt.date(2025, 5, 1))", sp)
    assert "year_literal_rel" not in C.quality_issues("# hôm nay (TODAY = 2025-05-16)\n" + clean, sp)
    assert "dead_code" in C.code_issues(base + "pct = (a - b) / b * 100\nANSWER = dm.pct_change(a, b)", sp, q)
