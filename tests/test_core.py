import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from datamate import clean, dm, formatter, timeutil  # noqa: E402
from datamate.llm import extract_code  # noqa: E402
import score  # noqa: E402

D = dt.date


# ---------------------------------------------------------------- clean

@pytest.mark.parametrize("raw,want", [
    ("38,000", 38000), ("45,000.0000", 45000), ("1.234.567", 1234567), ("12,5", 12.5), ("38.000đ", 38000),
    ("1.5", 1.5), (1200, 1200), ("-12,45", -12.45), ("0.125", 0.125), ("(5,000)", -5000), (" 7 000 ", 7000),
])
def test_parse_number(raw, want):
    assert clean.parse_number(raw) == pytest.approx(want)


def test_parse_number_bad():
    assert pd.isna(clean.parse_number("1 bao 25kg"))
    assert pd.isna(clean.parse_number("ORD000001"))


@pytest.mark.parametrize("raw,want", [
    ("01/03/2025", "2025-03-01"), ("1 thg 3 2025", "2025-03-01"), ("2025-03-01", "2025-03-01"),
    ("1/3/2025 14:05", "2025-03-01 14:05"), (45717, "2025-03-01"), ("1 tháng 3, 2025", "2025-03-01"),
])
def test_parse_datetime(raw, want):
    assert clean.parse_datetime(raw) == pd.Timestamp(want)


def test_parse_qty():
    assert clean.parse_qty("5 chai") == (5.0, "chai")
    assert clean.parse_qty("1 bao 25kg") == (1.0, "bao 25kg")
    assert clean.parse_qty(3) == (3.0, "")


@pytest.mark.parametrize("raw,want", [
    ("Đã hoàn thành", "hoan_thanh"), ("xong", "hoan_thanh"), ("Đã hủy", "huy"), ("hủy", "huy"),
    ("Trả hàng", "tra_hang"), ("hàng trả về", "tra_hang"), ("Đang giao", "dang_giao"), ("Đã giao", "hoan_thanh"),
])
def test_norm_status(raw, want):
    assert clean.norm_status(raw) == want


def test_walkin():
    for s in ["kl", "lẻ", "Khách lẻ", "khách vãng lai", None, ""]:
        assert clean.is_walkin(s)
    assert not clean.is_walkin("Nguyễn Văn A")


# ---------------------------------------------------------------- thời gian (rulebook §2)

def test_weeks():
    today = D(2026, 3, 4)  # thứ Tư
    assert timeutil.this_week(today) == timeutil.Period(D(2026, 3, 2), D(2026, 3, 4), "week_to_date")
    lw = timeutil.last_week(today)
    assert (lw.start, lw.end) == (D(2026, 2, 23), D(2026, 3, 1))


def test_this_month_vs_prev():
    today = D(2026, 3, 31)
    assert timeutil.this_month(today).start == D(2026, 3, 1)
    p = timeutil.prev_month_same_days(today)
    assert (p.start, p.end) == (D(2026, 2, 1), D(2026, 2, 28))
    assert timeutil.previous(timeutil.this_month(D(2026, 3, 15))).end == D(2026, 2, 15)


def test_month_no_year():
    assert timeutil.month_no_year(5, D(2026, 3, 4)).start == D(2025, 5, 1)
    assert timeutil.month_no_year(3, D(2026, 3, 4)).start == D(2026, 3, 1)
    assert timeutil.quarter_no_year(2, D(2026, 3, 4)).start == D(2025, 4, 1)


def test_last_n_days_and_mask():
    p = timeutil.last_n_days(7, D(2025, 10, 25))
    assert (p.start, p.end, p.days) == (D(2025, 10, 19), D(2025, 10, 25), 7)
    s = pd.Series(pd.to_datetime(["2025-10-18 23:59", "2025-10-19 00:00", "2025-10-25 23:00", "2025-10-26 00:00"]))
    assert p.mask(s).tolist() == [False, True, True, False]


def test_months_ending():
    ms = timeutil.months_ending(2026, 1, 3)
    assert [m.start for m in ms] == [D(2025, 11, 1), D(2025, 12, 1), D(2026, 1, 1)]


def test_tet():
    cal = timeutil.LunarCalendar(ROOT / "data" / "train" / "calendar.csv")
    t = cal.tet(2026)  # mùng 1 Tết Bính Ngọ 2026 = 17/02/2026
    assert cal.lunar(t.start)[1:3] == (12, 23)
    assert cal.lunar(t.end)[1:3] == (1, 10)
    assert t.start == D(2026, 2, 10) and t.end == D(2026, 2, 26)
    b = cal.before_tet(2026)
    assert b.end == D(2026, 2, 9) and b.days == t.days


def test_shift():
    s = pd.Series(pd.to_datetime(["2025-01-01 06:00", "2025-01-01 13:59", "2025-01-01 14:00", "2025-01-01 22:00"]))
    assert timeutil.shift_mask(s, "sang").tolist() == [True, True, False, False]
    assert timeutil.shift_mask(s, "chieu").tolist() == [False, False, True, False]


# ---------------------------------------------------------------- dm, formatter

def test_rounding():
    assert dm.money(72709.5)["answer"] == 72710
    assert dm.percent(-10.685)["answer"] == -10.69
    assert dm.pct_change(90, 100)["answer"] == -10.0
    assert dm.share(1, 3)["answer"] == 33.33


def test_rank_ties_by_code():
    df = pd.DataFrame({"ma": ["SP2", "SP1", "SP3"], "ten": ["b", "a", "c"], "sl": [5, 5, 9]})
    assert dm.rank(df, by="sl", code="ma")["ma"].tolist() == ["SP3", "SP1", "SP2"]


def test_match():
    names = ["Clarityne 10mg hộp 1 vỉ x 10 viên", "Panadol Extra", "Sting dâu 330ml"]
    assert dm.match_one(names, "clarityne 10mg hop 1 vi x 10 vien") == names[0]
    assert dm.match_one(names, "sting dau") == names[2]


def test_finalize():
    assert formatter.finalize("q", {"type": "number", "answer": 5.0}) == {"id": "q", "type": "number", "answer": 5}
    assert formatter.finalize("q", {"type": "number", "answer": "5"})["type"] == "abstain"
    assert formatter.finalize("q", {"type": "date", "answer": "15/08/2026"})["type"] == "abstain"
    assert formatter.finalize("q", {"type": "list", "answer": []})["type"] == "abstain"
    t = formatter.finalize("q", dm.table(["a", "b"], [["x", 1.0]]))
    assert t["answer"] == {"columns": ["a", "b"], "rows": [["x", 1]]}


def test_extract_code():
    assert extract_code("<think>x</think>\n```python\nANSWER = 1\n```") == "ANSWER = 1\n"
    # câu trả lời bị cắt giữa khối code: không đem văn bản lẫn code đi chạy
    assert extract_code("Bước 1: lọc đơn.\n```python\nANSWER = dm.money(") is None
    assert extract_code("ANSWER = dm.count(3)") == "ANSWER = dm.count(3)"


# ---------------------------------------------------------------- score

def test_score_matches_gold_on_itself():
    gold = score.load_jsonl(ROOT / "data" / "train" / "answers.jsonl")
    res, _ = score.score(gold, gold)
    assert res["PA"] == 1.0


def test_score_rules():
    g = {"type": "string", "answer": "Sting dâu 330ml", "alias": ["SP0042"]}
    assert score.match({"type": "string", "answer": "sting  DÂU 330ml"}, g) == 1
    assert score.match({"type": "string", "answer": "sp0042"}, g) == 1
    assert score.match({"type": "string", "answer": "Sting dầu 330ml"}, g) == -1
    assert score.match({"type": "abstain"}, g) == 0
    assert score.match({"type": "percent", "answer": -12.449}, {"type": "percent", "answer": -12.45}) == 1
    assert score.match({"type": "number", "answer": 0.0}, {"type": "unanswerable", "answer": None}) == -1
    tg = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["A", 1], ["B", 2]]},
          "alias": [["KH1"], ["KH2"]]}
    assert score.match({"type": "table", "answer": {"columns": ["x", "y"], "rows": [["kh2", 2], ["A", 1]]}}, tg) == 1


def test_variants_see_different_examples():
    from datamate.agent import Agent, DEFAULT_VARIANTS
    from datamate.fewshot import FewShot

    fs = FewShot(ROOT / "data" / "train" / "questions.jsonl", ROOT / "data" / "train" / "answers.jsonl")
    agent = Agent(None, {"fewshot_k": 4}, "RB", ROOT / "data" / "train" / "calendar.csv", fs)

    class B:
        business_id, readme = "x", ""

        def profile(self):
            return ""

    q = {"id": "q", "business_id": "biz_118", "question": "dthu tháng 6/2025 bao nhiêu", "as_of_date": "2026-09-10"}
    users = [agent._messages(q, B(), v)[1]["content"] for v in DEFAULT_VARIANTS]
    assert len(set(users)) == len(users)
    assert all("biz_118" not in u for u in users)


@pytest.mark.parametrize("raw,want", [
    ("45k", 45000), ("100K", 100000), ("1tr270", 1270000), ("2tr760", 2760000), ("1tr5", 1500000),
    ("1,5tr", 1500000), ("3 triệu", 3000000), ("(20k)", -20000),
])
def test_parse_money_shorthand(raw, want):
    """Tiền ghi tắt trong sổ tay (EDA: biz_105, biz_117, biz_124…)."""
    assert clean.parse_number(raw) == pytest.approx(want)
