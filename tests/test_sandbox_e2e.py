"""Kiểm tra loader + dm + sandbox bằng lời giải viết tay (như LLM sẽ viết) cho vài câu train biz_118."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from datamate import formatter, loader  # noqa: E402
from datamate.run import cache_path  # noqa: E402
from datamate.sandbox import run_code  # noqa: E402
import score  # noqa: E402

DATA = ROOT / "data" / "train"
CACHE = ROOT / "outputs" / "cache"

PRE = '''
s = T['So_chi_tiet_ban_hang'].copy()
s['st'] = s['Trạng thái'].map(dm.norm_status)
bad = set(s.loc[s.st.isin(['huy', 'tra_hang']), 'Mã đơn hàng'])
ok = s[~s['Mã đơn hàng'].isin(bad)]
'''

CASES = {
    # dthu tháng 6/2025
    "q_158b312e8f5b": PRE + "ANSWER = dm.money(ok[dm.month(2025, 6).mask(ok['Ngày'])]['Thành tiền'].sum())",
    # tỉ lệ đơn huỷ tháng 3/2026 (mọi đơn đặt trong kỳ)
    "q_4293a4130594": PRE + """
m = s[dm.month(2026, 3).mask(s['Ngày'])]
o = m.groupby('Mã đơn hàng')['st'].first()
ANSWER = dm.share((o == 'huy').sum(), len(o))""",
    # sp bán chạy nhất tuần rồi, theo số lượng quy về đơn vị cơ bản
    "q_06de33fa21cc": PRE + """
m = ok[dm.last_week(TODAY).mask(ok['Ngày'])]
g = m.groupby('Tên hàng hóa', as_index=False)['Số lượng quy đổi'].sum()
sp = T['Danh_sach_hang_hoa'][['Tên hàng hóa', 'Mã hàng hóa']]
g = g.merge(sp, on='Tên hàng hóa', how='left')
ANSWER = dm.string(dm.rank(g, by='Số lượng quy đổi', code='Mã hàng hóa').iloc[0]['Tên hàng hóa'])""",
    # từ đầu tháng này đến hết hôm nay so với cùng kỳ tháng trước
    "q_09581d4bcf0b": PRE + """
p = dm.this_month(TODAY)
cur = ok[p.mask(ok['Ngày'])]['Thành tiền'].sum()
prev = ok[dm.previous(p).mask(ok['Ngày'])]['Thành tiền'].sum()
ANSWER = dm.pct_change(cur, prev)""",
    # 7 ngày gần nhất bán mấy m3 (mặt hàng có đơn vị cơ bản m3)
    "q_2646a315a83f": PRE + """
sp = T['Danh_sach_hang_hoa']
m3 = set(sp.loc[sp['Đơn vị tính'] == 'm3', 'Tên hàng hóa'])
m = ok[dm.last_n_days(7, TODAY).mask(ok['Ngày']) & ok['Tên hàng hóa'].isin(m3)]
ANSWER = dm.count(m['Số lượng quy đổi'].sum())""",
    # top 5 khách chi tiêu nhiều nhất tháng 8/2025
    "q_6a561b61690c": PRE + """
m = ok[dm.month(2025, 8).mask(ok['Ngày']) & ~ok['Khách hàng'].map(dm.is_walkin)]
g = m.groupby('Khách hàng', as_index=False)['Thành tiền'].sum()
ANSWER = dm.list_(dm.rank(g, by='Thành tiền', code='Khách hàng', k=5)['Khách hàng'])""",
    # doanh thu trung bình một đơn tháng 1/2026
    "q_409cf8f109ac": PRE + """
m = ok[dm.month(2026, 1).mask(ok['Ngày'])]
ANSWER = dm.money(m['Thành tiền'].sum() / m['Mã đơn hàng'].nunique())""",
    # số đợt giao tháng 4/2026: mỗi (đơn, lần giao) một đợt
    "q_087a4081a1a9": """
g = T['Bang_ke_phieu_giao_hang']
g = g[g['Ngày giao hàng'].dt.date <= TODAY]
m = g[dm.month(2026, 4).mask(g['Ngày giao hàng'])]
ANSWER = dm.count(m[['Mã đơn hàng', 'Lần giao']].drop_duplicates().shape[0])""",
}


@pytest.fixture(scope="module")
def setup():
    biz_dir = DATA / "data" / "biz_118"
    loader.load_cached(biz_dir, CACHE)
    qs = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "questions.jsonl").read_text().splitlines()}
    gold = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "answers.jsonl").read_text().splitlines()}
    return cache_path(biz_dir, CACHE), qs, gold


@pytest.mark.parametrize("qid", list(CASES))
def test_handwritten_solution(setup, qid):
    pkl, qs, gold = setup
    res = run_code(CASES[qid], pkl, DATA / "calendar.csv", qs[qid]["as_of_date"], timeout=120)
    assert res["ok"], res.get("error")
    pred = formatter.finalize(qid, res["answer"])
    assert score.match(pred, gold[qid]) == 1, (pred, gold[qid])


class FakeLLM:
    """Lần gọi đầu của mỗi lượt trả code lỗi, lần sửa trả code đúng."""

    def __init__(self, good_code):
        self.good = good_code
        self.calls = 0

    def chat(self, messages, temperature, seed):
        self.calls += 1
        if messages[-1]["content"].startswith("Code lỗi"):
            return f"```python\n{self.good}\n```"
        return "```python\nANSWER = dm.money(T['khong_co_bang']['x'].sum())\n```"


def test_agent_fix_and_vote(setup):
    from datamate.agent import Agent

    pkl, qs, gold = setup
    qid = "q_158b312e8f5b"
    llm = FakeLLM(CASES[qid])
    cfg = {"votes": 3, "min_agree": 2, "max_fixes": 1, "exec_timeout_s": 120}
    agent = Agent(llm, cfg, (DATA / "rulebook.md").read_text(), DATA / "calendar.csv")
    biz = loader.load_cached(DATA / "data" / "biz_118", CACHE)
    final, log = agent.answer(qs[qid], biz, pkl)
    assert score.match(final, gold[qid]) == 1
    assert llm.calls == 4  # 2 lượt × (lỗi + sửa), dừng sớm khi đủ 2 phiếu
    assert "KeyError" in log["steps"][0]["result"]["error"]
    assert log["decision"] == "consensus" and log["llm_calls"] == 4
    assert log["steps"][0]["error_code"] == "EXECUTION_FAILED"


class WrongTypeLLM:
    """Lần đầu trả số thay vì phần trăm (verifier phải bắt), lần sửa trả đúng."""

    def __init__(self, good_code):
        self.good = good_code

    def chat(self, messages, temperature, seed):
        if messages[-1]["content"].startswith("Code lỗi"):
            return f"```python\n{self.good}\n```"
        return "```python\nANSWER = dm.count(5)\n```"


def test_agent_verifier_repair(setup):
    from datamate.agent import Agent

    pkl, qs, gold = setup
    qid = "q_4293a4130594"  # "... bao nhiêu % đơn bị huỷ?"
    cfg = {"votes": 2, "min_agree": 2, "max_fixes": 1, "exec_timeout_s": 120}
    agent = Agent(WrongTypeLLM(CASES[qid]), cfg, (DATA / "rulebook.md").read_text(), DATA / "calendar.csv")
    biz = loader.load_cached(DATA / "data" / "biz_118", CACHE)
    final, log = agent.answer(qs[qid], biz, pkl)
    assert score.match(final, gold[qid]) == 1
    assert log["steps"][0]["error_code"] == "TYPE_MISMATCH"


def test_agent_min_agree_by_type(setup):
    from datamate.agent import Agent

    pkl, qs, gold = setup
    qid = "q_158b312e8f5b"

    class Good:
        def chat(self, messages, temperature, seed):
            return f"```python\n{CASES[qid]}\n```"

    cfg = {"votes": 2, "min_agree": 2, "min_agree_by_type": {"number": 3}, "max_fixes": 0, "exec_timeout_s": 120}
    agent = Agent(Good(), cfg, (DATA / "rulebook.md").read_text(), DATA / "calendar.csv")
    biz = loader.load_cached(DATA / "data" / "biz_118", CACHE)
    final, log = agent.answer(qs[qid], biz, pkl)
    assert final["type"] == "abstain" and log["decision"] == "no_consensus"


def _run(setup, code):
    pkl, _, _ = setup
    return run_code(code, pkl, DATA / "calendar.csv", "2026-03-04", timeout=120)


def test_sandbox_import_dm_and_main_guard(setup):
    """Lỗi hay gặp nhất ở holdout B0: `import dm` và ANSWER nằm trong `if __name__ == '__main__':`."""
    res = _run(setup, "import dm\nfrom dm import money\nif __name__ == '__main__':\n    ANSWER = money(5)")
    assert res["ok"] and res["answer"] == {"type": "number", "answer": 5}


def test_sandbox_raw_answer_message(setup):
    res = _run(setup, "ANSWER = 42")
    assert not res["ok"] and "ANSWER đang là int" in res["error"]


def test_sandbox_keyerror_lists_columns(setup):
    res = _run(setup, "x = T['So_chi_tiet_ban_hang']['Mã chứng từ']")
    assert not res["ok"] and "Gợi ý" in res["error"] and "'Mã đơn hàng'" in res["error"]


def test_sandbox_exit(setup):
    res = _run(setup, "exit()")
    assert not res["ok"] and "SystemExit" in res["error"]


def test_sandbox_model_style_code_now_runs(setup):
    """Các cách viết hay gặp trong log holdout v2.2 giờ chạy được và cho cùng kết quả với cách chuẩn (TODAY = 2026-03-04)."""
    code = """
s = T['So_chi_tiet_ban_hang']
p = dm.last_week(TODAY)
a = s[p.mask(s['Ngày'])]['Thành tiền'].sum()                                         # cách chuẩn
b = s[(s['Ngày'] >= p.start) & (s['Ngày'] <= p.end)]['Thành tiền'].sum()            # so cột datetime với date
c = s[s['Ngày'].dt.date.isin(p)]['Thành tiền'].sum()                                 # isin(Period)
d = s[s['Ngày'].dt.date.isin(pd.date_range(dm.date(2026, 2, 23), dm.date(2026, 3, 1)).date)]['Thành tiền'].sum()
k = s[~dm.is_walkin(s['Khách hàng'])].shape[0] + s[dm.is_walkin(s['Khách hàng'])].shape[0]
assert a == b == c == d and a > 0, (a, b, c, d)
assert k == len(s)
assert (s['Ngày'] == TODAY).sum() == p.mask(s['Ngày']).sum() * 0 + dm.day(TODAY).mask(s['Ngày']).sum()
ANSWER = dm.money(dm.money(a))
"""
    res = _run(setup, code)
    assert res["ok"], res.get("error")
    assert res["answer"]["type"] == "number"


def test_sandbox_date_answer_autowrap(setup):
    res = _run(setup, "ANSWER = dm.date(2026, 3, 1)")
    assert res["ok"] and res["answer"] == {"type": "date", "answer": "2026-03-01"}


def test_sandbox_new_hints(setup):
    res = _run(setup, "x = pd.date_range(dm.date('2026-02-01'), TODAY)")
    assert not res["ok"] and "dt.date(y, m, d)" in res["error"]
    res = _run(setup, "x = T['So_chi_tiet_ban_hang']['Thanh tien']")
    assert not res["ok"] and "Gần nhất" in res["error"] and "'Thành tiền'" in res["error"]
    res = _run(setup, "x = dm.match_one(T['So_chi_tiet_ban_hang']['Tên hàng hóa'], 'zzz qqq wwwvvv')")
    assert not res["ok"] and "LookupError" in res["error"] and "không tìm thấy" in res["error"]
