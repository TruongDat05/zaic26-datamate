"""Verifier, holdout, thống kê log, mã có số 0 đầu (đặc tả V2)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datamate import loader, verify  # noqa: E402
from datamate.fewshot import FewShot  # noqa: E402
from datamate.run import log_stats  # noqa: E402

DATA = ROOT / "data" / "train"


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- verifier

@pytest.mark.skipif(not (DATA / "answers.jsonl").is_file(), reason="requires local train answers")
def test_verifier_accepts_every_train_gold():
    """Quy tắc chỉ được giữ khi không chặn nhầm đáp án chuẩn nào của train."""
    gold = {r["id"]: r for r in _jsonl(DATA / "answers.jsonl")}
    bad = [(q["question"], gold[q["id"]]) for q in _jsonl(DATA / "questions.jsonl")
           if verify.check(q["question"], gold[q["id"]])]
    assert not bad, bad[:5]


@pytest.mark.parametrize("question,ans,code", [
    ("tháng 3 có bao nhiêu % đơn bị huỷ", {"type": "number", "answer": 12}, "TYPE_MISMATCH"),
    ("tháng nào doanh thu cao nhất", {"type": "date", "answer": "2025-10-01"}, "TYPE_MISMATCH"),
    ("ngày nào bán được nhiều tiền nhất", {"type": "date", "answer": "2025-10"}, "TYPE_MISMATCH"),
    ("tháng này có bán được Sting không?", {"type": "number", "answer": 3}, "TYPE_MISMATCH"),
    ("hôm qua bán được bao nhiêu tiền", {"type": "number", "answer": 1500.5}, "TYPE_MISMATCH"),
    ("top 3 khách mua nhiều nhất", {"type": "string", "answer": "Lan"}, "TYPE_MISMATCH"),
    ("top 3 khách mua nhiều nhất", {"type": "list", "answer": ["a", "b", "c", "d"]}, "TOPK_TOO_LONG"),
])
def test_verifier_flags(question, ans, code):
    assert [i.code for i in verify.check(question, ans)] == [code]


@pytest.mark.parametrize("question,ans", [
    ("tháng 3 có bao nhiêu % đơn bị huỷ", {"type": "unanswerable"}),
    ("top 3 khách mua nhiều nhất", {"type": "list", "answer": ["a", "b"]}),  # ít hơn k ứng viên
    ("hôm qua bán được bao nhiêu tiền", {"type": "number", "answer": 1500}),
])
def test_verifier_passes(question, ans):
    assert verify.check(question, ans) == []


# ---------------------------------------------------------------- holdout

@pytest.mark.skipif(not (DATA / "questions.jsonl").is_file(), reason="requires local train questions")
def test_fewshot_excludes_holdout():
    holdout = (ROOT / "eval" / "holdout_businesses.txt").read_text().split()
    fs = FewShot(DATA / "questions.jsonl", DATA / "answers.jsonl", holdout)
    assert not {q["business_id"] for q, _ in fs.items} & set(holdout)
    assert fs.similar("doanh thu tháng này", "biz_101", 4)


# ---------------------------------------------------------------- thống kê log

def test_log_stats(tmp_path):
    for i, (calls, sec, dec) in enumerate([(1, 2.0, "consensus"), (3, 10.0, "no_consensus"), (2, 4.0, "consensus")]):
        (tmp_path / f"q_{i}.json").write_text(json.dumps({"llm_calls": calls, "elapsed_s": sec, "decision": dec}))
    (tmp_path / "q_9.json").write_text(json.dumps({"id": "q_9", "skipped": "hết thời gian"}))
    st = log_stats(tmp_path, {"q_0", "q_1", "q_2", "q_9"})
    assert st["llm_calls_mean"] == 2 and st["timed_out_questions"] == 1
    assert st["decisions"] == {"consensus": 2, "no_consensus": 1}
    assert st["question_seconds_p50"] == 4.0


# ---------------------------------------------------------------- loader

def test_leading_zero_codes_stay_text():
    df = pd.DataFrame({"Hàng": ["0012", "0047", "0100", "0003"], "SL": ["5", "7", "1", "2"]})
    out = loader._infer_columns(df, None, [])
    assert out["Hàng"].tolist() == ["0012", "0047", "0100", "0003"]
    assert out["SL"].tolist() == [5, 7, 1, 2]


@pytest.mark.parametrize("biz,tbl,st,oid,dcol,amt,y,m,qid", [
    ("biz_104", "DonHang_ChiTiet", "Trạng thái đơn hàng", "Mã đơn hàng", "Ngày đặt hàng", "Tổng tiền", 2026, 2,
     "q_cc3b3986ea00"),  # dòng 'Tổng ngày dd/mm' từng bị gán vào đơn cuối ngày
    ("biz_105", "SoBanHang", "Status", "Mã hóa đơn", "Ngày giao dịch", "Trị giá", 2026, 1,
     "q_2b96cf7eb6b3"),  # tiền '45k' + dòng trùng cùng đơn (cột hàng tên ngắn 'Hàng')
    ("biz_117", "SoBanHang", "Ghi chú TT", "Mã phiếu", "Ngày lập", "Tiền", 2025, 4,
     "q_6821f6e7e582"),  # tiền '1tr270' + dòng 'Tổng tháng'
    ("biz_130", "So_chi_tiet_ban_hang", "Trạng thái", "Mã đơn hàng", "Ngày", "Thành tiền", 2026, 6,
     "q_6e9a3164fe79"),  # dòng 'TỔNG NGÀY dd/mm'
])
@pytest.mark.skipif(not (DATA / "answers.jsonl").is_file(), reason="requires local competition train data")
def test_loader_monthly_revenue_matches_gold(biz, tbl, st, oid, dcol, amt, y, m, qid):
    """Doanh thu tháng tính thẳng từ bảng đã làm sạch phải khớp đáp án chuẩn (lỗi tìm được qua EDA)."""
    from datamate import dm
    s = loader.load_business(DATA / "data" / biz).tables[tbl].copy()
    s["st"] = s[st].map(dm.norm_status)
    bad = set(s.loc[s["st"].isin(["huy", "tra_hang"]), oid])
    ok = s[~s[oid].isin(bad)]
    gold = {r["id"]: r for r in _jsonl(DATA / "answers.jsonl")}[qid]["answer"]
    assert int(ok[dm.month(y, m).mask(ok[dcol])][amt].sum()) == gold
