"""Gợi ý luật theo câu hỏi (rules.py) và code mẫu few-shot (v2.4)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from datamate import agent, rules, templates_ext  # noqa: E402
from datamate.fewshot import FewShot  # noqa: E402

PUB_RB = (ROOT / "data/public_test/rulebook.md").read_text(encoding="utf-8")
TRAIN = ROOT / "data/train"


def _names(q, rb=PUB_RB):
    return [r.name for r in rules.relevant(rules.parse(rb), q)]


def test_concept_rules_found():
    assert _names("Khách sộp nhất tháng 5/2026 là ai?")[0] == "Khách sộp nhất"
    assert _names("Trong tháng 3/2026, giờ cao điểm trong ngày của cửa hàng là mấy giờ?") == ["Giờ cao điểm"]
    assert "Nhà cung cấp chính" in _names("tháng 12/2025, ncc chính của cửa hàng là ai?")
    assert "Từ đầu năm, từ đầu năm tới nay" in _names("Dthu tu dau nam toi het hom nay la bao nhieu z?")


def test_no_noise_for_plain_question():
    assert "Mặt hàng không bán được trong kỳ (ế)" not in _names("Tháng 3/2025 bán được bao nhiêu ly trà đào?")


def test_block_in_prompt():
    rb = rules.parse(PUB_RB)
    txt = rules.block(rb, "Khách sộp nhất tháng 5/2026 là ai?")
    assert "từ 3 đơn trở lên" in txt


def _bank(tmp_path, n=3):
    qs = [json.loads(l) for l in (TRAIN / "questions.jsonl").read_text(encoding="utf-8").splitlines()[:n]]
    p = tmp_path / "bank.jsonl"
    p.write_text("\n".join(json.dumps({"id": q["id"], "business_id": q["business_id"], "question": q["question"],
                                       "code": f"ANSWER = dm.count({i})"}) for i, q in enumerate(qs)), encoding="utf-8")
    return p, qs


def test_code_examples_in_messages(tmp_path):
    p, qs = _bank(tmp_path)
    fs = FewShot(TRAIN / "questions.jsonl", TRAIN / "answers.jsonl", (), p)
    got = fs.similar_code(qs[0]["question"], "dev_001", 2)
    assert got and got[0][0]["id"] == qs[0]["id"] and got[0][1].startswith("ANSWER")
    assert fs.similar_code(qs[0]["question"], qs[0]["business_id"], 3)[0][0]["business_id"] != qs[0]["business_id"]

    class B:
        business_id, readme = "dev_001", "readme"
        def profile(self):
            return "bảng"
    ag = agent.Agent(None, {"fewshot_k": 2}, PUB_RB, TRAIN / "calendar.csv", fs)
    msg = ag._messages({"question": "Khách sộp nhất tháng 5/2026 là ai?", "as_of_date": "2026-06-01",
                        "business_id": "dev_001"}, B(), agent.DEFAULT_VARIANTS[1])[1]["content"]
    assert "Code đã chấm đúng" in msg and "```python\nANSWER = dm.count(" in msg and "Khách sộp nhất" in msg


def test_fewshot_without_bank_unchanged():
    fs = FewShot(TRAIN / "questions.jsonl", TRAIN / "answers.jsonl", (), None)
    assert fs.similar_code("doanh thu", "dev_001") == []


def test_build_code_bank(tmp_path, monkeypatch):
    import build_code_bank
    gold = {json.loads(l)["id"]: json.loads(l) for l in (TRAIN / "answers.jsonl").read_text(encoding="utf-8").splitlines()[:1]}
    (gid, g), = gold.items()
    ans = {k: g[k] for k in ("type", "answer") if k in g}
    final = {"id": gid, **ans}
    log = {"id": gid, "business_id": "biz_x", "question": "q", "decision": "consensus", "final": final,
           "steps": [{"response": "```python\nANSWER = 1\n```", "result": {"ok": True, "answer": ans}}]}
    (tmp_path / "run" / "logs").mkdir(parents=True)
    (tmp_path / "run" / "logs" / f"{gid}.json").write_text(json.dumps(log), encoding="utf-8")
    monkeypatch.setattr(build_code_bank, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--runs", str(tmp_path / "run"), "--gold", str(TRAIN / "answers.jsonl")])
    build_code_bank.main()
    rows = [json.loads(l) for l in (tmp_path / "data" / "code_bank.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": gid, "business_id": "biz_x", "question": "q", "code": "ANSWER = 1"}]


# ---------------------------------------------------------------- lỗi nạp dữ liệu public (v2.6)

PUB = ROOT / "data/public_test/data"


def _load(b):
    from datamate import loader
    return loader.load_business(PUB / b).tables


def test_daily_total_rows_dropped():
    """Dòng 'Cộng ngày', 'Tổng ngày', dòng chỉ có số tiền từng làm doanh thu gấp đôi."""
    from datamate import dm
    t = _load("dev_009")["danh_sach_hoa_don"]
    assert t["Mã hóa đơn"].notna().all() and not t["Đối tác"].astype(str).str.startswith("Tổng ngày").any()
    t["st"] = t["Trạng thái đơn"].map(dm.norm_status)
    ok = t[~t["st"].isin(["huy", "tra_hang"]) & dm.quarter(2025, 1).mask(t["Ngày tháng"])]
    assert int(ok["TongTien"].sum()) == 1798595000  # sau khi bỏ cả dòng trùng của cùng đơn (v2.7)
    for b, tb, idc in [("dev_001", "So_chi_tiet_ban_hang", "Mã đơn hàng"), ("dev_007", "chi_tiet_hoa_don", "Mã bill"),
                       ("dev_008", "ChiTietHoaDon", "Bill"), ("dev_004", "SoBanHang", "Ma don")]:
        assert _load(b)[tb][idc].notna().all()


def test_sub_rows_get_order_date():
    t = _load("dev_013")["BanHang"]
    assert t["Ngày"].notna().all() and t.loc[t["Tên hàng"] == "Phí ship", "Ngày"].notna().all()
    assert t["ĐG"].isna().any()  # cột cấp dòng không bị điền
    assert _load("dev_014")["Tra_gop"]["Khách"].notna().all()


def test_notebook_order_id_filled_money_not():
    t = _load("dev_011")["So_ban_hang"]
    assert t["STT"].notna().all() and t["Khách"].notna().all()
    assert t["Bớt"].isna().any()  # tiền cả đơn không chép xuống, cộng cột không bị gấp


def test_alias_columns():
    t = _load("dev_012")["SoSuaXe"]
    assert set(t["Thợ__ho_ten"].dropna()) == {"Hà Loan Đạt", "Lương Chi Đức", "Đoàn Bảo Phúc", "Đàm Lan An"}
    assert t["Thợ__ho_ten"].notna().mean() > 0.95 and t["Xe__bien_so"].iloc[0] == "59B6-219.16"
    b = _load("dev_011")["So_ban_hang"]
    assert b["Hàng__ten"].notna().all() and (b.loc[b["Hàng"] == "Cáp C", "Hàng__ten"] == "Cáp Type-C 1m").all()


def test_zero_suspect():
    from datamate import verify
    assert verify.zero_suspect("Tháng 9/2025 Cháo ngao bán được mấy nồi?", {"type": "number", "answer": 0})
    assert not verify.zero_suspect("7 ngày gần nhất có mấy mặt hàng ế?", {"type": "number", "answer": 0})
    assert not verify.zero_suspect("bán được mấy nồi?", {"type": "number", "answer": 3})
    gold = {json.loads(l)["id"]: json.loads(l) for l in (TRAIN / "answers.jsonl").read_text(encoding="utf-8").splitlines()}
    qs = [json.loads(l) for l in (TRAIN / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not [q["question"] for q in qs if verify.zero_suspect(q["question"], gold[q["id"]])]


# ---------------------------------------------------------------- v2.7

def test_merge_name_variants_gop_cach_viet_giu_ten_khac_dau():
    from datamate.loader import _merge_name_variants
    df = pd.DataFrame({"Tên sản phẩm": ["Sinh tố xoài (L)"] * 3 + ["SINH TỐ XOÀI (L)", "Sinh  tố xoài (l)", "sinh to xoai (L)",
                                        "Sting dâu", "Sting dâu", "Sting dầu"]})
    notes = []
    out = _merge_name_variants(df, notes)
    assert out["Tên sản phẩm"].value_counts().to_dict() == {"Sinh tố xoài (L)": 6, "Sting dâu": 2, "Sting dầu": 1}
    assert notes


def test_merge_name_variants_khong_dung_cot_khac():
    from datamate.loader import _merge_name_variants
    df = pd.DataFrame({"dvt": ["bộ", "bó", "bó"], "Mã hàng": ["a1", "A1", "A1"], "Ghi chú": ["X", "x", "x"]})
    assert _merge_name_variants(df, []).equals(df)


def test_dem_don_khong_dem_dong():
    from datamate import dm
    df = pd.DataFrame({"ma": ["A", "A", "A", "B", "C", "C"], "tt": ["Đã hủy", "Đã hủy", "Đã hủy", "Đã giao", "Đã trả hàng", "Đã trả hàng"]})
    assert dm.n_orders(df, "ma") == 3
    assert dm.cancel_rate(df, "ma", "tt")["answer"] == 33.33
    assert dm.n_distinct(pd.DataFrame({"ma": ["A", "A", "A"], "luot": [1, 1, 2]}), ["ma", "luot"]) == 2


def test_round_money_chi_lam_tron_tien():
    from datamate.verify import round_money
    assert round_money("Tháng 2/2026 doanh thu bình quân mỗi đơn là bao nhiêu?", {"type": "number", "answer": 70873.95})["answer"] == 70874
    assert round_money("Giá nhập bình quân một cây thép?", {"type": "number", "answer": 203031.5})["answer"] == 203032
    assert round_money("trung bình mỗi đơn mấy món", {"type": "number", "answer": 2.35})["answer"] == 2.35
    assert round_money("doanh thu tăng mấy %", {"type": "percent", "answer": 2.35})["answer"] == 2.35


def test_concept_block_nhan_dung_cau_da_sai():
    from datamate.rules import concept_block
    assert "cancel_rate" in concept_block("thang 1/2026 ti le don bi huy la bnhiu %")
    assert "cancel_rate" in concept_block("Tháng 2/2026 cửa hàng có bao nhiêu % đơn bị huỷ ạ?")
    assert "n_orders" in concept_block("thang 10/2025 bi huy may don")
    assert "n_distinct" in concept_block("thang 2/2025 xe da giao tong cong may dot hang")
    assert "prev_month_same_days" in concept_block("từ đầu tháng này đến hết hôm nay, doanh thu thay đổi bnhiu % so vs cùng kỳ th trước")
    assert "dm.money" in concept_block("Tháng 2/2026 doanh thu bình quân mỗi đơn hàng là bao nhiêu?")
    assert concept_block("Tháng 11/2025 nhóm trà có mặt hàng nào bán chạy nhất?") == ""


def test_status_dang_gui():
    from datamate.clean import norm_status
    assert norm_status("đang gửi") == "dang_giao"


def test_drop_types_bo_trong_du_dong_thuan():
    ag = agent.Agent(None, {"votes": 2, "min_agree": 2, "drop_types": ["unanswerable"]}, PUB_RB, TRAIN / "calendar.csv")
    ag._messages = lambda *a: []
    ag._attempt = lambda *a, **k: {"id": "q", "type": "unanswerable", "reason": "x"}
    final, log = ag.answer({"id": "q", "question": "?", "as_of_date": "2026-01-01", "business_id": "b"}, None, None)
    assert final == {"id": "q", "type": "abstain"} and log["decision"] == "dropped_type"
    ag._attempt = lambda *a, **k: {"id": "q", "type": "number", "answer": 5}
    assert ag.answer({"id": "q", "question": "?", "as_of_date": "2026-01-01", "business_id": "b"}, None, None)[0]["answer"] == 5


# ---------------------------------------------------------------- v2.8: lớp tính toán

@pytest.fixture(autouse=True)
def _rulebook_trong():
    templates_ext.RULEBOOK = ""  # Agent tạo ở test khác gán rulebook giả
    yield
    templates_ext.RULEBOOK = ""


def _shop(b, split="train"):
    from datamate import loader, shop
    return shop.build(loader.load_business(ROOT / f"data/{split}/data/{b}"))


def test_loader_bo_dong_trung_khi_cot_hang_ten_la():
    from datamate import dm
    s = _shop("biz_124")
    assert s.revenue(dm.month(2025, 5)) == 346554700  # đáp án train; để lại dòng trùng thì ra 348973500


def test_shop_tinh_dung_dap_an_train():
    from datamate import dm
    s = _shop("biz_104")
    assert s.n_cancelled(dm.month(2025, 10)) == 60
    assert s.cancel_rate(dm.month(2026, 1)) == 1.89
    assert s.revenue(dm.month(2025, 11), item="Sinh tố xoài (L)") == 2640000
    assert s.avg_order(dm.month(2026, 2)) == 70874


def test_templates_khong_doan_khi_cau_co_y_khac():
    from datamate import templates
    s = _shop("biz_104")
    for q in ["Tháng 5/2025 doanh thu ca sáng là bao nhiêu?", "Tháng 5/2025 khách Lan mua bao nhiêu tiền?",
              "Doanh thu của Món Không Có Thật tháng 5/2025 là bao nhiêu?", "Tháng 5/2025 và 6/2025 doanh thu bao nhiêu?"]:
        assert templates.answer(q, "2026-06-02", s) is None, q
    assert templates.answer("dthu t5/25 bnhiu", "2026-06-02", s)["template"] == "revenue"
    assert templates.answer("hqua dc may don", "2026-06-02", s)["template"] == "n_orders"


def test_loader_dien_dong_phu_khong_keo_ghi_chu_don_truoc():
    from datamate import loader
    df = pd.DataFrame({"Mã đơn": ["A", None, "B", None], "Tên hàng": ["x", "y", "z", "t"], "Ghi chú": ["khách trả lại", None, None, None]})
    out = loader._group_fill_and_dedupe(df, [])
    assert out["Ghi chú"].tolist()[:2] == ["khách trả lại", "khách trả lại"]
    assert out["Ghi chú"].isna().tolist()[2:] == [True, True]


def test_templates_mo_rong():
    from datamate import dm, templates
    s = _shop("biz_124")
    assert templates.answer("Tháng 5/2025 doanh thu ca sáng là bao nhiêu?", "2026-05-25", s)["template"] == "shift"
    assert templates.answer("dthu ca chieu chenh ca sang may % thang 5/2025", "2026-05-25", s)["template"] == "shift_change"
    assert templates.answer("Tháng 5/2025 ai mua hàng nhiều tiền nhất?", "2026-05-25", s)["template"] == "top_customer"
    assert templates.answer("Tháng 5/2025 ca sáng khách nào chi nhiều nhất?", "2026-05-25", s) is None
    assert s.rank_items(dm.month(2025, 5))[0] != "Phí phục vụ"


def test_bo_phieu_hai_model_moi_luot_mot_model():
    calls = []

    class M:
        def __init__(self, n):
            self.n = n

        def chat(self, msgs, temperature, seed):
            calls.append(self.n)
            return "```python\nANSWER = dm.count(5)\n```"
    ag = agent.Agent(M("a"), {"votes": 2, "min_agree": 2, "templates": False}, PUB_RB, TRAIN / "calendar.csv", None, [M("a"), M("b")])
    ag._messages = lambda *a: []
    import datamate.agent as am
    orig = am.run_code
    am.run_code = lambda *a, **k: {"ok": True, "answer": {"type": "number", "answer": 5}}
    try:
        final, _ = ag.answer({"id": "q", "question": "?", "as_of_date": "2026-01-01", "business_id": "b"}, None, None)
    finally:
        am.run_code = orig
    assert calls == ["a", "b"] and final["answer"] == 5


def test_chuan_hoa_cau_hoi():
    from datamate.normq import normalize
    assert normalize("cuwar hàng dduwowcj bnhiu") == "cua hang duoc bao nhieu"
    assert normalize("tieenf bán") == "tien ban"
    assert normalize("2-2026 tong doanh thu duoc bao nheiu") == "2 2026 tong doanh thu duoc bao nhieu"
    assert normalize("bua qua dthu dc bn") == "hom qua doanh thu duoc bao nhieu"
    assert normalize("Trà sữa hop 24 gói") == "tra sua hop 24 goi"  # đơn vị trong tên hàng không bị sửa nhầm
    assert normalize("chong tham", {"chong", "tham"}) == "chong tham"  # từ trong tên hàng của shop giữ nguyên


def test_lop_tinh_toan_mo_rong_dung_dap_an_train():
    from datamate import templates
    s = _shop("biz_106")
    cases = [("Tháng 1/2026 cửa hàng bỏ ra bao nhiêu tiền nhập hàng?", "2026-09-14", 93227900, "purchase_money"),
             ("Tính đến hết ngày 16/12/2025, khách hàng còn nợ cửa hàng tổng cộng bao nhiêu tiền?", "2026-09-14", 1005670, "debt_total"),
             ("Giá bán một ly Trà sữa ô long (L) vào ngày 30/03/2026 là bao nhiêu?", "2026-09-14", 53500, "price_on_day")]
    for q, today, want, name in cases:
        r = templates.answer(q, today, s)
        assert r is not None and r["answer"] == want and r["template"] == name, q


def test_lop_tinh_toan_khong_tra_so_khi_hoi_doi_tuong():
    from datamate import templates
    s = _shop("biz_104")
    r = templates.answer("Tháng 5/2025 mặt hàng nào thu về nhiều tiền nhất?", "2026-06-02", s)
    assert r is None or r["type"] == "string"
    assert templates.answer("Tổng tiền thu máy cũ của khách tháng 5/2025?", "2026-06-02", s) is None


def test_so_hai_thang_moc_la_thang_sau_chu_so():
    from datamate import templates
    s = _shop("biz_104")
    a = templates.answer("So với T4/2025 thì T4/2026 doanh thu tăng trưởng bao nhiêu %?", "2026-06-02", s)
    b = templates.answer("T4/2026 so với T4/2025 doanh thu tăng trưởng bao nhiêu %?", "2026-06-02", s)
    assert a is not None and a["answer"] == b["answer"]


def test_nam_nay_tinh_toi_hom_nay_la_tu_dau_nam():
    from datamate.normq import normalize
    assert "tu dau nam den hom nay" in normalize("Năm nay tính tới hôm nay bán được bao nhiêu tiền?")


def test_ten_ngan_khong_khop_tu_thuong():
    from datamate.templates import find_name
    assert find_name("doanh thu cac don tra bang chuyen khoan", ["Trà"]) is None
    assert find_name("doanh thu nhom tra thang 5", ["Trà"]) == "Trà"


def test_tra_loi_khong_duoc_khi_du_lieu_khong_ghi():
    from datamate import templates
    s = _shop("biz_104")
    r = templates.answer("Nhân viên nào phục vụ khách thân thiện nhất?", "2026-06-02", s)
    assert r is None or r["type"] == "unanswerable"


def test_go_sai_giua_tu_va_tu_chua_doc_duoc():
    from datamate.normq import has_unread_typo, normalize
    assert "thang nay" in normalize("thasng nafy")
    assert has_unread_typo("khach hoafng", set())
    assert not has_unread_typo("khach hoafng", {"hoafng"})
    assert not has_unread_typo("paracetamol ibuprofen", set())


def test_tu_nhan_vai_tro_cot_khi_ten_la():
    df = pd.DataFrame({"A": ["HD001", "HD001", "HD002"], "B": ["Hàng số một", "Hàng số hai", "Hàng số một"], "C": [2.0, 1.0, 3.0], "D": [10000.0, 20000.0, 10000.0], "E": [20000.0, 20000.0, 30000.0],
                       "F": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]), "G": ["Hoàn thành", "Hoàn thành", "Đã huỷ"]})
    from types import SimpleNamespace
    from datamate.shop import _infer_by_values
    out = {}
    _infer_by_values(SimpleNamespace(tables={"T": df}), out)
    assert out["T"]["amount"] == "E" and out["T"]["date"] == "F" and out["T"]["status"] == "G"


def test_dong_luat_cua_lop_tinh_toan_co_trong_rulebook_train():
    from datamate import templates_ext
    book = (ROOT / "data/train/rulebook.md").read_text(encoding="utf-8")
    missing = [k for k, v in templates_ext.RULE_OF.items() if v not in book]
    assert not missing, missing


def test_dong_luat_doi_thi_chuyen_cho_model():
    from datamate import templates, templates_ext
    s = _shop("biz_104")
    q = "Tháng 5/2025 doanh thu là bao nhiêu?"
    book = (ROOT / "data/train/rulebook.md").read_text(encoding="utf-8")
    templates_ext.RULEBOOK = book
    assert templates.answer(q, "2026-06-02", s) is not None
    templates_ext.RULEBOOK = book.replace(templates_ext._R_REV, "Đơn tính vào doanh thu | luật khác")
    assert templates.answer(q, "2026-06-02", s) is None
    templates_ext.RULEBOOK = ""


class _FakeLLM:
    """Model giả: trả lần lượt các câu đã chuẩn bị."""
    def __init__(self, codes=(), jsons=()):
        self.codes, self.jsons = list(codes), list(jsons)

    def chat(self, messages, temperature, seed):
        return self.codes.pop(0) if self.codes else "```python\nANSWER = dm.abstain('x')\n```"

    def chat_json(self, messages, schema, seed=42):
        return self.jsons.pop(0) if self.jsons else None


def test_hop_dong_chi_nhan_vai_tro_qua_kiem_tra():
    from types import SimpleNamespace
    from datamate import contract
    df = pd.DataFrame({"A": ["HD1", "HD1"], "B": ["Hàng một", "Hàng hai"], "C": [2.0, 1.0], "D": [10.0, 20.0], "E": [20.0, 20.0], "F": pd.to_datetime(["2025-01-01", "2025-01-02"]), "G": ["Hoàn thành", "Đã huỷ"]})
    biz = SimpleNamespace(tables={"T": df})
    good = contract.validate(biz, {"tables": [{"table": "T", "roles": {"order": "A", "item": "B", "qty": "C", "price": "D", "amount": "E", "date": "F", "status": "G"}}]})
    assert good["T"]["amount"] == "E" and good["T"]["date"] == "F"
    bad = contract.validate(biz, {"tables": [{"table": "T", "roles": {"date": "B", "amount": "B", "status": "C", "item": "Không có"}}, {"table": "X", "roles": {"item": "B"}}]})
    assert bad == {}
    wrong_math = contract.validate(biz, {"tables": [{"table": "T", "roles": {"qty": "D", "price": "D", "amount": "C"}}]})
    assert "amount" not in wrong_math.get("T", {})
    assert contract.validate(biz, None) == {} and contract.validate(biz, "rác") == {}


def test_hop_dong_chi_dien_cho_con_thieu():
    from datamate import shop as sm
    b = __import__("datamate.loader", fromlist=["x"]).load_business(ROOT / "data/train/data/biz_104")
    roles = sm._table_roles(b)
    t = next(k for k, r in roles.items() if "amount" in r)
    b.extra_roles = {t: {"amount": "Cột sai"}}
    assert sm._table_roles(b)[t]["amount"] == roles[t]["amount"]


def test_plan_doc_ra_cau_chuan_qua_lop_tinh_toan():
    from datamate import plan, templates
    s = _shop("biz_104")
    period = {"kind": "month", "year": 2025, "month": 5}
    cases = [({"measure": "revenue", "period": period}, "revenue"), ({"measure": "n_orders", "period": period}, "n_orders"), ({"measure": "n_cancelled", "period": period}, "n_cancelled"),
             ({"measure": "cancel_rate", "period": period}, "cancel_rate"), ({"measure": "avg_order", "period": period}, "avg_order"), ({"measure": "best_day", "period": period}, "best_day"),
             ({"measure": "top_items", "period": period, "top_k": 3}, "top_items"), ({"measure": "top_customers_money", "period": period}, "top_customer"),
             ({"measure": "revenue", "period": period, "item": s.items[0]}, "revenue"), ({"measure": "qty", "period": period, "item": s.items[0]}, "qty")]
    for slots, name in cases:
        r = templates.answer(plan.render(slots, s), "2026-06-02", s)
        assert r is not None and r["template"].replace("g_avg", "avg_order") == name, slots
    assert plan.render({"measure": "unsupported", "period": period}, s) is None
    assert plan.render({"measure": "revenue", "period": period, "item": "Món không có"}, s) is None
    assert plan.render({"measure": "revenue", "period": {"kind": "month"}}, s) is None


def _agent(llm, **cfg):
    import datamate.agent as ag
    return ag.Agent(llm, {"votes": 2, "min_agree": 2, "rule_hints": False, "max_fixes": 0, "templates": False, **cfg}, (ROOT / "data/train/rulebook.md").read_text(encoding="utf-8"), Path("/nonexistent"))


def test_hai_duong_khop_moi_nop_khac_thi_bo_trong(monkeypatch):
    from datamate import agent as ag, templates
    s = _shop("biz_104")
    q = {"id": "q", "question": "câu lạ", "as_of_date": "2026-06-02", "business_id": "biz_104"}
    slots = {"measure": "n_orders", "period": {"kind": "month", "year": 2025, "month": 5}}
    want = templates.answer("thang 5 2025 co bao nhieu don", "2026-06-02", s)["answer"]
    biz = type("B", (), {"business_id": "biz_104", "readme": "r", "profile": lambda self: "p", "tables": {}})()
    def run(answer):
        def fake(code, *a, **k):
            return {"ok": True, "answer": {"type": "number", "answer": answer}}
        monkeypatch.setattr(ag, "run_code", fake)
        llm = _FakeLLM(codes=["```python\nANSWER = 1\n```"] * 4, jsons=[slots])
        return _agent(llm, plan=True, dual_path=True).answer(q, biz, Path("x"), s)
    final, log = run(want)
    assert log["decision"] == "plan_agree" and final["answer"] == want
    final, log = run(want + 1)
    assert log["decision"] == "plan_conflict" and final["type"] == "abstain"


def test_buoc_cham_chan_dap_an_chi_khi_ket_luan_sai(monkeypatch):
    from datamate import agent as ag
    q = {"id": "q", "question": "câu lạ", "as_of_date": "2026-06-02", "business_id": "b"}
    biz = type("B", (), {"business_id": "b", "readme": "r", "profile": lambda self: "p", "tables": {}})()
    monkeypatch.setattr(ag, "run_code", lambda *a, **k: {"ok": True, "answer": {"type": "number", "answer": 7}})
    code = "```python\nANSWER = 1\n```"
    final, log = _agent(_FakeLLM(codes=[code, code], jsons=[{"dung": False, "ly_do": "sai kỳ"}]), judge=True).answer(q, biz, Path("x"))
    assert log["decision"] == "judge_rejected" and final["type"] == "abstain"
    final, log = _agent(_FakeLLM(codes=[code, code], jsons=[{"dung": True}]), judge=True).answer(q, biz, Path("x"))
    assert log["decision"] == "consensus" and final["answer"] == 7
    final, log = _agent(_FakeLLM(codes=[code, code], jsons=[None]), judge=True).answer(q, biz, Path("x"))
    assert log["decision"] == "consensus"  # không đọc được phán quyết thì không chặn


def test_phan_loai_loi_de_sua():
    from datamate.repair import classify
    from types import SimpleNamespace
    biz = SimpleNamespace(tables={"T": pd.DataFrame({"Thành tiền": [1]})})
    kind, tip = classify("KeyError: 'Thanh tien'", biz)
    assert kind == "SAI_TEN_COT" and "Thành tiền" in tip
    assert classify("TypeError: Invalid comparison between dtype=datetime64 and date")[0] == "SO_SANH_NGAY"
    assert classify("ZeroDivisionError: division by zero")[0] == "CHIA_0"
    assert classify("điều gì đó lạ")[0] == "KHAC"


def test_doc_json_co_the_kem_think():
    from datamate.llm import parse_json
    assert parse_json('<think>a {b}</think>{"x": 1}') == {"x": 1}
    assert parse_json("không có") is None


def test_dummy_khong_doi_duong_ong():
    from datamate.llm import DummyLLM
    assert DummyLLM().chat_json([], {}) is None



def _vote_agent(monkeypatch, answers, **cfg):
    """Mỗi phiếu lấy lần lượt một đáp án số trong answers; None là code lỗi."""
    from datamate import agent as ag
    seq = list(answers)
    def fake(*a, **k):
        v = seq.pop(0)
        return {"ok": False, "error": "x"} if v is None else {"ok": True, "answer": {"type": "number", "answer": v}}
    monkeypatch.setattr(ag, "run_code", fake)
    code = "```python\nANSWER = 1\n```"
    return _agent(_FakeLLM(codes=[code] * 20), **cfg), seq


_Q = {"id": "q", "question": "câu lạ", "as_of_date": "2026-06-02", "business_id": "b"}
_BIZ = type("B", (), {"business_id": "b", "readme": "r", "profile": lambda self: "p", "tables": {}})()


def test_duong_them_du_phieu_va_dan_xa_moi_nop(monkeypatch):
    a, seq = _vote_agent(monkeypatch, [1, 2, 1, 1], escalate_paths=["p3", "quote", "sql"])
    final, log = a.answer(_Q, _BIZ, Path("x"))
    assert log["decision"] == "escalated" and final["answer"] == 1 and not seq  # 3 phiếu dẫn 2
    a, seq = _vote_agent(monkeypatch, [1, 2, 1, 2, 3], escalate_paths=["p3", "quote", "sql"])
    final, log = a.answer(_Q, _BIZ, Path("x"))
    assert final["type"] == "abstain" and log["decision"] == "no_consensus"


def test_duong_them_khong_doi_cau_da_dong_y_va_bo_qua_model_chua_bat(monkeypatch):
    a, seq = _vote_agent(monkeypatch, [5, 5, 9], escalate_paths=["m2_a", "p3"])
    final, log = a.answer(_Q, _BIZ, Path("x"))
    assert log["decision"] == "consensus" and final["answer"] == 5 and seq == [9]
    a, seq = _vote_agent(monkeypatch, [1, 2], escalate_paths=["m2_a", "m2_b"])
    final, log = a.answer(_Q, _BIZ, Path("x"))
    assert final["type"] == "abstain" and {v["path"] for v in log["votes"]} == {"base0", "base1"}


def test_thu_moi_phieu_giu_nguyen_quyet_dinh_hai_phieu_dau(monkeypatch):
    a, seq = _vote_agent(monkeypatch, [5, 5, 7, 7], escalate_paths=["p3", "sql"], collect_all=True)
    final, log = a.answer(_Q, _BIZ, Path("x"))
    assert log["decision"] == "consensus" and final["answer"] == 5 and len(log["votes"]) == 4 and not seq


def test_bac_dap_an_khi_du_phieu_cung_ra_dap_an_khac(monkeypatch):
    a, _ = _vote_agent(monkeypatch, [5, 5, 7, 7], consensus_guard=True, guard_paths=["p3", "sql"])
    final, log = a.answer(_Q, _BIZ, Path("x"))
    assert log["decision"] == "consensus_rejected" and final["type"] == "abstain"
    a, _ = _vote_agent(monkeypatch, [5, 5, 7, 8], consensus_guard=True, guard_paths=["p3", "sql"])
    final, log = a.answer(_Q, _BIZ, Path("x"))
    assert log["decision"] == "consensus" and final["answer"] == 5


def test_ke_hoach_theo_so_cau_khong_theo_dong_ho():
    from datamate.agent import budget_paths
    cfg = {"escalate_paths": ["p3", "quote", "sql", "think"], "template_guard": True, "budget": {"target_s": 14400, "calls_per_s": 0.5}}
    assert budget_paths(1000, cfg) == {"paths": ["p3", "quote", "sql", "think"], "template_guard": True}
    few = budget_paths(5000, cfg)
    assert few == {"paths": [], "template_guard": False}  # 1,44 lượt mỗi câu: chỉ đủ hai phiếu đầu
    assert budget_paths(1000, cfg) == budget_paths(1000, cfg)


def test_dong_luat_doi_phan_cuoi_thi_chuyen_cho_model(monkeypatch):
    from datamate import templates_ext as te
    book = (ROOT / "data/train/rulebook.md").read_text(encoding="utf-8")
    monkeypatch.setattr(te, "RULEBOOK", book)
    assert all(te.rule_ok(t) for t in te.RULE_OF)
    line = te._RULE_LINES[te.RULE_OF["stock"]]
    monkeypatch.setattr(te, "RULEBOOK", book.replace(line, line[:-3] + " chưa giao |"))
    assert not te.rule_ok("stock") and te.rule_ok("revenue")


def test_sql_trong_sandbox():
    import pickle
    from types import SimpleNamespace
    from datamate.sandbox import _sql_helper
    t = {"Đơn hàng": pd.DataFrame({"Mã": ["A", "B"], "Tiền": [10.0, 20.0], "Ngày": pd.to_datetime(["2025-01-02", "2025-02-03"])})}
    sql = _sql_helper(t, pd)
    assert sql('SELECT SUM("Tiền") AS s FROM "Đơn hàng"').s[0] == 30.0
    assert sql("""SELECT "Mã" AS m FROM "Đơn hàng" WHERE "Ngày" >= '2025-02-01'""").m.tolist() == ["B"]
    assert t["Đơn hàng"]["Ngày"].dtype.kind == "M"  # T không đổi
