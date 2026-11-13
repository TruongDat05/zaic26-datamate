"""data/synth_source và công cụ sinh dữ liệu: nguồn đúng quy tắc chia shop, oracle/sandbox dùng được, biến thể luật bị chặn ở oracle,
eval/mine_errors.py dựng lại đúng nguyên văn phản hồi sửa lỗi mà Agent đã gửi và xếp nhóm kết quả đúng."""
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from datamate import __version__, loader  # noqa: E402
from datamate.agent import Agent  # noqa: E402
from datamate.run import cache_path, make_fewshot  # noqa: E402

DATA = ROOT / "data" / "train"
CACHE = ROOT / "outputs" / "cache"
SRC = ROOT / "data" / "synth_source"
QID = "q_158b312e8f5b"  # biz_118, "dthu tháng 6/2025 bao nhiêu", đáp án 1523470700
GOOD = ("s = T['So_chi_tiet_ban_hang']\nbad = set(s.loc[s['Trạng thái'].map(dm.norm_status).isin(['huy', 'tra_hang']), 'Mã đơn hàng'])\n"
        "ok = s[~s['Mã đơn hàng'].isin(bad)]\nANSWER = dm.money(ok[dm.month(2025, 6).mask(ok['Ngày'])]['Thành tiền'].sum())")
need_source = pytest.mark.skipif(not (SRC / "splits.json").exists(), reason="chưa dựng data/synth_source (python eval/build_synth_source.py)")


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _q(qid):
    return next(q for q in _jsonl(DATA / "questions.jsonl") if q["id"] == qid)


class ScriptLLM:
    """Trả lời theo seed (seed = seed phiếu + lượt) và ghi lại tin nhắn cuối mà agent gửi ở mỗi lần gọi."""

    def __init__(self, script, default="```python\nANSWER = dm.abstain('x')\n```"):
        self.script, self.default, self.seen = script, default, {}

    def chat(self, messages, temperature, seed, thinking=False):
        self.seen[seed] = messages[-1]["content"]
        return self.script.get(seed, self.default)


def _run(tmp_path, q, script, name="run"):
    cfg = yaml.safe_load((ROOT / "datamate/config.yaml").read_text(encoding="utf-8"))
    cfg["agent"].update(templates=False, contract=False)
    biz_dir = DATA / "data" / q["business_id"]
    biz = loader.load_cached(biz_dir, CACHE)
    llm = ScriptLLM(script)
    agent = Agent(llm, cfg["agent"], (DATA / "rulebook.md").read_text(encoding="utf-8"), DATA / "calendar.csv", make_fewshot(cfg))
    _, log = agent.answer(q, biz, cache_path(biz_dir, CACHE))
    run = tmp_path / name
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "logs" / f"{q['id']}.json").write_text(json.dumps(log, ensure_ascii=False, default=str), encoding="utf-8")
    (run / "logs" / "run_meta.json").write_text(json.dumps({"code_version": __version__, "config": cfg}, ensure_ascii=False), encoding="utf-8")
    return run, llm, log


def _mine(run, tmp_path):
    import mine_errors
    out = tmp_path / ("mined_" + run.name)
    assert mine_errors.main([str(run), "--out", str(out), "--source", str(SRC)]) == 0
    return {k: _jsonl(out / f"{k}.jsonl") for k in ("questions", "positives", "negatives")}


def code(s):
    return f"```python\n{s}\n```"


@need_source
def test_feedback_rebuilt_verbatim(tmp_path):
    q = _q(QID)
    s0, s1 = 42, 1042  # seed phiếu base0, base1 (seed + 1000·i); lượt k dùng seed + k
    script = {s0: "Không có code ở đây", s0 + 1: code("ANSWER = dm.money(T['khong_co_bang']['x'].sum())"), s0 + 2: code(GOOD),
              s1: code("ANSWER = dm.money(0)"), s1 + 1: code("ANSWER = {'type': 'number', 'answer': 'abc'}"), s1 + 2: code(GOOD)}
    run, llm, log = _run(tmp_path, q, script)
    codes = [s.get("error_code") for s in log["steps"]]
    assert codes == ["NO_CODE_BLOCK", "EXECUTION_FAILED", None, "ZERO_RESULT", "FORMAT_INVALID", None], codes
    m = _mine(run, tmp_path)
    assert [x["group"] for x in m["questions"]] == ["E6"]
    pos = {p["path"]: p for p in m["positives"]}
    assert set(pos) == {"base0", "base1"} and all(p["is_repair"] and p["feedback_exact"] for p in pos.values())
    for path, seed in (("base0", s0), ("base1", s1)):
        turns = pos[path]["turns"]
        assert len(turns) == 3 and turns[-1]["user_feedback"] is None
        for k in range(2):  # phản hồi sau lượt k chính là tin nhắn cuối agent gửi ở lượt k + 1
            assert turns[k]["user_feedback"] == llm.seen[seed + k + 1], (path, k)
    assert "Loại lỗi: SAI_TEN_COT" in pos["base0"]["turns"][1]["user_feedback"]


@need_source
def test_feedback_type_mismatch(tmp_path):
    q = next(x for x in _jsonl(DATA / "questions.jsonl") if x["business_id"] == "biz_118" and "%" in x["question"])
    run, llm, log = _run(tmp_path, q, {42: code("ANSWER = dm.money(5)")})
    assert log["steps"][0]["error_code"] == "TYPE_MISMATCH"
    import mine_errors
    from show_trace import build
    t = build(json.loads((run / "logs" / f"{q['id']}.json").read_text(encoding="utf-8")), None, 42)
    first = t["paths"]["base0"]["turns"][0]
    assert mine_errors.feedback(first, q, loader.load_cached(DATA / "data" / "biz_118", CACHE), True) == llm.seen[43]


@need_source
def test_groups(tmp_path):
    q = _q(QID)
    wrong = {s: code("ANSWER = dm.money(123)") for s in (42, 1042)}
    m = _mine(_run(tmp_path, q, wrong, "wrong")[0], tmp_path)
    assert m["questions"][0]["group"] == "E1" and len(m["negatives"]) == 2 and not m["positives"]
    split = {42: code(GOOD), 1042: code("ANSWER = dm.money(123)")}
    m = _mine(_run(tmp_path, q, split, "split")[0], tmp_path)
    assert m["questions"][0]["group"] == "E2" and len(m["positives"]) == 1 and len(m["negatives"]) == 1
    none = {s + k: "không có code" for s in (42, 1042) for k in range(3)}
    m = _mine(_run(tmp_path, q, none, "none")[0], tmp_path)
    assert m["questions"][0]["group"] == "E4" and set(m["questions"][0]["step_codes"]) == {"NO_CODE_BLOCK"}  # early_stop: có thể chỉ chạy phiếu đầu
    assert m["questions"][0]["role"] == "tune" and not m["questions"][0]["usable_as_seed"]  # biz_118 thuộc tập tune


@need_source
def test_source_integrity():
    sp = json.loads((SRC / "splits.json").read_text(encoding="utf-8"))
    groups = [set(sp[k]) for k in ("mining", "tune", "holdout")]
    assert sum(map(len, groups)) == 30 and len(set().union(*groups)) == 30
    assert set(sp["holdout"]) == set((ROOT / "eval/holdout_businesses.txt").read_text().split())
    seeds = _jsonl(SRC / "seeds/questions.jsonl")
    assert len(seeds) == 3000 and all(s["usable_as_seed"] == (s["role"] == "mining") for s in seeds)
    assert all(g["role"] in ("tune", "holdout") for g in _jsonl(SRC / "guard/eval_questions.jsonl"))
    assert all(not b["usable_for_training"] or (b["verified"] and b["role"] == "mining") for b in _jsonl(SRC / "seeds/code_bank.jsonl"))
    rb = (DATA / "rulebook.md").read_text(encoding="utf-8")
    assert all((v["old"] if v["kind"] == "replace" else v["anchor"]) in rb for v in _jsonl(SRC / "rules/counterfactual.jsonl"))
    tmpl = {t["template"] for t in _jsonl(SRC / "taxonomy/templates.jsonl")}
    assert {s["route"]["template"] for s in seeds if s["route"]["template"]} <= tmpl
    profiles = {p["business_id"]: p for p in _jsonl(SRC / "shops/index.jsonl")}
    assert len(profiles) == 30 and all((SRC / "shops" / f"{b}.md").exists() for b in profiles)


@need_source
def test_kit_oracle_execute_prompt():
    from synth_tools import Kit
    kit = Kit()
    q, gold = _q(QID), {"type": "number", "answer": 1523470700}
    a = kit.oracle(q["business_id"], q["question"], q["as_of_date"])
    assert a is not None and kit.check(a, gold) == 1
    r = kit.execute(q["business_id"], GOOD, q["as_of_date"], question=q["question"])
    assert r["ok"] and kit.check(r["answer"], gold) == 1 and r["issues"] == []
    msgs = kit.prompt(q)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "## 1. Đơn hàng và doanh thu" in msgs[0]["content"] and q["question"] in msgs[1]["content"]
    assert kit.shop_block(q["business_id"]).strip() in msgs[1]["content"]


@need_source
def test_counterfactual_blocks_oracle():
    from datamate import templates_ext
    from synth_tools import Kit
    kit = Kit()
    rb = kit.rulebook(["cf_tuan_chu_nhat"])
    assert "bắt đầu **Chủ nhật**" in rb and "bắt đầu **thứ Hai**" not in rb
    before = templates_ext.RULEBOOK
    week = next(s for s in kit.seeds() if "tuần" in s["question"].lower() and s["route"]["template"] and s["route"]["template_correct"])
    assert kit.oracle(week["business_id"], week["question"], week["as_of_date"]) is not None
    assert kit.oracle(week["business_id"], week["question"], week["as_of_date"], ["cf_tuan_chu_nhat"]) is None
    other = next(s for s in kit.seeds() if s["business_id"] == week["business_id"] and "tuần" not in s["question"].lower()
                 and "tuan" not in s["question"].lower() and s["route"]["template"] and s["route"]["template_correct"])
    a = kit.oracle(other["business_id"], other["question"], other["as_of_date"], ["cf_tuan_chu_nhat"])  # câu đối chứng vẫn có oracle
    assert a is not None and kit.check(a, other["gold"]) == 1
    assert kit.oracle(other["business_id"], other["question"], other["as_of_date"], ["cf_dang_giao_khong_tinh"]) is None
    kit.prompt({"id": "x", "business_id": week["business_id"], "question": week["question"], "as_of_date": week["as_of_date"]}, variants=["cf_tuan_chu_nhat"])
    assert templates_ext.RULEBOOK == before  # dựng prompt với biến thể không làm đổi rulebook của oracle


@need_source
def test_contaminated():
    from synth_tools import Kit
    kit = Kit()
    g = _jsonl(SRC / "guard/eval_questions.jsonl")[0]
    assert kit.contaminated(g["question"])["hit"]
    assert not kit.contaminated("tổng tiền điện nước của tiệm năm ngoái là bao nhiêu nhỉ")["hit"]


SPEC = SRC / "data_spec"


@pytest.mark.skipif(not (SPEC / "calibration.json").exists(), reason="chưa dựng data_spec (python eval/build_data_spec.py)")
def test_data_spec():
    import synth_spec
    cal = json.loads((SPEC / "calibration.json").read_text(encoding="utf-8"))
    assert {"tap_hoa", "ca_phe", "vlxd", "nha_thuoc", "quan_nhau"} <= set(cal["theo_nganh"]) and len(cal["theo_shop"]) == 30
    arch = json.loads((SPEC / "archetypes.json").read_text(encoding="utf-8"))
    assert set(arch) == {c for c, _, _ in synth_spec.ARCHETYPES} == set(synth_spec.RENDER_RECIPES)
    cols = json.loads((SPEC / "column_names.json").read_text(encoding="utf-8"))["theo_vai_tro"]
    assert {"order", "date", "status", "item", "amount", "qty", "cust"} <= set(cols)
    fmt = json.loads((SPEC / "value_formats.json").read_text(encoding="utf-8"))
    assert {"hoan_thanh", "huy", "tra_hang"} <= set(fmt["trang_thai"]) and fmt["khach_le"]
    schema = json.loads((SPEC / "canonical_schema.json").read_text(encoding="utf-8"))
    assert {"orders", "order_lines", "items", "units", "returns", "payments"} <= set(schema)
    import re
    prompt = re.sub(r"<!--.*?-->", "", (SRC / "prompts" / "7_danh_muc_nganh_moi.md").read_text(encoding="utf-8"), flags=re.S)
    assert not any(t in prompt.lower() for ts in synth_spec.LOCAL_BLOCKLIST.values() for t in ts)  # không gửi danh sách chặn ra API


def test_data_spec_plan_and_complexity():
    import synth_spec
    cx = json.loads((SPEC / "shop_complexity.json").read_text(encoding="utf-8"))
    assert len(cx["theo_shop"]) == 30 and sum(cx["so_shop_theo_tang"].values()) == 30
    for f in ("new_industries.json", "volume_plan.json", "model_routing.json"):
        assert (SPEC / f).exists()
    codes = [n["code"] for n in synth_spec.NEW_INDUSTRIES]
    assert len(codes) == len(set(codes)) and not set(codes) & set(synth_spec.INDUSTRY_NAMES)
    block = [t for ts in synth_spec.LOCAL_BLOCKLIST.values() for t in ts]
    assert not any(t in (n["ten"] + " " + n["dac_trung"]).lower() for n in synth_spec.NEW_INDUSTRIES for t in block)
    provs = {m.split(":")[0] for r in synth_spec.MODEL_ROUTING for m in r["model"]}
    import llm
    assert provs <= set(llm.PROVIDERS)


# ---------------------------------------------------------------- log cũ (v4.2), log thật, gọi API, kiểm đầu ra teacher
def test_show_trace_old_log_candidates():
    import show_trace
    r = {"id": "q1", "steps": [{"seed": 42, "turn": 0, "response": "", "result": {"ok": True, "answer": {"type": "number", "answer": 1}}},
                               {"seed": 1042, "turn": 0, "response": "", "result": {"ok": False, "error": "x"}, "error_code": "EXECUTION_FAILED"}],
         "candidates": [{"id": "q1", "type": "number", "answer": 1}, None], "decision": "no_candidate", "final": {"type": "abstain"}}
    t = show_trace.build(r, {"q1": {"id": "q1", "type": "number", "answer": 1}}, 42)
    assert t["paths"]["base0"]["answer"] == {"type": "number", "answer": 1} and t["paths"]["base0"]["mark"] == 1
    assert t["paths"]["base1"]["answer"] is None


def test_ingest_stats_only_has_no_questions(tmp_path):
    import ingest_real_logs as ing
    run = tmp_path / "real" / "pub"
    (run / "logs").mkdir(parents=True)
    (run / "logs/run_meta.json").write_text(json.dumps({"data_dir": "/content/khong_co"}), encoding="utf-8")
    (run / "logs/q_1.json").write_text(json.dumps({"id": "q_1", "question": "BÍ_MẬT", "steps": [{"error_code": "EXECUTION_FAILED"}],
                                                   "decision": "no_candidate", "final": {"type": "abstain"}, "llm_calls": 2}), encoding="utf-8")
    res = ing.ingest(run, tmp_path / "src", tmp_path / "real")
    assert res["loai"] == "chi_thong_ke"
    out = tmp_path / "src/errors/pub"
    assert sorted(p.name for p in out.iterdir()) == ["summary.json"] and "BÍ_MẬT" not in (out / "summary.json").read_text(encoding="utf-8")


def test_real_logs_ingested():
    idx = SRC / "errors/index.json"
    if not idx.exists():
        pytest.skip("chưa chạy eval/ingest_real_logs.py")
    for x in json.loads(idx.read_text(encoding="utf-8")):
        d = SRC / "errors" / x["run"]
        if x["loai"] == "chi_thong_ke":
            assert not (d / "questions.jsonl").exists()
        else:
            qs = [json.loads(l) for l in (d / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
            assert all(q["usable_as_seed"] == (q["role"] == "mining") for q in qs)


def test_llm_helpers_offline():
    import llm
    assert llm.endpoint("https://a.b/v1") == "https://a.b/v1/chat/completions"
    assert llm.endpoint("https://a.b/chat/completions") == "https://a.b/chat/completions"
    assert llm.extract_json('chú thích\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert llm.extract_json('{"r": ["| A | b\nc |"]}') == {"r": ["| A | b\nc |"]}
    assert llm.extract_code("x\n```python\nanswer = 1\n```") == "answer = 1"
    with pytest.raises(llm.LeakError):
        llm.guard("dữ liệu shop biz_101")
    with pytest.raises(llm.LeakError):
        llm.guard("ngành sửa xe máy")
    llm.guard("tiệm hoa tươi bán bó hoa hồng 20/10")
    rb = (ROOT / "data/train/rulebook.md").read_text(encoding="utf-8")
    with pytest.raises(llm.LeakError):
        llm.guard("xin chép: " + " ".join(rb.split()[200:215]))


def test_synth_checks():
    import synth_checks
    item = {"name": "Vở 96 trang", "group": "Vở", "base_unit": "quyển", "big_units": [{"unit": "lốc", "factor": 10}],
            "price_vnd": [8000, 12000], "cost_ratio": 0.7, "popularity": 3}
    cat = {"industry": "nha_sach", "ten": "nhà sách", "items": [dict(item, name=f"Vở {i} trang", group=f"N{i % 6}", popularity=1 + i % 3)
                                                                for i in range(40)],
           "service_lines": [], "customer_groups": ["lẻ"], "channels": ["tại quầy"], "pay_methods": ["tiền mặt"], "open_hours": [7, 21],
           "orders_per_day": [30, 80], "lines_per_order": [2, 5],
           "seasonality": {"tet_multiplier": 1.2, "events": [{"ten": "tựu trường", "thang_ngay": "08-25", "he_so": 2.0}], "weekday": [1] * 7},
           "has": {k: False for k in synth_checks.HAS_KEYS}, "rules_section_6": ["| Doanh thu sách | không tính bao bì |", "| A | B |"]}
    assert synth_checks.check_catalog(cat, 40) == []
    bad = json.loads(json.dumps(cat))
    bad["items"][1]["name"] = "vo 0 trang"
    bad["seasonality"]["weekday"] = [1] * 6
    bad["ten"] = "mẹ và bé"
    assert {"trung_ten", "he_so_thu", "blocklist"} <= set(synth_checks.check_catalog(bad, 40))
    assert "thieu_mat_hang" in synth_checks.check_catalog(dict(cat, items=cat["items"][:10]), 40)


def test_bakeoff_toy_gold_and_harness(tmp_path):
    import pickle
    import model_bakeoff as mb
    T = mb.toy_tables()
    qs = mb._gold(T)
    assert [q["qid"] for q in qs] == ["q1_doanh_thu", "q2_ti_le_huy", "q3_top_khach", "q4_quy_doi", "q5_dt_thuan", "q6_tang_truong"]
    assert qs == mb._gold(mb.toy_tables())  # hạt cố định
    (tmp_path / "harness.py").write_text(mb.HARNESS, encoding="utf-8")
    pickle.dump(T, open(tmp_path / "t.pkl", "wb"))
    code = ("o = T['orders']; d = pd.to_datetime(o['order_date'], format='%d/%m/%Y')\n"
            "ok = o[o['status'].isin(['hoàn thành', 'Hoàn thành', 'xong']) & (d.dt.month == 3)]\n"
            "answer = int(T['order_lines'][T['order_lines']['order_id'].isin(ok['order_id'])]['amount'].sum())")
    ans, err = mb.run_code(code, tmp_path / "t.pkl", tmp_path, "q1")
    assert err is None and mb.same(ans, qs[0])
    ans, err = mb.run_code("answer = 1/0", tmp_path / "t.pkl", tmp_path, "q1b")
    assert ans is None and err.startswith("loi_chay")


def test_normalize_catalog_and_prompt7():
    import llm
    import model_bakeoff as mb
    import synth_checks
    import synth_spec
    raw = json.loads(json.dumps(synth_spec.CATALOG_EXAMPLE))
    raw["items"] = [dict(raw["items"][1], name=f"Lưỡi câu số {i}", group=f"G{i % 6}", popularity=str(1 + i % 3), cost_ratio="0.55",
                         price_vnd="[25.000, 30.000]", big_units=[{"unit": "hộp", "factor": "20"}, {"unit": "vỉ", "factor": 1}]) for i in range(40)]
    raw.update(open_hours="06:30-20:30", orders_per_day="[15, 40]")
    raw["seasonality"]["weekday"] = "0.8, 0.8, 0.9, 0.9, 1.1, 1.4, 1.5"
    raw["rules_section_6"] = ["| Luật | Mặc định |", "|---|---|"] + raw["rules_section_6"] + ["Khái niệm A | cách tính A"]
    assert synth_checks.hard_errors(synth_checks.check_catalog(raw, 40))
    norm, fixed = synth_checks.normalize_catalog(raw)
    assert synth_checks.check_catalog(norm, 40) == [] and {"gia", "gio_mo", "he_so_thu", "luat_nganh", "quy_doi"} <= set(fixed)
    assert norm["open_hours"] == [6.5, 20.5] and norm["items"][0]["price_vnd"] == [25000, 30000] and len(norm["rules_section_6"]) == 3
    assert not synth_checks.hard_errors(synth_checks.check_catalog(synth_spec.CATALOG_EXAMPLE, 2))  # ví dụ trong prompt đúng kiểu
    p = mb.prompt_t2("tiệm hoa tươi")
    assert "{" + "ten_nganh}" not in p and "<!--" not in p and "do_cau_ca" in p
    llm.guard(p)
    assert synth_checks.blocklist_hits("khách đặt qua điện thoại, số điện thoại 0909") == []
    assert synth_checks.blocklist_hits("cửa hàng điện thoại di động") == ["điện thoại"]


def test_bakeoff_toy_sets():
    """Hai bộ đồ chơi T4: hạt cố định, đủ bẫy (trạng thái/khách lẻ nhiều cách ghi, dòng âm, lô, nợ), số liệu tự nhất quán."""
    import llm
    import model_bakeoff as mb
    import synth_checks
    dev, ho = mb.bo("dev")["T"], mb.bo("holdout")["T"]
    assert all(dev[k].equals(mb.bo_dev()["T"][k]) for k in dev) and all(ho[k].equals(mb.bo_holdout()["T"][k]) for k in ho)
    assert dev["HoaDon"]["Trạng thái"].nunique() >= 8 and {"KL", "Khách lẻ"} <= set(dev["HoaDon"]["Khách hàng"]) and dev["HoaDon"]["Khách hàng"].isna().any()
    assert {"Đang giao", "KVL"} <= set(ho["don_ban"]["Tình trạng"]) | set(ho["don_ban"]["Khách"])
    xn = mb.bo("xac_nhan")  # bộ xác nhận: cùng khuôn holdout, hạt mới nên dữ liệu khác
    assert list(xn["T"]) == list(ho) and not xn["T"]["don_ban"].equals(ho["don_ban"]) and xn["quy_uoc"] == mb.bo("holdout")["quy_uoc"]
    assert (dev["ChiTietHD"]["SL"] < 0).any() and ho["dong_don"].loc[ho["dong_don"]["Số lượng"] < 0, "Ngày trả"].notna().all()
    assert dev["QuyDoiDVT"]["Mã thuốc"].duplicated().any()  # có hàng quy đổi nhiều cấp
    no = dev["SoCongNo"].groupby("Mã KH")[["Phát sinh nợ", "Đã thu"]].sum()
    assert (no["Phát sinh nợ"] >= no["Đã thu"]).all()  # không thu quá số nợ
    ct = dev["ChiTietHD"].merge(dev["HoaDon"], on="Số HĐ").merge(dev["QuyDoiDVT"], on=["Mã thuốc", "ĐVT"], how="left")
    ct = ct[~ct["Trạng thái"].map(synth_checks.fold).str.contains("huy")]
    ban = (ct["SL"] * ct["Quy ra ĐVT gốc"].fillna(1)).groupby(ct["Số lô"]).sum()
    assert (dev["LoThuoc"].set_index("Số lô")["SL nhập"].reindex(ban.index) >= ban).all()  # tồn lô không âm
    p = mb.prompt_t4(mb.bo("dev"))
    assert "{" + "mo_ta_bang}" not in p and "<!--" not in p and "'HT'" in p and "Paracetamol" in p
    llm.guard(p)
    llm.guard(mb.prompt_giai(mb.bo("holdout"), ["Doanh thu tháng 3/2025 là bao nhiêu?"]))


def test_bakeoff_same_answer():
    """Chấm như bộ chấm BTC sau formatter: cùng kiểu khai báo, làm tròn như pipeline, khác dạng là sai."""
    import model_bakeoff as mb
    assert mb.same_answer(614944.25, 614944, "number", "Giá trị trung bình mỗi đơn là bao nhiêu tiền?")  # tiền: làm tròn về đồng
    assert not mb.same_answer(614944.25, 614944, "number", "Trung bình mỗi đơn có mấy sản phẩm?")  # không phải tiền: giữ 2 chữ số
    assert mb.same_answer(-9.7249, -9.72, "percent") and not mb.same_answer(-9.09, -9.72, "percent")
    assert not mb.same_answer("Gạo thơm túi 5kg", ["Gạo thơm túi 5kg"], "list") and mb.same_answer(["bia lon 330ml"], ["Bia lon 330ml"], "list")
    assert mb.same_answer("2025-03-05", "2025-03-05", "date") and not mb.same_answer("2025-03-05 00:00:00", "2025-03-05", "date")
    assert mb.same_answer(True, True, "boolean") and not mb.same_answer(1, True, "boolean") and not mb.same_answer(True, 1, "number")
    assert not mb.same_answer(["A", "B"], ["B", "A"], "list") and not mb.same_answer(None, 1, "number")
    assert mb.same_answer("Bùi Thị Bình", "bùi thị  bình", "string") and mb.typed([], "list") is None
    assert mb.tach_khoi("```python\n# cau 2\nanswer = 2\n```\n```python\n# câu 1\nanswer = 1\n```", 3) == ["# câu 1\nanswer = 1\n", "# cau 2\nanswer = 2\n", None]
    assert mb._doc_lo('```json\n[{"question": "a"}]\n```')[0] == [{"question": "a"}] == mb._doc_lo('{"mau": [{"question": "a"}]}')[0]
    assert mb._doc_lo('{"question": "a"},\n{"question": "b"}')[0] is None and mb._doc_lo('{"error": "x"}')[1]  # không đúng khuôn → lượt sửa JSON


def test_t4_question_checks():
    import pandas as pd
    import synth_checks as sc
    assert sc.ky_ro("Doanh thu tháng 3/2025?") and sc.ky_ro("nợ đến hết ngày 30/6/2025") and sc.ky_ro("quý 2 năm 2025") and not sc.ky_ro("doanh thu tháng 3")
    assert sc.lo_dap_an("Doanh thu tháng 3/2025 có phải 134.082.100 không?", {"type": "number", "answer": 134082100})
    assert not sc.lo_dap_an("Có bao nhiêu đơn tháng 3/2025?", {"type": "number", "answer": 2025})  # năm của kỳ không phải đáp án
    assert sc.lo_dap_an("Tỉ lệ huỷ 12,35% của tháng 3/2025?", {"type": "percent", "answer": 12.35})
    assert sc.lo_dap_an("Ai mua nhiều hơn: Bùi Thị Bình hay Lê Văn An?", {"type": "string", "answer": "Bùi Thị Bình"})
    assert sc.ma_noi_bo("Khách KH05 tháng 3/2025 nợ bao nhiêu? (khách KL)", {"kh05", "kl"}) == ["kh05", "kl"]
    assert sc.codes_of({"a": pd.DataFrame({"x": ["KH01", "KL", "Khách lẻ", None], "y": [1, 2, 3, 4]})}) == {"kh01", "kl"}
    assert len(sc.goi_y("Top 3 khách (không tính khách lẻ) tháng 3/2025, chỉ tính đơn hoàn thành, quy về đơn vị cơ bản")) == 3
    assert sc.goi_y("Khách lẻ chiếm bao nhiêu % doanh thu tháng 3/2025?") == []


def test_style_layer():
    """Lớp văn phong cục bộ: telex như câu train, giữ số/kỳ, tỉ lệ từng nét sát style_profile.json (±0,05 trên 120 câu)."""
    import random
    import build_synth_source as b
    from stress_noise import VanPhong, style_question, telex
    assert [telex(w) for w in ("tiền", "được", "tháng", "cửa")] == ["tieenf", "dduwowcj", "thangs", "cuwar"]
    vec = {"no_diacritic": True, "abbreviations": ["dthu", "bnhiu"], "fillers": [], "telex": [], "lower_start": True, "no_qmark": True}
    assert style_question("Doanh thu tháng 3/2025 là bao nhiêu?", vec, random.Random(0), {"dthu": 1}, {"vậy": 1}) == "dthu thang 3/2025 la bnhiu"
    vp = VanPhong(seed=0)
    qs = ["Doanh thu tháng 3/2025 của nhóm Kháng sinh là bao nhiêu?", "Trong tháng 5/2025 có bao nhiêu hoá đơn bị huỷ?",
          "Ba khách hàng mua nhiều tiền nhất quý 1/2025 là ai, xếp từ cao xuống thấp?", "Tỉ lệ hoá đơn huỷ trong tháng 2/2025 là bao nhiêu phần trăm?",
          "Ngày nào trong tháng 4/2025 có doanh thu cao nhất?", "Tháng 6/2025 có bán được hộp Paracetamol nào không?"] * 20
    out = [vp.render(q) for q in qs]
    st = [VanPhong.net(b.style_of(s)) for s in out]
    assert all(abs(sum(x[t] for x in st) / len(st) - vp.prof["ti_le"][t]) <= 0.05 for t in VanPhong.NET)
    assert all(q.split("/2025")[0].split()[-1] in s for q, s in zip(qs, out))  # số của kỳ còn nguyên


T4_CASES = [{"question": "Doanh thu tháng 1/2025 là bao nhiêu?", "loai": "number", "do_kho": 1, "code": "answer = 1500000"},
            {"question": "Tháng 2/2025 bán được bao nhiêu viên Paracetamol 500mg?", "loai": "number", "do_kho": 2, "code": "answer = 1/0"},
            {"question": "Cho biết doanh thu của tháng 1/2025 nhé", "loai": "number", "do_kho": 1, "code": "answer = 1500000.0"},
            {"question": "Ngày nào trong tháng 3/2025 có doanh thu cao nhất?", "loai": "date", "do_kho": 2, "code": "answer = pd.Timestamp('2025-03-05')"},
            {"question": "Doanh thu thuần tháng 4/2025 của nhóm Kháng sinh là bao nhiêu?", "loai": "number", "do_kho": 3, "code": "answer = 2345678"},
            {"question": "Ba thuốc bán chạy nhất tháng 5/2025 là gì, xếp từ cao xuống thấp?", "loai": "list", "do_kho": 2, "code": "answer = ['A', 'B', 'C']"},
            {"question": "Tỉ lệ hoá đơn huỷ tháng 6/2025 là bao nhiêu phần trăm, không tính khách lẻ?", "loai": "percent", "do_kho": 2,
             "code": "answer = 3.456"}]
T4_GIAI = {1: 1500000, 2: 640, 4: 2345678, 5: ["A", "B", "C"], 6: 3.46}  # đáp án đúng theo tháng trong câu


def _fake_t4(provider, messages, think="off", max_tokens=0, tag="", **kw):
    """Model giả cho lô T4: soạn T4_CASES; sửa mẫu 2; model yếu sai câu khó, model giải chính trả chuỗi cho câu list, câu viết lại tháng 6 bị hiểu sai."""
    import re as _re
    text = messages if isinstance(messages, str) else messages[-1]["content"]
    if tag.endswith("soan"):
        out = json.dumps([dict(c, ky_nang=["doanh_thu_ky"], spec="x") for c in T4_CASES])
    elif tag.endswith("sua"):
        assert "mẫu 2" in text and "mẫu 4" in text and "ZeroDivisionError" in text
        out = json.dumps([dict(T4_CASES[1], i=2, code="answer = 640")])
    else:
        ans = []
        for q in _re.findall(r"^\d+\. (.*)$", text.split("Câu hỏi:\n")[-1], _re.M):
            m = int(_re.search(r"(\d+)/2025", q).group(1))
            ans.append(2000000 if tag.endswith("yeu") and m == 4 else "A" if provider == "gpt" and m == 5 and tag.endswith("giai")
                       else 9.99 if tag.endswith("vp") and m == 6 else T4_GIAI[m])
        out = "\n".join(f"```python\n# cau {k}\nanswer = {a!r}\n```" for k, a in enumerate(ans, 1))
    return {"content": out, "usage": {"total_tokens": 100}, "latency_s": 0.1, "cost_usd": 0.001 if provider in ("qwen", "deepseek") else None,
            "finish": "stop", "provider": provider, "reasoning": ""}


def test_bakeoff_t4_offline(tmp_path, monkeypatch):
    """Một lô T4 với model giả: sửa một lượt, trùng, sai dạng ngày, khó thật, model giải chính lệch dạng, câu viết lại lệch nghĩa bị bỏ."""
    import pickle
    import model_bakeoff as mb
    from stress_noise import VanPhong
    b = mb.bo("dev")
    (tmp_path / "harness.py").write_text(mb.HARNESS, encoding="utf-8")
    pickle.dump(b["T"], open(tmp_path / "dev.pkl", "wb"))
    monkeypatch.setattr(mb.llm, "chat", _fake_t4)
    j = {"task": "T4", "provider": "gemini", "think": "high", "sample": 0, "key": "dev_lo12", "bo": "dev", "prompt": mb.prompt_t4(b), "pv": "vX"}
    rec = mb.t4_batch(j, tmp_path, VanPhong(seed=0, known=mb._known("dev")))
    by = {c["i"]: c for c in rec["cases"]}
    assert rec["solvers"] == ["gpt:off", "qwen:off"] and rec["weak"] == "deepseek:off" and rec["sua"] == "code" and rec["loi_lo"] == ["thieu_mau"]
    assert by[1]["sua"] and not by[1]["errors"] and by[2]["errors"] == ["trung"] and by[3]["errors"] == ["sai_loai"]
    assert by[4]["that_kho"] and by[5]["errors"] == ["giai_lai_lech"] and by[6]["canh_bao"] == ["goi_y"] and by[6]["vp"]["lan"] == 2
    assert (rec["n_valid"], rec["n_khop_ca_hai"], rec["n_that_kho"], rec["n_dung_duoc"], rec["n_sua_dat"]) == (4, 4, 1, 3, 1)
    again = mb.judge_t4(json.loads(json.dumps(rec, default=str)))  # chấm lại từ bản ghi đã lưu ra cùng kết quả
    assert (again["n_valid"], again["n_that_kho"], again["n_dung_duoc"]) == (4, 1, 3)
    row = mb.mau_finetune(by[0], rec, b)
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"] and row["answer"] == 1500000 and row["type"] == "number"
    assert row["question_goc"] == T4_CASES[0]["question"] and row["question"] in row["messages"][1]["content"]
    s = mb.t4_summary([mb.ban_ghi_t4(rec) | {"run": "x"}])
    assert s["mau_hop_le"] == "4/12" and s["kho_that"] == 1 and s["kiem_nghia_vp"] == "3/4" and s["sua"]["mau_sua_hop_le"] == 1
    old = {"task": "T4", "provider": "gpt", "think": "high", "sample": 0, "key": "lo_8_case", "pv": "v1", "api_ok": True, "errors": [],
           "giai_lai_boi": ["qwen", "deepseek"], "cases": [{"i": 0, "question": "Mặt hàng nào bán chạy nhất tháng 7/2025?", "do_kho": 3,
           "ky_nang": [], "loai": "list", "answer": ["Bia"], "errors": [], "giai_lai": {"qwen": {"answer": "Bia", "loi": None},
                                                                                         "deepseek": {"answer": ["Bia"], "loi": None}}}]}
    assert mb.judge_t4(mb._t4_v1(old))["cases"][0]["errors"] == ["giai_lai_lech"]  # v1 chấm lỏng: chuỗi khớp list một phần tử


@need_source
def test_gen_data_offline(tmp_path, monkeypatch):
    """Luồng sinh chính thức với model giả: đọc shop (tables.pkl + rules.md), lô hai lặp lại lô một nên mẫu bị bỏ vì trùng giữa lô."""
    import pickle
    import gen_data
    import model_bakeoff as mb
    shop = tmp_path / "shops" / "s1"
    shop.mkdir(parents=True)
    pickle.dump(mb.bo("dev")["T"], open(shop / "tables.pkl", "wb"))
    (shop / "rules.md").write_text(mb.QUY_UOC_DEV, encoding="utf-8")
    monkeypatch.setattr(mb.llm, "chat", _fake_t4)
    assert gen_data.main(["--shops", str(tmp_path / "shops"), "--dry-run"]) == 0
    assert gen_data.main(["--shops", str(tmp_path / "shops"), "--batches", "2", "--out", str(tmp_path / "out")]) == 0
    rows = _jsonl(tmp_path / "out/raw_samples.jsonl")
    res = json.loads((tmp_path / "out/tong_hop.json").read_text(encoding="utf-8"))
    assert len(rows) == res["mau_giu"] == 3
    assert res["bo_qua"] == {"vp_lech_nghia": 4, "trung_giua_lo": 9, "thieu_mau_sau_fallback": 21}
    assert not (tmp_path / "out/train.jsonl").exists()  # T4 never claims to be runtime SFT
    assert all(r["shop"] == "s1" and r["messages"][2]["content"].startswith("```python") and r["question_goc"] for r in rows)
    assert '10/12/2026' not in mb.bo("s1")["khoang"]  # expiry cannot extend transaction range


def test_style_features():
    import build_synth_source as b
    import synth_spec
    assert synth_spec.telex_words("thangs 1/2026 bán đc nhiêu tieenf", frozenset()) == ["thangs", "tieenf"]
    assert synth_spec.telex_words("tong dthu thang 3 bnhiu z", frozenset()) == []
    assert synth_spec.telex_words("Sơn Dulux 5L bán bao nhiêu", frozenset({"dulux"})) == []
    s = b.style_of("tong dthu thang 3 bnhiu z")
    assert s["no_diacritic"] and s["lower_start"] and s["no_qmark"] and "dthu" in s["abbreviations"] and not s["telex"]
    prof = json.loads((SRC / "style/style_profile.json").read_text(encoding="utf-8"))
    assert {"khong_dau", "co_viet_tat", "co_tu_dem", "go_telex", "chu_thuong_dau", "khong_dau_hoi"} <= set(prof["ti_le"])
