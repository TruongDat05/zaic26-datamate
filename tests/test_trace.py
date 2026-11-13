"""eval/show_trace.py dựng lại được luồng trả lời từ log của Agent: phiếu, lượt sửa lỗi, đáp án, prompt."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from datamate import __version__, loader  # noqa: E402
from datamate.agent import Agent  # noqa: E402
from datamate.run import cache_path, make_fewshot  # noqa: E402
import show_trace as trace_mod  # noqa: E402

DATA = ROOT / "data" / "train"
CACHE = ROOT / "outputs" / "cache"
QID = "q_158b312e8f5b"  # doanh thu tháng 6/2025 của biz_118
GOOD = ("s = T['So_chi_tiet_ban_hang']\nbad = set(s.loc[s['Trạng thái'].map(dm.norm_status).isin(['huy', 'tra_hang']), 'Mã đơn hàng'])\n"
        "ok = s[~s['Mã đơn hàng'].isin(bad)]\nANSWER = dm.money(ok[dm.month(2025, 6).mask(ok['Ngày'])]['Thành tiền'].sum())")


class FakeLLM:
    def chat(self, messages, temperature, seed, thinking=False):
        if messages[-1]["content"].startswith("Code lỗi"):
            return f"Sửa lại:\n```python\n{GOOD}\n```"
        return "```python\nANSWER = dm.money(T['khong_co_bang']['x'].sum())\n```"


def test_trace_rebuilds_flow(tmp_path):
    cfg = yaml.safe_load((ROOT / "datamate/config.yaml").read_text(encoding="utf-8"))
    cfg["agent"].update(templates=False, contract=False)
    biz_dir = DATA / "data" / "biz_118"
    biz = loader.load_cached(biz_dir, CACHE)
    q = next(json.loads(l) for l in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines() if QID in l)
    agent = Agent(FakeLLM(), cfg["agent"], (DATA / "rulebook.md").read_text(encoding="utf-8"), DATA / "calendar.csv", make_fewshot(cfg))
    final, log = agent.answer(q, biz, cache_path(biz_dir, CACHE))
    assert log["decision"] == "consensus" and [s["path"] for s in log["steps"]] == ["base0", "base0", "base1", "base1"]

    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / f"{QID}.json").write_text(json.dumps(log, ensure_ascii=False, default=str), encoding="utf-8")
    (run / "logs" / "run_meta.json").write_text(json.dumps({"code_version": __version__, "config": cfg}, ensure_ascii=False), encoding="utf-8")

    r = json.loads((run / "logs" / f"{QID}.json").read_text(encoding="utf-8"))
    gold = {QID: next(json.loads(l) for l in (DATA / "answers.jsonl").read_text(encoding="utf-8").splitlines() if QID in l)}
    t = trace_mod.build(r, gold, 42)
    assert list(t["paths"]) == ["base0", "base1"] and t["mark"] == 1
    first, fixed = t["paths"]["base0"]["turns"]
    assert not first["ok"] and "khong_co_bang" in first["error"] and fixed["ok"] and fixed["code"].strip() == GOOD
    assert t["paths"]["base0"]["mark"] == 1

    # log cũ chưa có path: suy từ seed
    for s in r["steps"]:
        s.pop("path")
    assert list(trace_mod.build(r, None, 42)["paths"]) == ["base0", "base1"]

    prompts = trace_mod.Prompts(run, DATA, CACHE, cfg)
    text = trace_mod.render_text(t, prompts, json.loads((run / "logs" / f"{QID}.json").read_text(encoding="utf-8")))
    assert "phiếu base0" in text and "[ĐÚNG]" in text and "Câu hỏi:" in text and "prompt người dùng" in text

    out = tmp_path / "t.html"
    assert trace_mod.main([str(run), "--gold", str(DATA / "answers.jsonl"), "--data", str(DATA), "--cache", str(CACHE), "--html", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert QID in page and "prompt đã gửi" in page and "khong_co_bang" in page
