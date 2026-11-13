"""Công cụ cho pipeline sinh dữ liệu, đọc data/synth_source (dựng bằng eval/build_synth_source.py).

    from synth_tools import Kit
    kit = Kit()
    kit.oracle("biz_101", "Doanh thu tháng 5/2026 là bao nhiêu?", "2026-08-24")   # {'type', 'answer', 'template'} hoặc None
    kit.execute("biz_101", code, "2026-08-24", question=...)                       # chạy code teacher trong sandbox
    kit.check(answer, gold)                                                         # +1 / −1 / 0 như bộ chấm
    kit.prompt(q)                                                                   # messages đúng như lúc chạy
    kit.rulebook(["cf_tuan_chu_nhat"])                                              # rulebook đã áp biến thể luật
    kit.contaminated("câu hỏi mới")                                                 # trùng câu tune/holdout?

Dòng lệnh:
    python eval/synth_tools.py oracle biz_101 2026-08-24 "Doanh thu tháng 5/2026 là bao nhiêu?"
    python eval/synth_tools.py exec biz_101 2026-08-24 code.py --question "..."
    python eval/synth_tools.py prompt q_001883280e83 [--path base1] [--cf cf_tuan_chu_nhat]
    python eval/synth_tools.py guard "câu hỏi"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _shingles(text: str, n: int = 3) -> set:
    from datamate.clean import norm_key
    w = norm_key(text).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))} if w else set()


class Kit:
    def __init__(self, source=ROOT / "data/synth_source", data=ROOT / "data/train", cache=ROOT / "outputs/cache", config=ROOT / "datamate/config.yaml"):
        import yaml
        self.source, self.data, self.cache = Path(source), Path(data), Path(cache)
        if not (self.source / "splits.json").exists():
            raise FileNotFoundError(f"Chưa có {self.source}: chạy python eval/build_synth_source.py")
        self.cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        self.base_rulebook = (self.data / "rulebook.md").read_text(encoding="utf-8")
        self.splits = json.loads((self.source / "splits.json").read_text(encoding="utf-8"))
        self.role_of = {b: r for r in ("mining", "tune", "holdout") for b in self.splits[r]}
        self.cf = {r["id"]: r for r in _jsonl(self.source / "rules/counterfactual.jsonl")}
        self._biz, self._shop, self._agents = {}, {}, {}
        self._seeds = self._profiles = self._guard = None

    # ------------------------------------------------------------ dữ liệu nguồn
    def seeds(self, roles=("mining",)) -> list[dict]:
        if self._seeds is None:
            self._seeds = _jsonl(self.source / "seeds/questions.jsonl")
        return [s for s in self._seeds if s["role"] in roles]

    def seed(self, qid: str) -> dict:
        self.seeds()
        return next(s for s in self._seeds if s["id"] == qid)

    def hard_cases(self, roles=("mining",)) -> list[dict]:
        return [h for h in _jsonl(self.source / "seeds/hard_cases.jsonl") if h["role"] in roles]

    def profile(self, business_id: str) -> dict:
        if self._profiles is None:
            self._profiles = {p["business_id"]: p for p in _jsonl(self.source / "shops/index.jsonl")}
        return self._profiles[business_id]

    def shop_block(self, business_id: str) -> str:
        return (self.source / "shops" / f"{business_id}.md").read_text(encoding="utf-8")

    def biz(self, business_id: str):
        if business_id not in self._biz:
            from datamate import loader
            self._biz[business_id] = loader.load_cached(self.data / "data" / business_id, self.cache)
        return self._biz[business_id]

    def shop(self, business_id: str):
        if business_id not in self._shop:
            from datamate import shop as shop_mod
            self._shop[business_id] = shop_mod.build(self.biz(business_id))
        return self._shop[business_id]

    # ------------------------------------------------------------ rulebook và oracle
    def rulebook(self, variants=()) -> str:
        text = self.base_rulebook
        for vid in variants:
            v = self.cf[vid]
            key = v["old"] if v["kind"] == "replace" else v["anchor"]
            if key not in text:
                raise ValueError(f"{vid}: dòng gốc không còn trong rulebook (đã áp biến thể khác chạm cùng dòng?)")
            text = text.replace(key, v["new"]) if v["kind"] == "replace" else text.replace(key, key + "\n" + v["new"])
        return text

    def affected(self, question: str, variants=()) -> list[str]:
        """Các biến thể luật mà câu hỏi chạm tới (theo trigger): oracle không dùng được cho các câu này."""
        t = question.lower()
        return [vid for vid in variants if self.cf[vid]["affected_templates"] == "*" or re.search(self.cf[vid]["trigger"], t)]

    def oracle(self, business_id: str, question: str, as_of: str, variants=()) -> dict | None:
        """Đáp án lớp tính toán (đúng 2.233/2.235 câu train ngoài holdout) hoặc None khi không nhận chắc dạng câu.

        Có biến thể luật: lớp tính toán vẫn tính theo luật mặc định, nên câu chạm biến thể (trigger hoặc affected_templates) trả None."""
        from datamate import templates, templates_ext
        s = self.shop(business_id)
        if s is None or self.affected(question, variants):
            return None
        old = templates_ext.RULEBOOK
        templates_ext.RULEBOOK = self.rulebook(variants)
        try:
            a = templates.answer(question, as_of, s)
        except Exception:  # noqa: BLE001
            a = None
        finally:
            templates_ext.RULEBOOK = old
        if a is None or any(a["template"] in self.cf[v]["affected_templates"] for v in variants):
            return None
        return a

    # ------------------------------------------------------------ chạy code, chấm
    def execute(self, business_id: str, code: str, as_of: str, question: str | None = None, qid: str = "synth", timeout: float = 120.0) -> dict:
        """Chạy code trong sandbox như agent: đáp án đã qua formatter và làm tròn tiền; kèm lỗi verify và nghi ngờ 0."""
        from datamate import formatter, loader, verify
        from datamate.sandbox import run_code
        from synth_execution import child_environment
        pkl = loader.cache_file(self.data / "data" / business_id, self.cache)
        if not pkl.exists():
            self.biz(business_id)
        res = run_code(code, pkl, self.data / "calendar.csv", as_of, timeout,
                       isolation_hook=Path(__file__).resolve().with_name("synth_execution.py"), child_env=child_environment())
        out = {"ok": bool(res.get("ok")), "error": res.get("error"), "timeout": bool(res.get("timeout")), "stdout": res.get("stdout", ""), "answer": None,
               "issues": [], "zero_suspect": False}
        if res.get("ok"):
            ans = formatter.finalize(qid, res["answer"])
            if question:
                ans = verify.round_money(question, ans)
                out["issues"] = [i.code for i in verify.check(question, ans)]
                out["zero_suspect"] = bool(verify.zero_suspect(question, ans))
            out["answer"] = {k: v for k, v in ans.items() if k != "id"}
        return out

    @staticmethod
    def check(answer: dict | None, gold: dict) -> int:
        from score import match
        g = {"alias": [], **{k: v for k, v in gold.items() if k in ("type", "answer", "alias")}}
        return match({"id": "x", **(answer or {})}, g)

    # ------------------------------------------------------------ prompt đúng như lúc chạy
    def prompt(self, q: dict, path: str = "base0", variants=()) -> list[dict]:
        """messages [system, user] mà Agent gửi cho phiếu `path` (base0, base1, p3, quote, sql…). q cần id, business_id, question, as_of_date."""
        from datamate import agent as agent_mod, templates_ext
        from datamate.llm import DummyLLM
        from datamate.run import make_fewshot
        key = tuple(variants)
        if key not in self._agents:
            old = templates_ext.RULEBOOK  # Agent.__init__ ghi đè biến toàn cục này
            self._agents[key] = agent_mod.Agent(DummyLLM(), self.cfg["agent"], self.rulebook(variants), self.data / "calendar.csv", make_fewshot(self.cfg))
            templates_ext.RULEBOOK = old
        ag = self._agents[key]
        if path.startswith("base") and path[4:].isdigit():
            vs = self.cfg["agent"].get("variants") or agent_mod.DEFAULT_VARIANTS
            variant = vs[int(path[4:]) % len(vs)]
        else:
            variant = agent_mod.ESCALATE_PATHS[path]
        return ag._messages(q, self.biz(q["business_id"]), variant)

    # ------------------------------------------------------------ lọc trùng với câu đánh giá
    def contaminated(self, question: str, threshold: float = 0.8) -> dict:
        """Jaccard 3-gram từ (bỏ dấu) với câu tune/holdout. hit = True thì không đưa câu vào train."""
        if self._guard is None:
            self._guard = [(g["id"], g["role"], _shingles(g["question"])) for g in _jsonl(self.source / "guard/eval_questions.jsonl")]
        sh = _shingles(question)
        best = (0.0, None, None)
        for gid, role, gs in self._guard:
            if sh and gs:
                j = len(sh & gs) / len(sh | gs)
                if j > best[0]:
                    best = (j, gid, role)
        return {"hit": best[0] >= threshold, "score": round(best[0], 3), "id": best[1], "role": best[2]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("biz"), o.add_argument("as_of"), o.add_argument("question"), o.add_argument("--cf", nargs="*", default=[])
    e = sub.add_parser("exec")
    e.add_argument("biz"), e.add_argument("as_of"), e.add_argument("code_file"), e.add_argument("--question")
    p = sub.add_parser("prompt")
    p.add_argument("qid"), p.add_argument("--path", default="base0"), p.add_argument("--cf", nargs="*", default=[])
    g = sub.add_parser("guard")
    g.add_argument("question")
    a = ap.parse_args(argv)
    kit = Kit()
    if a.cmd == "oracle":
        r = kit.oracle(a.biz, a.question, a.as_of, a.cf)
    elif a.cmd == "exec":
        r = kit.execute(a.biz, Path(a.code_file).read_text(encoding="utf-8"), a.as_of, a.question)
    elif a.cmd == "prompt":
        s = kit.seed(a.qid)
        r = kit.prompt({"id": s["id"], "business_id": s["business_id"], "question": s["question"], "as_of_date": s["as_of_date"]}, a.path, a.cf)
    else:
        r = kit.contaminated(a.question)
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
