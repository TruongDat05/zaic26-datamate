"""Exercise adaptive retries with real DuckDB execution and a deterministic model stub."""
from types import SimpleNamespace

import duckdb
import pytest

from datamate.pipeline import Pipeline, Shop
from datamate.predict import parse_args

SUM = 'TYPE: number\n```sql\nSELECT SUM(amount) FROM sales\n```'
MAX = 'TYPE: number\n```sql\nSELECT MAX(amount) FROM sales\n```'
ZERO = 'TYPE: number\n```sql\nSELECT COUNT(*) FROM sales WHERE amount < 0\n```'


class RetryEngine:
    def __init__(self, initial=(SUM, MAX), thinking=None, repaired=SUM):
        self.initial, self.thinking, self.repaired = initial, thinking, repaired
        self.thinking_calls, self.repairs = [], []

    def params(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def n_tokens(self, chat):
        return 100

    def chat(self, chats, params, **kwargs):
        if kwargs.get("thinking"):
            self.thinking_calls.extend([dict(m) for m in c] for c in chats)
            texts = [self.thinking] * len(chats)
        elif chats and chats[0][-1]["content"].startswith("ERROR:"):
            self.repairs.extend([dict(m) for m in c] for c in chats)
            texts = [self.repaired] * len(chats)
        else:
            texts = list(self.initial)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=t) for _ in range(params.n)]) for t in texts]


def sample(tmp_path, engine, question="Tổng tiền là bao nhiêu?", repairs=1):
    (tmp_path / "rulebook.md").write_text("Tiền bằng tổng các dòng.", encoding="utf-8")
    args = parse_args(["--data", str(tmp_path), "--output", str(tmp_path / "out"), "--model", "unused",
                       "--views", "none", "--k", "8", "--think-k", "4", "--repairs", str(repairs)])
    pipe = Pipeline(args, engine)
    db = duckdb.connect()
    db.execute("CREATE TABLE sales AS SELECT * FROM (VALUES (10), (20)) t(amount)")
    shop = Shop("shop", db, {}, {})
    todo = [{"question": question, "as_of_date": "2026-01-01", "_oor": False}]
    chat = [{"role": "system", "content": "Use DuckDB."}, {"role": "user", "content": question}]
    try:
        result = pipe.sample(shop, todo, [[chat, chat]], lambda: False)
        texts, preds, errs, pools, _, n_repair, _ = result
        decision, = pipe.decide(todo, preds, pools)
        return texts[0], preds[0], errs[0], n_repair, decision, pipe
    finally:
        db.close()


def test_cutoff_thoughts_get_bounded_non_thinking_retries(tmp_path):
    unfinished = "<think>\n" + "unfinished reasoning " * 10000
    engine = RetryEngine(thinking=unfinished)
    texts, preds, errs, n_repair, (final, _, conf), pipe = sample(tmp_path, engine)
    assert len(engine.thinking_calls) == 4 and len(engine.repairs) == n_repair == 4
    assert final == {"type": "number", "answer": 30} and conf == 8 / 12
    assert all(p == final for p in preds[8:]) and errs[8:] == [None] * 4
    assert all("#REPAIR" in t for t in texts[8:])
    assert all("unfinished reasoning" not in m["content"] for c in engine.repairs for m in c)
    assert all("Do not start another <think>" in c[-1]["content"] for c in engine.repairs)
    assert sum(c["kind"] == "repair" and not c["thinking_mode"] for c in pipe._calls[0]) == 4
    # The audit keeps the failed execution before the successful retry.
    assert sum(e["error"] is not None for e in pipe._runs[0]) == 4


def test_cutoff_retries_respect_the_repair_limit(tmp_path):
    engine = RetryEngine(thinking="<think>unfinished", repaired='TYPE: number\n```sql\nSELECT missing FROM sales\n```')
    _, preds, errs, n_repair, (final, _, _), _ = sample(tmp_path, engine, repairs=2)
    assert len(engine.repairs) == n_repair == 8
    assert preds[8:] == [None] * 4 and all("Binder Error" in e for e in errs[8:])
    assert final == {"type": "abstain"}


def test_consensus_on_a_guarded_zero_still_gets_thinking(tmp_path):
    engine = RetryEngine(initial=(ZERO, ZERO), thinking="<think>Check the filters.</think>\n" + SUM)
    _, _, _, _, (final, _, conf), _ = sample(tmp_path, engine)
    assert len(engine.thinking_calls) == 4
    assert final == {"type": "number", "answer": 30} and conf == 0.5


@pytest.mark.parametrize("text,question,answer", [
    ("TYPE: boolean\n```sql\nSELECT SUM(amount) < 0 FROM sales\n```", "Tổng tiền có âm không?", {"type": "boolean", "answer": False}),
    ("TYPE: percent\n```sql\nSELECT 100.0 * (SUM(amount) - SUM(amount)) / SUM(amount) FROM sales\n```",
     "Doanh thu tăng bao nhiêu %?", {"type": "percent", "answer": 0}),
])
def test_real_false_and_zero_change_do_not_trigger_a_retry(tmp_path, text, question, answer):
    engine = RetryEngine(initial=(text, text))
    _, _, _, _, (final, _, conf), _ = sample(tmp_path, engine, question)
    assert not engine.thinking_calls and final == answer and conf == 1.0
