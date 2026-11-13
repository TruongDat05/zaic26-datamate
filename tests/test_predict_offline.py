"""CSV -> DuckDB -> SQL/repair/vote -> shard checkpoints, logs and final submission, without vLLM."""
import json
from types import SimpleNamespace

from datamate import predict
from datamate.submission import check


class FakeEngine:
    def __init__(self, *args, **kwargs):
        pass

    def params(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def n_tokens(self, chat):
        return sum(len(m["content"].split()) for m in chat)

    def chat(self, chats, params, **kwargs):
        out = []
        for i, chat in enumerate(chats):
            # One initial sample has a binder error; the other and the repair read the loaded amount column.
            col = '"Thành tiền"' if i or chat[-1]["content"].startswith("ERROR:") else "missing_amount"
            text = f"TYPE: number\n```sql\nSELECT SUM({col}) FROM sales\n```"
            out.append(SimpleNamespace(outputs=[SimpleNamespace(text=text) for _ in range(params.n)]))
        return out


def test_two_shard_predictions_keep_repairs_logs_and_all_ids(tmp_path, monkeypatch):
    data, out = tmp_path / "input", tmp_path / "result"
    qs = []
    expected = {"q_a1": -8_575_000, "q_b1": 8_625_500}
    for b, amounts in [("shop_a", ["-8tr600", "25k"]), ("shop_b", ["8tr600", "25k500"])]:
        shop = data / "data" / b
        shop.mkdir(parents=True)
        (shop / "README.md").write_text("Thành tiền là số tiền của mỗi dòng.\n", encoding="utf-8")
        (shop / "sales.csv").write_text('Mã đơn,Ngày,Thành tiền\n' + "".join(f"HD{i},01/03/2025,{a}\n" for i, a in enumerate(amounts)), encoding="utf-8")
        qs.append({"id": f"q_{b[-1]}1", "business_id": b, "as_of_date": "2025-03-01", "question": "Tổng thành tiền ngày 01/03/2025 là bao nhiêu?"})
    questions = data / "questions.jsonl"
    questions.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    (data / "rulebook.md").write_text("Tiền tính bằng đồng; cộng các dòng để tính tổng.\n", encoding="utf-8")
    (out / "work").mkdir(parents=True)
    monkeypatch.setattr(predict, "Engine", FakeEngine)
    for i in range(2):
        predict.main(["--data", str(data), "--output", str(out / "work" / f"shard{i}"), "--model", "unused",
                      "--shard", f"{i}/2", "--log-dir", str(out / "logs"), "--merge-into", str(out), "--views", "none", "--k", "2", "--repairs", "1"])
        rows = [json.loads(line) for line in (out / "submission.jsonl").read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0] == {"id": "q_a1", "type": "number", "answer": expected["q_a1"]}
        if i == 0:
            assert rows[1] == {"id": "q_b1", "type": "abstain"}
        run = json.loads((out / "work" / f"shard{i}" / "run.json").read_text())
        assert run["answered"] == 1 and run["repairs"] == 1
    assert {r["id"]: r["answer"] for r in rows} == expected
    assert check(out / "submission.jsonl", questions) == []
    for qid, answer in expected.items():
        log = json.loads((out / "logs" / f"{qid}.json").read_text())
        assert log["answer"] == {"type": "number", "answer": answer}
        assert [call["kind"] for call in log["calls"]] == ["sample", "sample", "repair"]
        assert log["executions"][0]["error"] and log["executions"][-1]["output"] == log["answer"]
        assert log["decision"]["share"] == 1.0
