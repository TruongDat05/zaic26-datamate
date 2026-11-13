"""The submission writer and checker."""
import json

import pytest

from datamate.submission import check, problem, write_submission


def test_written_file_passes_the_check(tmp_path):
    qs = [{"id": "q_1"}, {"id": "q_2"}]
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n", encoding="utf-8")
    write_submission(tmp_path / "submission.jsonl", qs, {"q_1": {"type": "number", "answer": 3}})
    rows = [json.loads(l) for l in (tmp_path / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": "q_1", "type": "number", "answer": 3}, {"id": "q_2", "type": "abstain"}]
    assert check(tmp_path / "submission.jsonl", tmp_path / "questions.jsonl") == []


def test_problems():
    assert problem({"type": "date", "answer": "15/08/2026"})
    assert problem({"type": "number", "answer": "5"})
    assert problem({"type": "table", "answer": {"columns": ["a"], "rows": [[1, 2]]}})
    assert problem({"type": "list", "answer": ["a", 2]}) is None


def test_missing_ids_fail_the_check(tmp_path):
    qs = [{"id": "q_1"}, {"id": "q_2"}]
    questions = tmp_path / "questions.jsonl"
    submission = tmp_path / "submission.jsonl"
    questions.write_text("".join(json.dumps(q) + "\n" for q in qs), encoding="utf-8")
    submission.write_text('{"id":"q_1","type":"number","answer":3}\n', encoding="utf-8")
    assert any("missing id q_2" in e for e in check(submission, questions))


@pytest.mark.parametrize("row", [None, [], {"id": []}, {"id": "q_1", "type": []}])
def test_malformed_rows_are_reported_as_errors(tmp_path, row):
    questions, submission = tmp_path / "questions.jsonl", tmp_path / "submission.jsonl"
    questions.write_text('{"id":"q_1"}\n', encoding="utf-8")
    submission.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert check(submission, questions)


@pytest.mark.parametrize("row", [
    {"type": "number", "answer": float("nan")},
    {"type": "percent", "answer": float("inf")},
    {"type": "list", "answer": [1, float("-inf")]},
    {"type": "table", "answer": {"columns": ["amount"], "rows": [[float("nan")]]}},
])
def test_nonfinite_numbers_are_invalid_json(row):
    assert problem(row)
