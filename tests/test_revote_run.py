"""A replay candidate must preserve its source, question coverage and per-question voting evidence."""
import json

import pytest

from analysis.revote_run import revote


def make_run(tmp_path):
    run = tmp_path / "source"
    shard = run / "shard0"
    shard.mkdir(parents=True)
    logs = run / "logs"
    logs.mkdir()
    (shard / "run.json").write_text(json.dumps({"args": {"k": "8"}}))
    qs, rows, trace = [], [], []
    for i, ps in enumerate([
        [{"type": "number", "answer": 7}] * 5 + [{"type": "unanswerable"}] * 2 + [None] * 5,
        [{"type": "number", "answer": 42}] * 8,
        [None] * 12,
    ]):
        q = {"id": f"q_{i}", "business_id": "shop", "question": "Tổng tiền là bao nhiêu?", "as_of_date": "2026-01-01"}
        vote = ps[0] or {"type": "abstain"}
        final = vote if i == 1 else {"type": "abstain"}
        share = [5 / 12, 1.0, 0.0][i]
        qs.append(q)
        rows.append({"id": q["id"], **final})
        trace.append({**q, "final": final, "vote": vote, "conf": share, "samples": [{"pred": p} for p in ps]})
        log = {**q, "answer": final, "decision": {"inputs": {"answers": ps, "samples": list(range(len(ps))),
               "threshold": 0.5, "out_of_range": False, "answer_unanswerable": True}}}
        (logs / f"{q['id']}.json").write_text(json.dumps(log))
    for path, items in [(run / "submission.jsonl", rows), (shard / "trace.jsonl", trace), (tmp_path / "questions.jsonl", qs)]:
        path.write_text("".join(json.dumps(r) + "\n" for r in items))
    return run, tmp_path / "questions.jsonl"


def test_revote_writes_a_separate_valid_candidate_and_decision_audit(tmp_path):
    run, qs = make_run(tmp_path)
    before = (run / "submission.jsonl").read_bytes()
    output = tmp_path / "candidate"
    report = revote(run, qs, output)
    assert report["before_types"]["abstain"] == 2 and report["after_types"]["abstain"] == 1
    assert report["rescued_abstains"] == 1 and report["changed_existing_answers"] == []
    assert (run / "submission.jsonl").read_bytes() == before
    assert "format OK" in (output / "submission_check.log").read_text()
    decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    assert len(decisions) == 3 and decisions[0]["share"] == 5 / 8
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        revote(run, qs, output)


def test_revote_rejects_inconsistent_source_evidence(tmp_path):
    run, qs = make_run(tmp_path)
    p = run / "logs/q_0.json"
    log = json.loads(p.read_text())
    log["decision"]["inputs"]["answers"][0] = {"type": "number", "answer": 99}
    p.write_text(json.dumps(log))
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="voting pool disagrees"):
        revote(run, qs, output)
    assert not output.exists()
