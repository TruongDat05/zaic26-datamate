"""Re-vote a completed run's saved SQL outputs without generating or executing anything again.

    python analysis/revote_run.py RUN --questions DATA/questions.jsonl --output NEW_DIRECTORY

The source submission, traces and per-question logs must agree and cover every question. The output directory must
be new. This measures selection/coverage changes only; sampling/repair fixes require a fresh inference run.
"""
import argparse
import contextlib
import hashlib
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datamate.ingest import norm  # noqa: E402
from datamate.sqlexec import CHANGE  # noqa: E402
from datamate.submission import check, write_submission  # noqa: E402
from datamate.voting import VOTE_POLICY, empty_answer, final_answer, vote  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def revote(run, questions, output):
    run, questions, output = Path(run).resolve(), Path(questions).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source = run / "submission.jsonl"
    source_sha = sha(source)
    qs, old_rows = jsonl(questions), jsonl(source)
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids) or ids != [row["id"] for row in old_rows]:
        raise ValueError("Source submission must match the exact question IDs and order")
    records = {}
    for trace in sorted(run.glob("shard*/trace.jsonl")) or [run / "trace.jsonl"]:
        settings = json.loads((trace.parent / "run.json").read_text())["args"]
        base_k = 2 * (int(settings["k"]) // 2)
        if base_k <= 0:
            raise ValueError("The source run has no initial sampling budget")
        for rec in jsonl(trace):
            qid = rec["id"]
            if qid in records:
                raise ValueError(f"Duplicate trace ID {qid}")
            records[qid] = (rec, base_k, trace.parent)
    if set(records) != set(ids):
        raise ValueError("Trace coverage must match every question; partial runs cannot be re-voted")

    results, decisions, changed = {}, [], []
    for q, old in zip(qs, old_rows):
        qid = q["id"]
        rec, base_k, shard = records[qid]
        log_path = run / "logs" / f"{qid}.json"
        if not log_path.exists():
            log_path = shard / "logs" / f"{qid}.json"
        log = json.loads(log_path.read_text(encoding="utf-8"))
        if old != {"id": qid, **rec["final"]} or log["answer"] != rec["final"]:
            raise ValueError(f"Submission, trace and question log disagree: {qid}")
        if any(log[k] != q[k] for k in ("question", "business_id", "as_of_date")):
            raise ValueError(f"Question contents differ from the source run: {qid}")
        inp = log["decision"]["inputs"]
        voters = [rec["samples"][j]["pred"] for j in inp["samples"]]
        if voters != inp["answers"]:
            raise ValueError(f"Logged voting pool disagrees with trace samples: {qid}")
        change_q = bool(CHANGE.search(norm(q["question"])))
        ans, conf = vote(voters, len(voters), base_k=base_k, change_q=change_q)
        final = final_answer(ans, conf, inp["threshold"], inp["out_of_range"], inp["answer_unanswerable"], change_q)
        results[qid] = final
        decision = {"id": qid, "business_id": q["business_id"], "question": q["question"], "source_log": str(log_path),
                    "old_final": rec["final"], "final": final, "old_vote": rec["vote"], "vote": ans,
                    "old_share": rec["conf"], "share": conf, "threshold": inp["threshold"], "base_k": base_k,
                    "samples": inp["samples"], "attempted": len(voters),
                    "usable": sum(bool(p) and p["type"] != "abstain" and (p["type"] == "unanswerable" or not empty_answer(p, change_q)) for p in voters),
                    "vote_policy": VOTE_POLICY}
        decisions.append(decision)
        if final != rec["final"]:
            changed.append(decision)

    if sha(source) != source_sha:
        raise RuntimeError("Source submission changed during replay")
    output.mkdir(parents=True)
    write_submission(output / "submission.jsonl", qs, results)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        errors = check(output / "submission.jsonl", questions)
    (output / "submission_check.log").write_text(captured.getvalue() + ("format OK\n" if not errors else str(errors)), encoding="utf-8")
    if errors:
        raise ValueError(f"Re-voted submission failed schema checks: {errors}")
    (output / "decisions.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decisions), encoding="utf-8")
    changed_answers = [d["id"] for d in changed if d["old_final"]["type"] != "abstain"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": "offline_revote_no_new_generation_or_sql",
              "source_run": str(run), "source_submission_sha256": source_sha, "questions_sha256": sha(questions),
              "submission": str(output / "submission.jsonl"), "submission_sha256": sha(output / "submission.jsonl"),
              "questions": len(qs), "vote_policy": VOTE_POLICY,
              "before_types": dict(Counter(row["type"] for row in old_rows)), "after_types": dict(Counter(r["type"] for r in results.values())),
              "rescued_abstains": sum(d["old_final"]["type"] == "abstain" and d["final"]["type"] != "abstain" for d in changed),
              "changed_existing_answers": changed_answers, "changed": changed,
              "not_replayed": ["retry of truncated thinking", "new thinking samples for guarded zero/empty results"],
              "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "datamate/voting.py", ROOT / "datamate/pipeline.py", ROOT / "datamate/prompts.py", Path(__file__).resolve()]}}
    (output / "revote_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Offline re-vote\n\nThis candidate reuses the source run's saved executed outputs. No model generation or SQL execution was repeated. "
        "The original submission is preserved. `decisions.jsonl` links every decision to its original model/SQL log.\n\n"
        "Coverage changes do not establish answer correctness or competition score. The truncated-thinking and guarded-result retry fixes "
        "take effect on fresh inference; they cannot be measured by this replay.\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = revote(args.run, args.questions, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in ("changed", "source_hashes")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
