"""Câu của một kho chưa có lời giải đạt (tính cả chấm lại tĩnh với luật ngày neo hiện hành).
    python eval/sqlv1/unsolved.py pool_train → questions/pool_train_unsolved*.jsonl"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1 import verify as V  # noqa: E402
from sqlv1.questions import WORK  # noqa: E402


def solved_by(pool: dict) -> dict:
    ok = collections.defaultdict(set)
    for f in (WORK / "solutions").glob("pool_*.jsonl"):
        for x in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            if x["qid"] not in pool:
                continue
            good = x.get("ok")
            if not good and (x.get("stage") or "").startswith("static:") and x.get("static"):
                de = lambda e: e.startswith("date_not_anchored") or (e.startswith("literal_not_visible:") and V.DATETIME_LIT.fullmatch(e.split(":", 1)[1].strip()))  # noqa: E731
                good = all(de(e) for e in x["static"]) and not V.restatic(x, pool[x["qid"]])
            if good:
                ok[x["qid"]].add(x["provider"])
    return ok


if __name__ == "__main__":
    name, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "unsolved")
    pool = {json.loads(x)["qid"]: json.loads(x) for x in (WORK / "questions" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    ok = solved_by(pool)
    uns = [q for q in pool if not ok[q]]
    with open(WORK / "questions" / f"{name}_{tag}.jsonl", "w", encoding="utf-8") as f:
        for q in uns:
            f.write(json.dumps(pool[q], ensure_ascii=False) + "\n")
    print(name, "families/question:", dict(collections.Counter(len(ok[q]) for q in pool)), "unsolved:", len(uns))
