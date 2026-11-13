"""Kho câu cần giải: câu dùng lại + câu mới đã qua đọc ngược; bỏ trùng trong shop, bỏ câu trùng y hệt câu BTC (so cục bộ, không in câu).

    python eval/sqlv1/pool.py
→ questions/pool_train.jsonl, pool_dev.jsonl (cần lời giải), final_set.jsonl (FINAL: chỉ câu + gold), pool_report.json
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1 import world as WB  # noqa: E402
from sqlv1.questions import WORK, fold  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NONCORE_TRAIN = 700  # đủ để chọn ≤ 10% dòng không thuộc ngành lõi


def load(name):
    p = WORK / "questions" / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def btc_texts() -> set:
    out = set()
    for p in (ROOT / "data/train/questions.jsonl", ROOT / "data/synth_source/guard/eval_questions.jsonl"):
        if p.exists():
            for x in p.read_text(encoding="utf-8").splitlines():
                if x.strip():
                    out.add(fold(json.loads(x)["question"]))
    return out


def main():
    rep = collections.Counter()
    rows = load("reuse.jsonl")
    for f in ("gen_train.jsonl", "gen_dev.jsonl", "gen_final.jsonl"):
        for x in load(f):
            if x["verify"]["ok"]:
                rows.append(x)
            else:
                rep["gen_rejected_verify"] += 1
    btc = btc_texts()
    seen, out = set(), []
    for x in rows:
        k = (x["business_id"], fold(x["question"]))
        if k in seen:
            rep["dup_in_shop"] += 1
            continue
        if fold(x["question"]) in btc:
            rep["same_as_btc_question"] += 1
            continue
        seen.add(k)
        x["qid"] = x["key"]
        out.append(x)
    rng = random.Random(7)
    train = [x for x in out if x["split"] == "train"]
    core = [x for x in train if x["industry"] in WB.CORE]
    non = [x for x in train if x["industry"] not in WB.CORE]
    rng.shuffle(non)
    by = collections.defaultdict(list)
    for x in non:
        by[(x["business_id"], x["spec"]["slot"])].append(x)
    pick, k = [], 0
    while len(pick) < NONCORE_TRAIN and any(by.values()):  # rải đều shop × slot
        for key in sorted(by):
            if by[key] and len(pick) < NONCORE_TRAIN:
                pick.append(by[key].pop())
        k += 1
    sets = {"pool_train.jsonl": core + pick, "pool_dev.jsonl": [x for x in out if x["split"] == "dev"], "final_set.jsonl": [x for x in out if x["split"] == "final"]}
    for name, xs in sets.items():
        with open(WORK / "questions" / name, "w", encoding="utf-8") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        rep[name] = len(xs)
    rep["train_core"], rep["train_noncore_pick"], rep["train_noncore_available"] = len(core), len(pick), len(non)
    (WORK / "questions" / "pool_report.json").write_text(json.dumps(dict(rep), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(rep), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
