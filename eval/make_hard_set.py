"""Bộ câu khó: shop đã đổi tên cột, câu hỏi đã làm nhiễu. Holdout thường có 143/150 câu đi lớp tính toán nên gần như không đo được model.

    python eval/make_hard_set.py --out /content/hard --cache /content/hard_cache                       # đo cuối (holdout)
    python eval/make_hard_set.py --split mining --n 100000 --out /content/hard_mining --cache /content/hard_mining_cache
    python -m datamate.run --data /content/hard_mining --out OUT/hard_mining --set cache_dir=/content/hard_mining_cache \
        --set agent.templates=false --set agent.collect_all=true
    python eval/mine_errors.py OUT/hard_mining --data /content/hard_mining --cache /content/hard_mining_cache

--split mining: các shop được dùng để sinh dữ liệu (data/synth_source/splits.json), lấy log lỗi cho eval/mine_errors.py.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from datamate import loader  # noqa: E402
from stress_noise import noise  # noqa: E402
from stress_schema import rename_all  # noqa: E402

KINDS = ["bo_dau", "viet_tat", "telex", "dem", "dao_ky"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--mode", default="syn", choices=["style", "syn", "syn_table"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="holdout", choices=["holdout", "tune", "mining"],
                    help="tune: 8 shop ngoài holdout, để chọn ngưỡng; mining: các shop còn lại, để lấy log lỗi sinh dữ liệu; holdout: chỉ kiểm lại")
    a = ap.parse_args()
    data, out = ROOT / "data/train", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("rulebook.md", "calendar.csv"):
        shutil.copy(data / f, out / f)
    (out / "data").mkdir(exist_ok=True)
    hold = (ROOT / "eval/holdout_businesses.txt").read_text().split()
    if a.split in ("tune", "mining"):
        rest = sorted({json.loads(l)["business_id"] for l in open(data / "questions.jsonl", encoding="utf-8")} - set(hold))
        tune = random.Random("hard-tune").sample(rest, 8)
        hold = tune if a.split == "tune" else sorted(set(rest) - set(tune))
    by = collections.defaultdict(list)
    for l in open(data / "questions.jsonl", encoding="utf-8"):
        q = json.loads(l)
        if q["business_id"] in hold:
            by[q["business_id"]].append(q)
    rows = []
    for b in hold:
        (out / "data" / b).mkdir(exist_ok=True)
        biz = rename_all(loader.load_cached(data / "data" / b, ROOT / "outputs/cache"), a.mode)
        path = loader.cache_file(out / "data" / b, a.cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(biz, fh)
        rnd = random.Random(f"hard-{b}")
        for q in rnd.sample(by[b], min(a.n // len(hold), len(by[b]))):
            k = rnd.choice(KINDS)
            rows.append({**q, "question": noise(q["question"], k, random.Random(q["id"] + k))})
    (out / "questions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} câu, {len(hold)} shop, đổi tên kiểu {a.mode} → {out}")


if __name__ == "__main__":
    main()
