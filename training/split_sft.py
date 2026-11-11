"""Split SFT rows by shop into train / dev, with the holdout shops kept out of both.

    python training/split_sft.py runs/sft/x/all_notes.jsonl --out-dir runs/sft/x --dev-shops 2
    python training/split_sft.py IN.jsonl --out-dir DIR --dev-shops biz_104,biz_119 --holdout-shops splits/holdout_shops.txt

Why by shop: rows of one shop share the README, the schema part of the prompt and often the SQL pattern. A random row split puts
near-copies on both sides, and the dev loss then stops measuring what public / private ask for (shops the adapter never saw).
Why drop the holdout shops: a pipeline run on splits/holdout600.txt (02_inference, SPLIT = "train") then measures the new adapter
on 6 shops it never trained on. LoRA v3 learned 390 holdout questions, so its holdout score is only a smoke test (docs/leaderboard.md).
Every row id must be a train question: rows of public / private questions are refused, never written.

Writes OUT_DIR/train.jsonl, OUT_DIR/dev.jsonl (empty with --dev-shops 0, the final fit on all shops) and OUT_DIR/split_manifest.json
(shops of each side, row counts, SHA-256 of the input and output files). notebooks/04_training_data.ipynb drives this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATAMATE_DATA") or R / "data")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_list(path: Path | None) -> set[str]:
    return set(Path(path).read_text(encoding="utf-8").split()) if path else set()


def pick_dev_shops(shops: Counter, n: int, seed: int) -> list[str]:
    """n shops drawn at random (fixed seed) among those whose row count lies between the 20th and 80th percentile, so the dev set
    is neither a single huge shop nor a handful of rows. Falls back to all shops when that band is too small."""
    names = sorted(shops)
    if n <= 0:
        return []
    if n >= len(names):
        raise SystemExit(f"--dev-shops {n}: only {len(names)} shops have rows, keep most of them for training")
    counts = sorted(shops.values())
    lo, hi = counts[int(0.2 * (len(counts) - 1))], counts[int(0.8 * (len(counts) - 1))]
    band = [s for s in names if lo <= shops[s] <= hi]
    pool = band if len(band) >= n else names
    return sorted(random.Random(seed).sample(pool, n))


def split(rows: list[dict], shop_of: dict[str, str], holdout: set[str], dev: int | list[str], seed: int = 0):
    """(train rows, dev rows, report). Rows keep their order. Raises on a row whose id is not a train question."""
    unknown = sorted({str(r.get("id")) for r in rows if str(r.get("id")) not in shop_of})
    if unknown:
        raise SystemExit(f"{len(unknown)} row ids are not train questions (e.g. {unknown[:3]}): SFT rows come from labelled train runs only")
    kept = [r for r in rows if shop_of[str(r["id"])] not in holdout]
    shops = Counter(shop_of[str(r["id"])] for r in kept)
    dev_shops = sorted(dev) if isinstance(dev, list) else pick_dev_shops(shops, dev, seed)
    missing = [s for s in dev_shops if s not in shops]
    if missing:
        raise SystemExit(f"dev shops {missing} have no row (holdout shops or not in the runs); shops with rows: {sorted(shops)}")
    if set(dev_shops) & holdout:
        raise SystemExit(f"dev shops {sorted(set(dev_shops) & holdout)} are holdout shops")
    train = [r for r in kept if shop_of[str(r["id"])] not in dev_shops]
    dev_rows = [r for r in kept if shop_of[str(r["id"])] in dev_shops]
    if not train:
        raise SystemExit("no train row left after the split")

    def side(rs):
        return {"rows": len(rs), "questions": len({str(r["id"]) for r in rs}), "thinking_rows": sum(bool(r.get("thinking")) for r in rs),
                "shops": dict(sorted(Counter(shop_of[str(r["id"])] for r in rs).items())),
                "routes": dict(Counter(r.get("route") for r in rs).most_common())}

    report = {"rows_in": len(rows), "holdout_shops": sorted(holdout), "holdout_rows_dropped": len(rows) - len(kept),
              "dev_shops": dev_shops, "seed": seed, "train": side(train), "dev": side(dev_rows)}
    return train, dev_rows, report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path, nargs="+", help="SFT JSONL file(s) from training/build_sft.py / training/add_notes.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dev-shops", default="2", help="a number of shops drawn at random, or a comma-separated list; 0 = no dev set")
    ap.add_argument("--holdout-shops", default=str(R / "splits/holdout_shops.txt"), help="file of shops kept out of train and dev ('' = none)")
    ap.add_argument("--questions", type=Path, default=DATA / "train/questions.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    shop_of = {q["id"]: q["business_id"] for q in read_jsonl(a.questions)}
    rows = [r for p in a.src for r in read_jsonl(p)]
    holdout = read_list(Path(a.holdout_shops)) if a.holdout_shops else set()
    dev = int(a.dev_shops) if a.dev_shops.strip().isdigit() else [s.strip() for s in a.dev_shops.split(",") if s.strip()]
    train, dev_rows, report = split(rows, shop_of, holdout, dev, a.seed)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out_dir / "train.jsonl", train)
    write_jsonl(a.out_dir / "dev.jsonl", dev_rows)
    report["files"] = {"inputs": {str(p): sha256(p) for p in a.src},
                       "train.jsonl": sha256(a.out_dir / "train.jsonl"), "dev.jsonl": sha256(a.out_dir / "dev.jsonl")}
    (a.out_dir / "split_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rows in {report['rows_in']}, holdout rows dropped {report['holdout_rows_dropped']} ({len(holdout)} shops); "
          f"train {report['train']['rows']} rows / {len(report['train']['shops'])} shops, dev {report['dev']['rows']} rows / {report['dev_shops']}")
    print(f"split -> {a.out_dir / 'train.jsonl'}, {a.out_dir / 'dev.jsonl'}, {a.out_dir / 'split_manifest.json'}")
    return report


if __name__ == "__main__":
    main()
