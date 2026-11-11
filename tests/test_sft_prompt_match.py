"""The SFT rows in data/data/ were rendered with the inference prompt: an adapter trained on them must be served the same system prompt.
An edit of datamate/prompts.py SYSTEM (or of the bundle) that breaks this is a train / serve skew, caught here."""
import json
from pathlib import Path

import pytest

from datamate.prompts import SYSTEM

TRAIN = Path(__file__).resolve().parents[1] / "data/data/train.jsonl"


def test_system_prompt_is_the_one_the_sft_rows_were_trained_on():
    if not TRAIN.exists() or TRAIN.stat().st_size < 1_000_000:
        pytest.skip("data/data/train.jsonl missing or a Git LFS pointer")
    with TRAIN.open(encoding="utf-8") as f:
        row = json.loads(f.readline())
    system = row["prompt_messages"][0]["content"]
    assert "\n\nRULEBOOK:\n" in system   # the pipeline sends SYSTEM + "\n\nRULEBOOK:\n" + rulebook (datamate/pipeline.py)
    assert system.split("\n\nRULEBOOK:\n")[0] == SYSTEM
