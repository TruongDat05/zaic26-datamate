"""training/split_sft.py: shop-level train / dev split of SFT rows, holdout shops dropped, non-train ids refused."""
import json

import pytest

from training import split_sft


def rows_and_shops(per_shop=None):
    per_shop = per_shop or {"biz_1": 5, "biz_2": 4, "biz_3": 6, "biz_4": 5, "biz_5": 3, "biz_9": 4}
    shop_of, rows = {}, []
    for shop, n in per_shop.items():
        for k in range(n):
            qid = f"q_{shop}_{k}"
            shop_of[qid] = shop
            rows.append({"id": qid, "system": "s", "user": "u", "reply": "```sql\nSELECT 1\n```", "thinking": k == 0, "route": "view"})
    return rows, shop_of


def test_split_by_shop_without_holdout_rows():
    rows, shop_of = rows_and_shops()
    train, dev, rep = split_sft.split(rows, shop_of, holdout={"biz_9"}, dev=["biz_2"])
    assert {shop_of[r["id"]] for r in dev} == {"biz_2"}
    assert {shop_of[r["id"]] for r in train} == {"biz_1", "biz_3", "biz_4", "biz_5"}
    assert rep["holdout_rows_dropped"] == 4 and rep["dev"]["rows"] == 4 and rep["train"]["rows"] == 19
    assert not {r["id"] for r in train} & {r["id"] for r in dev}


def test_random_dev_shops_are_deterministic_and_never_holdout():
    rows, shop_of = rows_and_shops()
    a = split_sft.split(rows, shop_of, holdout={"biz_9"}, dev=2, seed=7)[2]["dev_shops"]
    b = split_sft.split(rows, shop_of, holdout={"biz_9"}, dev=2, seed=7)[2]["dev_shops"]
    assert a == b and len(a) == 2 and "biz_9" not in a


def test_no_dev_set_keeps_every_non_holdout_row():
    rows, shop_of = rows_and_shops()
    train, dev, _ = split_sft.split(rows, shop_of, holdout=set(), dev=0)
    assert len(train) == len(rows) and dev == []


def test_rows_of_non_train_questions_are_refused():
    rows, shop_of = rows_and_shops()
    rows.append({"id": "q_public_1", "system": "s", "user": "u", "reply": "r"})
    with pytest.raises(SystemExit, match="not train questions"):
        split_sft.split(rows, shop_of, holdout=set(), dev=1)


def test_dev_shop_that_is_holdout_is_refused():
    rows, shop_of = rows_and_shops()
    with pytest.raises(SystemExit):
        split_sft.split(rows, shop_of, holdout={"biz_9"}, dev=["biz_9"])


def test_cli_writes_files_and_manifest(tmp_path):
    rows, shop_of = rows_and_shops()
    src = tmp_path / "all.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text("".join(json.dumps({"id": q, "business_id": s}) + "\n" for q, s in shop_of.items()), encoding="utf-8")
    holdout = tmp_path / "holdout_shops.txt"
    holdout.write_text("biz_9\n", encoding="utf-8")
    rep = split_sft.main([str(src), "--out-dir", str(tmp_path / "out"), "--dev-shops", "biz_1,biz_3", "--holdout-shops", str(holdout),
                          "--questions", str(questions)])
    out = tmp_path / "out"
    train = [json.loads(x) for x in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    dev = [json.loads(x) for x in (out / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) + len(dev) == len(rows) - 4 and rep["dev_shops"] == ["biz_1", "biz_3"]
    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"inputs", "train.jsonl", "dev.jsonl"}
