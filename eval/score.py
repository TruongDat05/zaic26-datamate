"""Bộ chấm PA cục bộ, theo luật chấm của BTC (đúng +1, sai −1, abstain 0).

    python eval/score.py --pred outputs/submission.jsonl --gold data/train/answers.jsonl [--ids eval/dev.jsonl]
        [--questions data/train/questions.jsonl] [--show-wrong 20]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import unicodedata
from pathlib import Path


def norm(s) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return " ".join(s.split())


def _num_eq(a, b, nd=None) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    if nd is not None:
        a, b = round(a + 0.0, nd), round(b + 0.0, nd)
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-6)


def _str_eq(a, ref, aliases=()) -> bool:
    if not isinstance(a, str):
        return False
    na = norm(a)
    return any(na == norm(r) for r in [ref, *aliases] if r is not None)


def _elem_eq(a, ref, aliases=()) -> bool:
    if isinstance(ref, str):
        return _str_eq(a, ref, aliases)
    return _num_eq(a, ref)


def match(pred: dict, gold: dict) -> int:
    """+1 đúng, −1 sai, 0 abstain."""
    pt = (pred or {}).get("type")
    if pt in (None, "abstain"):
        return 0
    gt = gold["type"]
    if pt != gt:
        return -1
    if gt == "unanswerable":
        return 1
    p, g = pred.get("answer"), gold.get("answer")
    alias = gold.get("alias") or []
    ok = False
    if gt == "number":
        ok = _num_eq(p, g)
    elif gt == "percent":
        ok = _num_eq(p, g, 2)
    elif gt == "date" or gt == "boolean":
        ok = p == g and type(p) is type(g)
    elif gt == "string":
        ok = _str_eq(p, g, alias)
    elif gt == "list":
        ok = isinstance(p, list) and len(p) == len(g) and all(
            _elem_eq(x, y, alias[i] if i < len(alias) and isinstance(alias[i], list) else ())
            for i, (x, y) in enumerate(zip(p, g)))
    elif gt == "table":
        ok = _table_eq(p, g, alias)
    return 1 if ok else -1


def _table_eq(p, g, alias) -> bool:
    if not isinstance(p, dict) or not isinstance(p.get("rows"), list) or not isinstance(p.get("columns"), list):
        return False
    gc, pc = [norm(c) for c in g["columns"]], [norm(c) for c in p["columns"]]
    if len(gc) != len(pc):
        return False
    order = [pc.index(c) for c in gc] if sorted(gc) == sorted(pc) else list(range(len(gc)))
    prows = [[r[i] for i in order] for r in p["rows"] if isinstance(r, list) and len(r) == len(gc)]
    if len(prows) != len(g["rows"]) or len(prows) != len(p["rows"]):
        return False
    used = [False] * len(prows)
    for gi, gr in enumerate(g["rows"]):
        al = alias[gi] if gi < len(alias) and isinstance(alias[gi], list) else []
        hit = False
        for j, pr in enumerate(prows):
            if used[j]:
                continue
            if all(_elem_eq(x, y, al if k == 0 else ()) for k, (x, y) in enumerate(zip(pr, gr))):
                used[j] = hit = True
                break
        if not hit:
            return False
    return True


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def score(pred_rows, gold_rows, ids=None):
    pred = {r["id"]: r for r in pred_rows}
    gold = {r["id"]: r for r in gold_rows}
    ids = ids or list(gold)
    per = {i: match(pred.get(i), gold[i]) for i in ids}
    n = len(ids)
    c = collections.Counter(per.values())
    return {"PA": sum(per.values()) / n if n else 0.0, "n": n, "correct": c[1], "wrong": c[-1], "abstain": c[0]}, per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--ids", help="jsonl có trường id (vd eval/dev.jsonl) để chỉ chấm tập con")
    ap.add_argument("--questions", help="questions.jsonl để in câu hỏi khi xem câu sai")
    ap.add_argument("--show-wrong", type=int, default=0)
    a = ap.parse_args(argv)
    ids = [r["id"] for r in load_jsonl(a.ids)] if a.ids else None
    gold = load_jsonl(a.gold)
    res, per = score(load_jsonl(a.pred), gold, ids)
    print(json.dumps(res, ensure_ascii=False))
    if a.show_wrong:
        q = {r["id"]: r for r in load_jsonl(a.questions)} if a.questions else {}
        pred = {r["id"]: r for r in load_jsonl(a.pred)}
        g = {r["id"]: r for r in gold}
        for i in [i for i, s in per.items() if s < 0][:a.show_wrong]:
            print("-", q.get(i, {}).get("business_id", ""), q.get(i, {}).get("question", i))
            print("   pred:", json.dumps(pred.get(i), ensure_ascii=False)[:200])
            print("   gold:", json.dumps(g[i], ensure_ascii=False)[:200])
    return res


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
