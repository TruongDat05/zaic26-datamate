"""Direction 4 of the baseline slides (a verifier or answer selector): answer-or-abstain rules replayed offline on saved traces,
each scored leave-one-shop-out (fit any free parameter on the other shops, score the held-out one). Train gold only, no LLM.

    python analysis/select_cv.py RUN [RUN ...] --gold data/train/answers.jsonl [--out report.json]

RUN = folder with shard*/trace.jsonl (or trace.jsonl), from a labelled run of datamate.predict (holdout or train pool).
Every rule keeps the voted answer itself (voting.vote over the same pool as inference) and only changes when to answer.
Rules read only the samples of the question: no question id, no business id, no gold (rule 4 of the track)."""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate import hints, ingest, voting  # noqa: E402
from datamate.scoring import score  # noqa: E402

TAUS = [0.25, 0.3, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6, 0.625, 0.65, 0.7, 0.75, 0.8]
MARGINS = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5]
PLAIN = 8  # samples 0..7 are plain, 8.. thinking (pipeline.sample)
OOR = "period outside the shop's data dates"


def load(runs):
    out = {}
    for run in map(Path, runs):
        for f in sorted(run.glob("shard*/trace.jsonl")) or [run / "trace.jsonl"]:
            for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
                out[r["id"]] = r
    return out


def debt_shops(runs):
    return {f.stem.removeprefix("_debt_") for run in runs for f in Path(run).glob("**/logs/_debt_*.json")
            if json.loads(f.read_text(encoding="utf-8")).get("used")}


def top_key(preds):
    c = Counter(voting.answer_key(p) for p in preds if p)
    return c.most_common(1)[0][0] if c else None


def features(r, debt):
    """Everything a rule may look at, computed as inference would (same voting pool, same vote)."""
    texts = [s["text"] for s in r["samples"]]
    preds = [s["pred"] for s in r["samples"]]
    dq = r["business_id"] in debt and bool(hints.DEBT_WORDS.search(ingest.norm(r["question"])))
    pool = voting.debt_pool(texts, PLAIN, dq) if texts else []
    voters = [preds[j] for j in pool]
    ans, share = voting.vote(voters, len(voters)) if voters else ({"type": "abstain"}, 0.0)
    sizes = sorted(Counter(voting.answer_key(p) for p in voters if p).values(), reverse=True) + [0, 0]
    k = max(1, len(voters))
    plain, think = preds[:PLAIN], preds[PLAIN:]
    pk, tk = top_key(plain), top_key(think)
    v = ans.get("answer")
    nil = v in (0, [], None) or (isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == 0 for x in v)) \
        or (ans.get("type") == "table" and not (v or {}).get("rows"))
    return {"ans": ans, "share": share, "margin": (sizes[0] - sizes[1]) / k, "groups": sum(1 for s in sizes if s), "k": len(voters),
            "has_think": bool(think), "agree": bool(think) and pk is not None and pk == tk, "fail": sum(p is None for p in preds) / max(1, len(preds)),
            "debt_q": dq, "nil": nil and ans.get("type") != "boolean", "oor": r["final"].get("reason") == OOR, "type": ans.get("type")}


def final(f, ok):
    """The submission line: out-of-range and unanswerable handling exactly as voting.final_answer (--answer-unanswerable)."""
    if f["oor"]:
        return {"type": "unanswerable", "reason": OOR}
    if f["type"] == "unanswerable":
        return f["ans"] if f["share"] >= 0.5 else {"type": "abstain"}
    if f["type"] == "abstain" or f["nil"]:
        return {"type": "abstain"}
    return f["ans"] if ok else {"type": "abstain"}


def tgroup(t):
    return "num" if t in ("number", "percent") else "list" if t == "list" else "other"


# ---- rules: fit(rows) -> params, gate(f, params) -> answer? -----------------------------------------------------------------------------
def grid(rows, cands, gate):
    return max(cands, key=lambda p: (pa(rows, gate, p), -dist(p)))


def dist(p):
    return sum(abs(x - .5) for x in (p if isinstance(p, tuple) else (p,)))


def pa(rows, gate, p):
    return sum(score(final(f, gate(f, p)), g) for f, g, _ in rows)


def rule_current():
    return lambda rows: 0.5, lambda f, t: f["share"] >= t


def rule_tau():
    gate = lambda f, t: f["share"] >= t
    return lambda rows: grid(rows, TAUS, gate), gate


def rule_tau_by_type():
    def fit(rows):
        return {g: grid([x for x in rows if tgroup(x[0]["type"]) == g], TAUS, lambda f, t: f["share"] >= t) for g in ("num", "list", "other")}
    return fit, lambda f, p: f["share"] >= p[tgroup(f["type"])]


def rule_margin():
    gate = lambda f, m: f["margin"] >= m
    return lambda rows: grid(rows, MARGINS, gate), gate


def rule_tau_margin():
    gate = lambda f, p: f["share"] >= p[0] and f["margin"] >= p[1]
    return lambda rows: grid(rows, [(t, m) for t in TAUS for m in MARGINS], gate), gate


def rule_think_agree():
    """Answer at share >= .5 as now, and below it down to tau_low when the plain and the thinking samples lead with the same answer."""
    gate = lambda f, lo: f["share"] >= .5 or (f["agree"] and f["share"] >= lo)
    return lambda rows: grid(rows, [t for t in TAUS if t <= .5], gate), gate


def vec(f):
    t = tgroup(f["type"])
    return [1.0, f["share"], f["margin"], min(f["groups"], 8) / 8, f["k"] / 12, float(f["has_think"]), float(f["agree"]), f["fail"], float(f["debt_q"]),
            float(t == "num"), float(t == "list"), f["share"] * float(f["agree"])]


def rule_logistic(l2=1.0, steps=3000, lr=0.5):
    """A 12-weight logistic selector P(vote correct | features). Answers when P > 0.5, the break-even point of +1 / -1 scoring.
    Fit only on questions where the gate matters (an answer that is neither out of range, unanswerable, nil nor abstain)."""
    def fit(rows):
        data = [(vec(f), 1.0 if score(f["ans"], g) == 1 else 0.0) for f, g, _ in rows if not (f["oor"] or f["nil"] or f["type"] in ("unanswerable", "abstain"))]
        w = [0.0] * len(vec(rows[0][0]))
        n = max(1, len(data))
        for _ in range(steps):
            grad = [l2 * wi / n if i else 0.0 for i, wi in enumerate(w)]
            for x, y in data:
                p = 1 / (1 + math.exp(-max(-30, min(30, sum(a * b for a, b in zip(w, x))))))
                for i, xi in enumerate(x):
                    grad[i] += (p - y) * xi / n
            w = [wi - lr * gi for wi, gi in zip(w, grad)]
        return w
    return fit, lambda f, w: sum(a * b for a, b in zip(w, vec(f))) > 0


RULES = {"current (tau .5)": rule_current, "tau (LOSO)": rule_tau, "tau by type": rule_tau_by_type, "margin": rule_margin,
         "tau + margin": rule_tau_margin, "thinking agrees": rule_think_agree, "logistic selector": rule_logistic}


def evaluate(rows):
    shops = sorted({b for _, _, b in rows})
    out = {}
    for name, make in RULES.items():
        fit, gate = make()
        c, per_shop, params = Counter(), {}, {}
        for b in shops:
            p = fit([x for x in rows if x[2] != b])
            params[b] = p
            s = Counter(score(final(f, gate(f, p)), g) for f, g, bb in rows if bb == b)
            per_shop[b] = s[1] - s[-1]
            c += s
        n = len(rows)
        out[name] = {"pa": (c[1] - c[-1]) / n, "correct": c[1], "wrong": c[-1], "abstain": c[0], "per_shop": per_shop,
                     "params": {b: (p if not isinstance(p, list) else [round(x, 3) for x in p]) for b, p in params.items()}}
    base = out["current (tau .5)"]["per_shop"]
    for v in out.values():
        v["shops_better"] = sum(v["per_shop"][b] > base[b] for b in shops)
        v["shops_worse"] = sum(v["per_shop"][b] < base[b] for b in shops)
    return out, shops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    gold = {r["id"]: r for r in map(json.loads, a.gold.read_text(encoding="utf-8").splitlines())}
    rs, debt = load(a.runs), debt_shops(a.runs)
    rows = [(features(r, debt), gold[i], r["business_id"]) for i, r in rs.items() if i in gold]
    same = sum(final(f, f["share"] >= .5) == rs[i]["final"] for (f, _, _), i in zip(rows, [i for i in rs if i in gold]))
    print(f"{len(rows)} questions, {len({b for *_, b in rows})} shops; replay reproduces {same}/{len(rows)} finals of the trace")
    res, shops = evaluate(rows)
    base = res["current (tau .5)"]["pa"]
    print(f"{'rule':<20} {'LOSO PA':>8} {'delta':>7} {'right':>6} {'wrong':>6} {'abst':>6} {'shops +/-':>10}")
    for name, v in res.items():
        print(f"{name:<20} {v['pa']:>8.4f} {v['pa'] - base:>+7.4f} {v['correct']:>6} {v['wrong']:>6} {v['abstain']:>6} {v['shops_better']:>4}/{v['shops_worse']:<4}")
    if a.out:
        a.out.write_text(json.dumps({"n": len(rows), "shops": shops, "replay_same": same, "rules": res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
