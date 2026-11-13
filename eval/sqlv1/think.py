"""Vết suy nghĩ (thinking) qua API cho dòng SFT thinking của adapter Qwen3.5-9B (production chạy --lora-thinking, --think-k 4).

    python eval/sqlv1/think.py plan [--n 1200 --dev-frac 0.12]          → outputs/synth_sql_v1/think/plan.json (chọn câu + dự toán; KHÔNG gọi API)
    python eval/sqlv1/think.py run --provider qwen [--limit 50 --tag pilot]  → outputs/synth_sql_v1/think/results.<provider>.jsonl
    python eval/sqlv1/think.py summary                                    → tỷ lệ nhận, độ dài vết, chi phí theo provider (từ ledger + kết quả)
    python eval/sqlv1/think.py candidates --provider qwen                 → <pkg>/candidates/thinking.jsonl (cho mutation_replay/probe/bundle)

Request (MỘT câu/request): system = prompts.SYSTEM của runtime + rulebook_api.md (viết lại, không chép rulebook BTC) bỏ nhãn mục A–J
(RULEBOOK_THINK); user = ctx shop tổng hợp (README + SCHEMA + CANONICAL VIEWS + DEBTS như runtime) + khối câu của solve.py (### id, DATE
ANCHORS, gợi ý kỳ, VALUES MATCHING, QUESTION) + THINK_FMT. Không có ví dụ few-shot BTC, không có gold. Chế độ thinking của provider.
Nhận một vết khi: finish = stop; reasoning qua OC.think_problem (không trích mục rulebook_api, không nhắc định dạng request); content đúng định
dạng runtime (OC.runtime_answer); judge() cục bộ đạt (chạy 2 lần, khớp gold oracle, ≥ 2 bản đột biến, kiểm tĩnh). Completion =
F.canonical_thinking(reasoning.strip(), OC.layout(type, sql, notes)) — đúng phép biến đổi exporter tự tính lại từ bản ghi cache.
Exporter (build_sft --offline-candidates) vẫn kiểm lại độc lập; dòng thinking có prompt runtime biến thể p1 (SYSTEM + ví dụ BTC đã lọc).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

from sqlv1 import shops as SH, solve as SV, verify as V  # noqa: E402

FOXEL = controller_root()
sys.path.append(str(FOXEL / "training"))
import offline_candidates as OC  # noqa: E402
import sft_format as F  # noqa: E402

WORK = SH.SYNTH_ROOT / "outputs" / "synth_sql_v1"
TDIR = WORK / "think"
PKG = FOXEL / "build" / "synthetic_sql_v1"
THINK_FMT = ("Answer this one question in the output format of the system prompt: a line 'Notes:', at most 6 short '- ' lines, the TYPE line, "
             "then one ```sql block. Name a rule by its content (e.g. 'valid orders only'), never by a letter or section number.")
MAX_TOKENS = 16000          # production: --think-budget 16384 + --max-tokens 2048; a cut-off thought (finish != stop) is never used
TEMPERATURE = 0.6           # Qwen's thinking-mode sampling, the runtime's thinking samples use the same
PRICE = {"qwen": (0.45, 3.20), "deepseek": (0.14, 0.28)}   # USD / 1M tokens in, out (eval/llm.py)


def sha8(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 16 ** 8


# rulebook_api.md without its section letters ('## A. ...', 'như mục B'): in the pilot 13/25 Qwen traces cited them ('Rule J', 'rulebook H')
# although asked not to, and the runtime rulebook the SFT row shows is another text. Without labels there is nothing to cite.
RULEBOOK_THINK = re.sub(r"(?m)^## [A-J]\. ", "## ", SV.RULEBOOK_API).replace("như mục B", "như phần Khách")
SYSTEM_THINK = SV.prompts.SYSTEM + "\n\nRULEBOOK:\n" + RULEBOOK_THINK
assert not re.search(r"(?m)^## [A-J]\.|mục [A-J]\b", RULEBOOK_THINK)


def think_messages(env: V.ShopEnv, q: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_THINK},
            {"role": "user", "content": env.layers["ctx"] + "\n\n" + SV.question_block(env, q) + "\n\n" + THINK_FMT}]


# ------------------------------------------------------------------ plan: which questions get a thinking trace
def load_rows():
    rows = []
    for s in ("train", "dev"):
        rows += F.read_rows(PKG / "sft" / f"{s}.jsonl")
    return rows


def hardness():
    """{qid (key): set of reasons} from the solver pools: a first attempt that failed, a repair, disagreement."""
    out = collections.defaultdict(set)
    for f in sorted((WORK / "solutions").glob("pool_*.jsonl")):
        if ".template." in f.name:
            continue
        for x in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            if x.get("attempt") == "first" and not x.get("ok"):
                out[x["qid"]].add("api_first_attempt_failed")
            if x.get("attempt") == "repair":
                out[x["qid"]].add("repaired")
    return out


def plan(a):
    sel = {x["id"]: x for x in map(json.loads, (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines()) if x.get("id")}
    rows = load_rows()
    by_q = collections.defaultdict(list)
    for r in rows:
        by_q[r["metadata"]["question_id"]].append(r)
    hard = hardness()
    cands = []
    for qid, rs in by_q.items():
        s = sel[qid]
        if s["split"] not in ("train", "dev"):
            raise SystemExit(f"{qid}: split {s['split']} in the SFT build")  # FINAL never gets here
        m = rs[0]["metadata"]
        why = set(hard.get(s["qid"], ()))
        if any(r["metadata"]["route"] == "macro" for r in rs):
            why.add("debt_macro")  # production sends every debt question on a debt shop to thinking samples
        if m["answer_type"] in ("unanswerable", "list", "date", "string"):
            why.add("type_" + m["answer_type"])
        if len(s["question"].split()) >= 20:
            why.add("long_question")
        cands.append({"id": qid, "qid": s["qid"], "business_id": s["business_id"], "split": s["split"], "slot": s["spec"].get("slot"),
                      "type": m["answer_type"], "why": sorted(why), "u": sha8(f"think/1|{qid}")})
    want = {"dev": round(a.n * a.dev_frac), "train": a.n - round(a.n * a.dev_frac)}
    chosen = []
    for split, n in want.items():
        pool = [c for c in cands if c["split"] == split]
        # every question with a hard signal first (in hash order), then the rest stratified by slot (round robin over slots, hash order)
        hard_q = sorted([c for c in pool if c["why"]], key=lambda c: (-len(c["why"]), c["u"]))
        pick = hard_q[:n]
        rest = collections.defaultdict(list)
        for c in sorted((c for c in pool if not c["why"]), key=lambda c: c["u"]):
            rest[c["slot"]].append(c)
        slots = sorted(rest)
        i = 0
        while len(pick) < n and any(rest.values()):
            s = slots[i % len(slots)]
            if rest[s]:
                pick.append(rest[s].pop(0))
            i += 1
        chosen += pick
    TDIR.mkdir(parents=True, exist_ok=True)
    est = estimate(chosen)
    out = {"n": len(chosen), "by_split": dict(collections.Counter(c["split"] for c in chosen)),
           "by_reason": dict(collections.Counter(w for c in chosen for w in (c["why"] or ["stratified_fill"]))),
           "by_type": dict(collections.Counter(c["type"] for c in chosen)), "slots": len({c["slot"] for c in chosen}),
           "estimate": est, "questions": chosen}
    (TDIR / "plan.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "questions"}, ensure_ascii=False, indent=1))


def estimate(chosen):
    """Input tokens per request measured on 40 requests (chars/token of past solve calls), output assumed 2k-8k thinking tokens."""
    db = sqlite3.connect(f"file:{WORK / 'requests.sqlite'}?mode=ro", uri=True)
    ratio = []
    for f in sorted((WORK / "api" / "solve").glob("*/*.json"))[:200]:
        rec = json.loads(f.read_text(encoding="utf-8"))
        u = (rec["response"].get("usage") or {}).get("prompt_tokens")
        if u and rec["settings"]["provider"] in PRICE:
            ratio.append(sum(len(m["content"]) for m in rec["messages"]) / u)
    db.close()
    cpt = statistics.median(ratio) if ratio else 3.0
    sample = sorted(chosen, key=lambda c: c["u"])[:40]
    sizes = []
    by = collections.defaultdict(list)
    for c in sample:
        by[c["business_id"]].append(c)
    for b, qs in by.items():
        env = V.ShopEnv(WORK, b, [])
        try:
            for c in qs:
                q = qrec(c)
                sizes.append(sum(len(m["content"]) for m in think_messages(env, q)) / cpt)
        finally:
            env.close()
    tin = statistics.mean(sizes)
    n = len(chosen)
    est = {"chars_per_token": round(cpt, 2), "input_tokens_per_request": round(tin), "requests": n}
    for p, (pi, po) in PRICE.items():
        est[p] = {f"out_{k}k": round(n * (tin * pi + k * 1000 * po) / 1e6, 2) for k in (2, 4, 8)}
    return est


_SEL = {}


def qrec(c):
    if not _SEL:
        _SEL.update({x["id"]: x for x in map(json.loads, (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines()) if x.get("id")})
    return _SEL[c["id"]]


# ------------------------------------------------------------------ run
def think_shop(args):
    from sqlv1 import api
    bid, qs, provider, all_specs = args
    try:
        env = V.ShopEnv(WORK, bid, all_specs)
    except Exception as e:  # noqa: BLE001  (no API call was made for this shop: it is retried by the next run)
        return {"bid": bid, "results": [], "error": f"{type(e).__name__}: {e}"[:300]}
    out = []
    try:
        for c in qs:
            q = qrec(c)
            msgs = think_messages(env, q)
            r, cached = api.call(WORK, "think", bid, provider, msgs, think="high", max_tokens=MAX_TOKENS, temperature=TEMPERATURE, timeout=900)
            rec = {"id": c["id"], "qid": q["qid"], "business_id": bid, "split": c["split"], "provider": provider, "cached": cached,
                   "finish": r.get("finish"), "usage": r.get("usage"), "model": r.get("model"), "reasoning_chars": len(r.get("reasoning") or "")}
            if r.get("finish") != "stop":
                out.append({**rec, "ok": False, "stage": "finish:" + str(r.get("finish"))})
                continue
            why = OC.think_problem(r.get("reasoning"))
            if why:
                out.append({**rec, "ok": False, "stage": why})
                continue
            item = OC.runtime_answer(r.get("content"))
            if not item or not item["sql"] or not item["type"]:
                out.append({**rec, "ok": False, "stage": "format"})
                continue
            hits, periods = env.hints(q)
            res = SV.judge(env, q, item, SYSTEM_THINK + "\n" + env.layers["ctx"] + "\n" + "\n".join(hits + periods))
            comp = F.canonical_thinking(r["reasoning"].strip(), OC.layout(item["type"], item["sql"], item["notes"]))
            out.append({**rec, "ok": bool(res["ok"]), "stage": res["stage"] if not res["ok"] else "ok", "error": res.get("error"),
                        "completion": comp, "sql": item["sql"], "type": item["type"], "mutants": res.get("mutants"),
                        "n_mutants_checked": res.get("n_mutants_checked"), "n_mutants_changed": res.get("n_mutants_changed"),
                        "static": res.get("static"), "route": res.get("route")})
    finally:
        env.close()
    return {"bid": bid, "results": out}


def run(a):
    from sqlv1 import api
    p = json.loads((TDIR / "plan.json").read_text(encoding="utf-8"))
    qs = sorted(p["questions"], key=lambda c: c["u"])
    path = TDIR / f"results.{a.provider}.jsonl"
    done = {x["id"] for x in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else set()
    qs = [c for c in qs if c["id"] not in done]
    if a.limit:
        qs = qs[: a.limit]
    if not qs:
        print("nothing to do")
        return
    left = api.CAP - api.used()
    if len(qs) > left:
        raise SystemExit(f"{len(qs)} requests > {left} left under the ledger cap {api.CAP}")
    if a.max_requests and api.used() + len(qs) > a.max_requests:
        raise SystemExit(f"ledger would pass --max-requests {a.max_requests} (now {api.used()})")
    pool = [json.loads(x) for x in (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    all_specs = collections.defaultdict(list)
    for x in pool:
        all_specs[x["business_id"]].append(x["spec"])
    by = collections.defaultdict(list)
    for c in qs:
        by[c["business_id"]].append(c)
    tot = collections.Counter()
    with cf.ProcessPoolExecutor(a.procs) as ex, open(path, "a", encoding="utf-8") as f:
        futs = [ex.submit(think_shop, (b, by[b], a.provider, all_specs[b])) for b in sorted(by)]
        for fut in cf.as_completed(futs):
            r = fut.result()
            for x in r["results"]:
                f.write(json.dumps({**x, "tag": a.tag}, ensure_ascii=False, default=str) + "\n")
                tot["ok" if x["ok"] else x["stage"]] += 1
            f.flush()
            print(r["bid"], dict(collections.Counter("ok" if x["ok"] else x["stage"] for x in r["results"])), r.get("error") or "", flush=True)
    print("total", dict(tot))


def summary(a):
    out = {}
    for path in sorted(TDIR.glob("results.*.jsonl")):
        xs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        name = path.name[len("results."):-len(".jsonl")]
        prov = name.split(".")[0]
        pi, po = PRICE.get(prov, (0, 0))
        ok = [x for x in xs if x["ok"]]
        cin = sum((x.get("usage") or {}).get("prompt_tokens", 0) for x in xs if not x.get("cached"))
        cout = sum((x.get("usage") or {}).get("completion_tokens", 0) for x in xs if not x.get("cached"))
        outs = [(x.get("usage") or {}).get("completion_tokens", 0) for x in xs]
        out[name] = {"requests": len(xs), "accepted": len(ok), "acceptance": round(len(ok) / max(len(xs), 1), 3),
                     "stages": dict(collections.Counter("ok" if x["ok"] else x["stage"] for x in xs)),
                     "completion_tokens_p50_p90_max": [int(statistics.median(outs)), int(sorted(outs)[int(0.9 * (len(outs) - 1))]), max(outs)] if outs else None,
                     "reasoning_chars_p50": int(statistics.median([x["reasoning_chars"] for x in ok])) if ok else None,
                     "usd_new_calls": round((cin * pi + cout * po) / 1e6, 3), "by_split_ok": dict(collections.Counter(x["split"] for x in ok))}
    db = sqlite3.connect(f"file:{WORK / 'requests.sqlite'}?mode=ro", uri=True)
    led = collections.defaultdict(lambda: [0, 0, 0])
    for prov, st, usage in db.execute("SELECT provider, status, usage FROM requests WHERE tag LIKE 'sqlv1:think:%'"):
        u = json.loads(usage or "{}")
        led[(prov, st)][0] += 1
        led[(prov, st)][1] += u.get("prompt_tokens", 0)
        led[(prov, st)][2] += u.get("completion_tokens", 0)
    out["ledger_think"] = {f"{p}:{st}": {"rows": n, "usd": round((i * PRICE.get(p, (0, 0))[0] + o * PRICE.get(p, (0, 0))[1]) / 1e6, 3)}
                           for (p, st), (n, i, o) in sorted(led.items())}
    out["ledger_total_rows"] = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (TDIR / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def candidates(a):
    """Accepted traces -> <pkg>/candidates/thinking.jsonl, the same shape as candidates/solutions.jsonl (mutation_replay.candidate_id)."""
    xs = [json.loads(x) for x in (TDIR / f"results.{a.provider}.jsonl").read_text(encoding="utf-8").splitlines()]
    seen, out, dropped = set(), [], collections.Counter()
    for x in xs:
        if not x["ok"] or x["id"] in seen:
            continue
        why = OC.think_problem(F.split_thinking(x["completion"])[0])  # the importer's current rules (e.g. echoes of THINK_FMT)
        if why:
            dropped[why] += 1
            continue
        seen.add(x["id"])
        out.append({"id": x["id"], "business_id": x["business_id"], "split": x["split"], "completion": x["completion"], "thinking_mode": "on",
                    "metadata": {"solution_source": "model", "provider": a.provider, "attempt": "thinking", "route": x.get("route"),
                                 "skeleton_sha": hashlib.sha256(V.skeleton(x["sql"]).encode()).hexdigest()[:16],
                                 "question_key": x["qid"], "checks": {"score": 1, "mutants": x.get("mutants"), "n_mutants_checked": x.get("n_mutants_checked"),
                                                                       "n_mutants_changed": x.get("n_mutants_changed"), "static": x.get("static")}}})
    dst = PKG / "candidates" / "thinking.jsonl"
    dst.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out), encoding="utf-8")
    print(json.dumps({"out": str(dst), "candidates": len(out), "dropped_by_importer_rules": dict(dropped),
                      "by_split": dict(collections.Counter(c["split"] for c in out)),
                      "sha256": F.file_hash(dst)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "summary", "candidates"])
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--dev-frac", type=float, default=0.12)
    ap.add_argument("--provider", default="qwen")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-requests", type=int, help="stop before the ledger would pass this many rows (approved budget)")
    a = ap.parse_args()
    {"plan": plan, "run": run, "summary": summary, "candidates": candidates}[a.cmd](a)


if __name__ == "__main__":
    main()
