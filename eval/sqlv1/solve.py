"""Lời giải SQL qua API cho câu hỏi đã có gold (gold KHÔNG vào prompt), kiểm cục bộ bằng upstream, sửa một lượt, lưới an toàn SQL mẫu.

Prompt gửi API: system = prompts.SYSTEM (luật SQL của đội) + RULEBOOK = rulebook_api.md (quy ước diễn đạt lại, không chép rulebook BTC);
user = ctx shop tổng hợp (README + SCHEMA + CANONICAL VIEWS + DEBTS như runtime) + từng câu kèm DATE ANCHORS, gợi ý kỳ, VALUES MATCHING.

    python eval/sqlv1/solve.py run --pool POOL.jsonl --provider gptsol [--procs 8]
    python eval/sqlv1/solve.py template --pool POOL.jsonl
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.up import hints, prompts, sqlexec  # noqa: E402
from sqlv1 import render as RD, shops as SH, sqlgen as G, verify as V  # noqa: E402

WORK = SH.SYNTH_ROOT / "outputs" / "synth_sql_v1"
RULEBOOK_API = (Path(__file__).parent / "rulebook_api.md").read_text(encoding="utf-8")
SYSTEM_API = prompts.SYSTEM + "\n\nRULEBOOK:\n" + RULEBOOK_API
BATCH = 10
OUT_FMT = ('Answer every question. Return ONE JSON array only: [{"id": "<id>", "notes": ["<short note>", ...], "type": "<TYPE>", "sql": "<one DuckDB query>"}, ...]. '
           "notes: at most 6 short lines (rules applied, tables/columns, exact date window, grain, status filter, units), no SQL in notes, "
           "name a rule by its content (e.g. 'valid orders only'), never by a letter or section number. "
           "type and sql follow the format rules above (TYPE value and the query that would go in the ```sql block). Use the DATE ANCHORS of each question.")


def question_block(env: V.ShopEnv, q: dict) -> str:
    hits, periods = env.hints(q)
    gloss = [f"{w} = {prompts.SLANG[w]}" for w in dict.fromkeys(re.findall(r"\w+", q["question"].lower())) if w in prompts.SLANG] + prompts.money_gloss(q["question"])
    return (f"### id: {q['qid']}\nDATE ANCHORS:\n" + hints.date_anchors(q["as_of_date"]) + "".join("\n" + p for p in periods)
            + ("\nVALUES MATCHING THE QUESTION:\n" + "\n".join(hits) if hits else "")
            + f"\nQUESTION: {q['question']}" + (f"\n(abbreviations: {'; '.join(gloss)})" if gloss else ""))


def solve_messages(env: V.ShopEnv, qs: list[dict]) -> list[dict]:
    user = env.layers["ctx"] + "\n\n" + "\n\n".join(question_block(env, q) for q in qs) + "\n\n" + OUT_FMT
    return [{"role": "system", "content": SYSTEM_API}, {"role": "user", "content": user}]


def repair_messages(env: V.ShopEnv, items: list[tuple[dict, dict, str]]) -> list[dict]:
    """items: (câu, lời giải cũ, lỗi). Lượt sửa: lỗi + prompts.repair_hint, như runtime."""
    parts = []
    for q, c, err in items:
        parts.append(question_block(env, q) + f"\nYOUR ANSWER:\nTYPE: {c.get('type')}\n```sql\n{c.get('sql')}\n```\nERROR: {err}\n{prompts.repair_hint(err)}"
                     "Fix the query. Answer again in the same format.")
    user = env.layers["ctx"] + "\n\n" + "\n\n".join(parts) + "\n\n" + OUT_FMT
    return [{"role": "system", "content": SYSTEM_API}, {"role": "user", "content": user}]


def parse_answers(text: str) -> dict:
    import llm
    try:
        arr = llm.extract_json(text or "")
    except ValueError:
        return {}
    out = {}
    for x in arr if isinstance(arr, list) else []:
        if not isinstance(x, dict) or "id" not in x:
            continue
        notes = x.get("notes") or []
        if isinstance(notes, str):
            notes = [n.strip(" -") for n in notes.splitlines() if n.strip(" -")]
        out[str(x["id"])] = {"type": str(x.get("type") or "").strip().lower(), "sql": str(x.get("sql") or "").strip().strip("`").strip(),
                             "notes": [clean_note(str(n)) for n in notes if clean_note(str(n))]}
    return out


RULE_REF = re.compile(r"(?i)\(?\b(?:rule(?:book)?|mục|quy ước|luật|section)\s*[A-J](?:\d+)?\b\)?[:.,]?\s*|§\s*\d+[:.]?\s*")


def clean_note(n: str) -> str:
    """Bỏ trích dẫn mã mục của rulebook_api (runtime thấy rulebook khác, đánh số khác)."""
    return re.sub(r"\s{2,}", " ", RULE_REF.sub("", n)).strip(" -;,:")


def completion(c: dict) -> str:
    return G.completion(c["type"], c["sql"].rstrip(";").strip(), c["notes"])


def judge(env, q, c, visible) -> dict:
    text = completion(c)
    _, periods = env.hints(q)
    st = V.static_check(c["sql"], c["notes"], q["question"], q["as_of_date"], periods, visible, q["gold"])
    res = V.check_solution(env, q, text)
    if res["ok"] and st:
        res.update(ok=False, stage="static:" + st[0])
    res["static"] = st
    res["route"], res["skeleton"] = V.route(c["sql"]), V.skeleton(c["sql"])
    return res


def solve_shop(args) -> dict:
    """Một shop, một provider: gọi API theo lô, kiểm, sửa một lượt các lời giải lỗi chạy, ghi kết quả."""
    from sqlv1 import api
    bid, qs, provider, work, all_specs = args
    env = V.ShopEnv(work, bid, all_specs)
    visible_base = env.layers["ctx"]
    out, stats = [], collections.Counter()
    try:
        for b0 in range(0, len(qs), BATCH):
            batch = qs[b0:b0 + BATCH]
            r, _ = api.call(work, "solve", bid, provider, solve_messages(env, batch), think="off", max_tokens=16000)
            ans = parse_answers(r.get("content") or "")
            stats["calls"] += 1
            stats["json_ok" if ans else "json_bad"] += 1
            fix = []
            for q in batch:
                c = ans.get(q["qid"])
                if not c or not c["sql"] or not c["type"]:
                    stats["missing"] += 1
                    out.append({"qid": q["qid"], "provider": provider, "attempt": "first", "ok": False, "stage": "missing"})
                    continue
                hits, periods = env.hints(q)
                vis = visible_base + "\n" + "\n".join(hits + periods)
                res = judge(env, q, c, vis)
                rec = {"qid": q["qid"], "provider": provider, "attempt": "first", "completion": completion(c), "sql": c["sql"], "type": c["type"], **res}
                out.append(rec)
                stats[res["stage"] if not res["ok"] else "ok"] += 1
                if not res["ok"] and res["stage"] in ("exec", "exec2", "parse"):
                    fix.append((q, c, res.get("error") or "invalid answer"))
            if fix:
                r2, _ = api.call(work, "repair", bid, provider, repair_messages(env, fix), think="off", max_tokens=16000)
                ans2 = parse_answers(r2.get("content") or "")
                stats["repair_calls"] += 1
                for q, c, err in fix:
                    c2 = ans2.get(q["qid"])
                    if not c2 or not c2["sql"]:
                        continue
                    if not c2["notes"]:
                        c2["notes"] = c["notes"]
                    hits, periods = env.hints(q)
                    res = judge(env, q, c2, visible_base + "\n" + "\n".join(hits + periods))
                    out.append({"qid": q["qid"], "provider": provider, "attempt": "repair", "completion": completion(c2), "sql": c2["sql"], "type": c2["type"], **res})
                    stats["repair_ok" if res["ok"] else "repair_" + (res["stage"] or "?").split(":")[0]] += 1
    finally:
        env.close()
    return {"bid": bid, "provider": provider, "results": out, "stats": dict(stats)}


def template_shop(args) -> dict:
    """Lưới an toàn: SQL mẫu tất định theo slot, qua cùng mọi bước kiểm."""
    bid, qs, work, all_specs = args
    env = V.ShopEnv(work, bid, all_specs)
    raw = RD.raw_info(env.info["roles"], RD.relations(env.layers["N"]))
    out = []
    try:
        for q in qs:
            try:
                kind, sql, notes = G.for_spec(q["spec"], env.rec, raw)
            except G.Unsupported:
                out.append({"qid": q["qid"], "provider": "template", "attempt": "template", "ok": False, "stage": "unsupported"})
                continue
            c = {"type": kind, "sql": sql, "notes": notes}
            hits, periods = env.hints(q)
            res = judge(env, q, c, env.layers["ctx"] + "\n" + "\n".join(hits + periods))
            out.append({"qid": q["qid"], "provider": "template", "attempt": "template", "completion": completion(c), "sql": sql, "type": kind, **res})
    finally:
        env.close()
    return {"bid": bid, "provider": "template", "results": out}


def load_pool(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "template"])
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--provider", default="gptsol")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--only-unsolved", action="store_true", help="chỉ câu chưa có lời giải đạt (theo các file kết quả đã có)")
    ap.add_argument("--max-shops", type=int)
    a = ap.parse_args()
    pool = load_pool(a.pool)
    all_specs = collections.defaultdict(list)  # m_target dựng từ MỌI câu của shop trong kho: cùng bản đột biến ở mọi lượt giải
    for q in pool:
        all_specs[q["business_id"]].append(q["spec"])
    res_dir = WORK / "solutions"
    res_dir.mkdir(parents=True, exist_ok=True)
    if a.only_unsolved:
        solved = set()
        for f in res_dir.glob("*.jsonl"):
            for x in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
                if x.get("ok"):
                    solved.add(x["qid"])
        pool = [q for q in pool if q["qid"] not in solved]
    by = collections.defaultdict(list)
    for q in pool:
        by[q["business_id"]].append(q)
    name = "template" if a.cmd == "template" else a.provider
    path = res_dir / f"{a.pool.stem}.{name}.jsonl"
    done = set()
    if path.exists():  # chạy tiếp: bỏ shop đã có kết quả của lượt này (lời gọi API đã cache)
        pool_qids = {q["qid"]: q["business_id"] for q in pool}
        got = collections.defaultdict(set)
        for x in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            if x["qid"] in pool_qids:
                got[pool_qids[x["qid"]]].add(x["qid"])
        done = {b for b, qs in got.items() if qs >= {q["qid"] for q in by[b]}}
    shops = [b for b in sorted(by) if b not in done]
    shops = shops[: a.max_shops] if a.max_shops else shops
    tot = collections.Counter()
    with cf.ProcessPoolExecutor(a.procs) as ex, open(path, "a", encoding="utf-8") as f:
        jobs = [(b, by[b], WORK, all_specs[b]) if a.cmd == "template" else (b, by[b], a.provider, WORK, all_specs[b]) for b in shops]
        futs = [ex.submit(template_shop if a.cmd == "template" else solve_shop, j) for j in jobs]
        for fut in cf.as_completed(futs):
            r = fut.result()
            for x in r["results"]:
                f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
            f.flush()
            st = r.get("stats") or dict(collections.Counter("ok" if x["ok"] else x["stage"] for x in r["results"]))
            tot.update(st)
            print(r["bid"], st, flush=True)
    print("total", dict(tot))


if __name__ == "__main__":
    main()
