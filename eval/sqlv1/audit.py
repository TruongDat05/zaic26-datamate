"""Audit chất lượng (Bước 7): mẫu phân tầng + mồi sai đã biết + auditor LLM khác họ với họ đã giải; thống kê Wilson.

    python eval/sqlv1/audit.py sample --n 160            # audit/sample.jsonl (audit bởi LLM độc lập, subagent) + audit/baits.jsonl (50 mồi, đã xác nhận sai cục bộ)
    python eval/sqlv1/audit.py llm --auditor gemini      # audit/llm_verdicts.jsonl, audit/llm_summary.json
    python eval/sqlv1/audit.py summarize                 # gộp audit bởi LLM độc lập (audit/manual_verdicts.jsonl, tên file cũ) + LLM
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

from sqlv1.up import hints, sqlexec, scoring  # noqa: E402
from sqlv1 import shops as SH, world as WB  # noqa: E402
from sqlv1.questions import WORK  # noqa: E402
import llm  # noqa: E402

AUD = WORK / "audit"
PKG = controller_root() / "build/synthetic_sql_v1"
FAMILY = llm.FAMILY


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def rows():
    sel = {json.loads(x)["id"]: json.loads(x) for x in (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    cands = [json.loads(x) for x in (PKG / "candidates" / "solutions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for c in cands:
        q = sel[c["id"]]
        out.append({"id": c["id"], "row": f"{c['id']}#{c['metadata']['skeleton_sha']}", "business_id": c["business_id"], "split": c["split"],
                    "question": q["question"], "as_of_date": q["as_of_date"], "gold": q["gold"], "spec": q["spec"], "completion": c["completion"],
                    "slot": q["spec"]["slot"], "family": q["family"], "industry": q["industry"], "source": c["metadata"]["solution_source"],
                    "provider": c["metadata"]["provider"], "route": c["metadata"]["route"]})
    return out


def sample(n: int, seed: int = 4242):
    rng = random.Random(seed)
    R = rows()
    by = collections.defaultdict(list)
    for r in R:
        by[r["slot"]].append(r)
    alloc = {k: max(1, round(n * len(v) / len(R))) for k, v in by.items()}
    while sum(alloc.values()) > n:
        k = max(alloc, key=lambda k: alloc[k])
        alloc[k] -= 1
    while sum(alloc.values()) < n:
        k = max(by, key=lambda k: len(by[k]) / alloc[k])
        alloc[k] += 1
    chosen = []
    for slot in sorted(by):  # trong tầng: rải họ schema, ngành, nguồn lời giải
        v = by[slot][:]
        rng.shuffle(v)
        seen = collections.Counter()
        picked = []
        for _ in range(min(alloc[slot], len(v))):
            best = min(v, key=lambda r: (seen[("f", r["family"])] + seen[("i", r["industry"])] + seen[("s", r["source"])] + seen[("p", r["provider"])]))
            v.remove(best)
            picked.append(best)
            seen.update([("f", best["family"]), ("i", best["industry"]), ("s", best["source"]), ("p", best["provider"])])
        chosen += picked
    tpl = [r for r in R if r["source"] == "template" and r not in chosen]
    if tpl and not any(r["source"] == "template" for r in chosen):
        chosen.append(rng.choice(tpl))
    AUD.mkdir(parents=True, exist_ok=True)
    with open(AUD / "sample.jsonl", "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    baits = make_baits([r for r in R if r not in chosen], rng, 50)
    with open(AUD / "baits.jsonl", "w", encoding="utf-8") as f:
        for r in baits:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"sample": len(chosen), "by_source": collections.Counter(r["source"] for r in chosen), "baits": len(baits),
                      "bait_kinds": collections.Counter(r["bait"] for r in baits)}, ensure_ascii=False, default=str))


CORRUPT = [
    ("status_all", lambda s: re.sub(r"\bv_sales\b(?!_all)", "v_sales_all", s) if re.search(r"\bv_sales\b(?!_all)", s) else re.sub(r"\bv_orders\b(?!_all)", "v_orders_all", s)),
    ("end_inclusive", lambda s: re.sub(r"<\s*DATE '(\d{4}-\d{2}-\d{2})'", lambda m: f"<= DATE '{m.group(1)}'", s, count=1)),
    ("start_shift", lambda s: re.sub(r">=\s*DATE '(\d{4})-(\d{2})-(\d{2})'", lambda m: f">= DATE '{m.group(1)}-{m.group(2)}-{min(28, int(m.group(3)) + 1):02d}'", s, count=1)),
    ("order_asc", lambda s: re.sub(r"\bDESC\b", "ASC", s, count=1)),
    ("count_lines", lambda s: s.replace("COUNT(DISTINCT order_id)", "COUNT(order_id)").replace("COUNT(*) FROM v_orders", "COUNT(*) FROM v_sales")),
    ("keep_walkins", lambda s: s.replace("customer_key IS NOT NULL AND ", "").replace("WHERE customer_key IS NOT NULL", "WHERE TRUE")),
    ("pct_no_minus", lambda s: re.sub(r"\(\s*(\w+)\.v\s*-\s*(\w+)\.v\s*\)\s*/\s*(\w+)\.v", r"\1.v / \3.v", s)),
]


def make_baits(R, rng, n):
    """Mồi sai đã biết: SQL đúng bị làm hỏng theo lỗi hay gặp, chạy lại cục bộ và xác nhận ra đáp án sai."""
    rng.shuffle(R)
    out, used = [], collections.Counter()
    envs = {}
    for r in R:
        if len(out) >= n:
            break
        kind, sql = sqlexec.parse(r["completion"])
        if not sql:
            continue
        opts = [(k, f(sql)) for k, f in CORRUPT if f(sql) != sql]
        rng.shuffle(opts)
        opts.sort(key=lambda kv: used[kv[0]])
        for k, bad in opts[:2]:
            if r["business_id"] not in envs:
                info, rec = SH.load_shop(WORK, r["business_id"])
                envs[r["business_id"]] = SH.open_db(WORK, r["business_id"], info)[0]
            db = envs[r["business_id"]]
            text = r["completion"].replace(sql, bad)
            try:
                p = sqlexec.execute(db, text, r["question"])
                wrong = scoring.score(p, r["gold"]) != 1
            except Exception:  # noqa: BLE001
                wrong = True
            if wrong:
                out.append({**r, "row": r["row"] + "#bait", "completion": text, "bait": k})
                used[k] += 1
                break
    for db in envs.values():
        db.close()
    return out


def audit_prompt(ctxs: dict, items: list[dict]) -> list[dict]:
    from sqlv1.solve import RULEBOOK_API
    head = ("Bạn kiểm định lời giải SQL (DuckDB) cho câu hỏi của chủ cửa hàng trên dữ liệu giả lập. KHÔNG có đáp án mẫu; tự đọc câu hỏi, quy ước và SQL.\n"
            "Với mỗi mục đánh giá ba tiêu chí (true/false):\n"
            "- dung_nghia: SQL tính đúng thứ câu hỏi hỏi (đúng số đo, đối tượng, kỳ: cửa sổ ngày đúng với DATE ANCHORS, đúng kiểu TYPE).\n"
            "- dung_quy_uoc: tuân QUY ƯỚC dưới đây (lớp đơn hợp lệ, đếm theo mã đơn, khách lẻ, quy đổi đơn vị, làm tròn, đồng hạng...).\n"
            "- sql_sach: một câu SELECT/WITH gọn, không ghi cứng đáp án, Notes ngắn và đúng với SQL.\n"
            "Nghi ngờ có lỗi thì đặt false và nêu lý do ngắn.\n\nQUY ƯỚC:\n" + RULEBOOK_API)
    shops = "\n\n".join(f"## DỮ LIỆU CỬA HÀNG {b}\n{c}" for b, c in ctxs.items())
    body = "\n\n".join(f"### {it['blind']} (cửa hàng {it['business_id']})\nDATE ANCHORS:\n{hints.date_anchors(it['as_of_date'])}\nQUESTION: {it['question']}\nANSWER:\n{it['completion']}"
                       for it in items)
    tail = '\n\nTrả về đúng một JSON: [{"id": "<mã mục>", "dung_nghia": true, "dung_quy_uoc": true, "sql_sach": true, "ly_do": "..."}, ...] đủ mọi mục, không giải thích thêm.'
    return [{"role": "user", "content": head + "\n\n" + shops + "\n\nCÁC MỤC:\n" + body + tail}]


def run_llm(auditor: str, fallback: str = "deepseek", per_call: int = 8, shops_per_call: int = 3, workers: int = 6):
    """Auditor chính cho mọi mục; mục do cùng họ auditor giải thì chuyển sang fallback (khác họ)."""
    from sqlv1 import api
    items = [json.loads(x) for x in (AUD / "sample.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    items += [json.loads(x) for x in (AUD / "baits.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rng = random.Random(77)
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["blind"] = f"m{i:03d}"
        fam = FAMILY.get(it["provider"], it["provider"])
        it["auditor"] = auditor if FAMILY.get(auditor) != fam else fallback
        assert FAMILY.get(it["auditor"]) != fam
    jobs = []
    for aud in sorted({it["auditor"] for it in items}):
        by = collections.defaultdict(list)
        for it in items:
            if it["auditor"] == aud:
                by[it["business_id"]].append(it)
        cur, cur_shops = [], []
        for b in sorted(by, key=lambda b: -len(by[b])):
            for it in by[b]:
                if len(cur) >= per_call or (b not in cur_shops and len(cur_shops) >= shops_per_call):
                    jobs.append((aud, cur_shops, cur))
                    cur, cur_shops = [], []
                if b not in cur_shops:
                    cur_shops.append(b)
                cur.append(it)
        if cur:
            jobs.append((aud, cur_shops, cur))

    def call(job):
        aud, shops, chunk = job
        ctxs = {b: json.loads((WORK / "shops" / b / "shop.json").read_text(encoding="utf-8"))["ctx"] for b in shops}
        r, _ = api.call(WORK, "audit", "+".join(shops)[:100], aud, audit_prompt(ctxs, chunk), think="off", max_tokens=12000)
        try:
            out = {x["id"]: x for x in llm.extract_json(r.get("content") or "") if isinstance(x, dict)}
        except ValueError:
            out = {}
        return aud, chunk, out
    res = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for aud, chunk, out in ex.map(call, jobs):
            for it in chunk:
                v = out.get(it["blind"])
                verdict = None if not v else all(bool(v.get(k)) for k in ("dung_nghia", "dung_quy_uoc", "sql_sach"))
                res.append({"row": it["row"], "id": it["id"], "bait": it.get("bait"), "auditor": aud, "solver": it["provider"], "pass": verdict, "verdict": v})
    with open(AUD / "llm_verdicts.jsonl", "w", encoding="utf-8") as f:
        for x in res:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summ = {"calls": len(jobs), "by_auditor": {}}
    for aud in sorted({x["auditor"] for x in res}) + ["all"]:
        rs = [x for x in res if aud == "all" or x["auditor"] == aud]
        real = [x for x in rs if not x["bait"] and x["pass"] is not None]
        bait = [x for x in rs if x["bait"] and x["pass"] is not None]
        k, n = sum(x["pass"] for x in real), len(real)
        caught = sum(not x["pass"] for x in bait)
        summ["by_auditor"][aud] = {"family": FAMILY.get(aud, "mixed"), "items": len(rs), "missing": sum(x["pass"] is None for x in rs),
                                   "sensitivity": round(caught / len(bait), 4) if bait else None, "baits": len(bait), "baits_caught": caught,
                                   "sensitivity_wilson95": wilson(caught, len(bait)),
                                   "sensitivity_by_kind": {kd: f"{sum(not x['pass'] for x in bait if x['bait'] == kd)}/{sum(1 for x in bait if x['bait'] == kd)}" for kd in sorted({x['bait'] for x in bait})},
                                   "pass_rate": round(k / n, 4) if n else None, "pass_wilson95": wilson(k, n), "n_real": n,
                                   "pass_rate_usable": bool(bait) and caught / len(bait) >= 0.8}
    a = summ["by_auditor"]["all"]
    summ.update(auditor=auditor, auditor_family=FAMILY.get(auditor), sensitivity=a["sensitivity"], baits=a["baits"], baits_caught=a["baits_caught"],
                pass_rate=a["pass_rate"], pass_wilson95=a["pass_wilson95"], pass_rate_usable=a["pass_rate_usable"], fallback=fallback)
    (AUD / "llm_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


def summarize():
    out = {}
    mp = AUD / "manual_verdicts.jsonl"
    if mp.exists():
        v = [json.loads(x) for x in mp.read_text(encoding="utf-8").splitlines() if x.strip()]
        k, n = sum(bool(x["pass"]) for x in v), len(v)
        out["manual"] = {"n": n, "pass": k, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n),
                         "fails": [x for x in v if not x["pass"]]}
    if (AUD / "llm_summary.json").exists():
        out["llm"] = json.loads((AUD / "llm_summary.json").read_text(encoding="utf-8"))
    (AUD / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "fails"} for k, v in out.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "llm", "summarize"])
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--auditor", default="gemini")
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()
    {"sample": lambda: sample(a.n, a.seed), "llm": lambda: run_llm(a.auditor), "summarize": summarize}[a.cmd]()
