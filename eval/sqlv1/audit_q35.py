"""Audit bởi LLM độc lập (subagent Claude, KHÔNG phải người) cho gói SFT Qwen3.5 (build/qwen35_sft_v1): mẫu mới gồm dòng v1.1
(câu unanswerable + câu giá dài), dòng p1 (prompt có ví dụ BTC) và dòng thinking.

    python eval/sqlv1/audit_q35.py sample --tag T [--p1 40 --thinking 40 --no-v11]   → outputs/synth_sql_v1/audit_q35/sample_T.jsonl + packets_T/<n>.json
    python eval/sqlv1/audit_q35.py summarize                         → outputs/synth_sql_v1/audit_q35/summary.json (Wilson 95%)

Gói cho auditor chỉ chứa dữ liệu tổng hợp: câu hỏi, as_of_date, ctx của shop tổng hợp (README + SCHEMA + VIEWS + DEBTS), completion
(dòng thinking: cả vết suy nghĩ), gold để tham khảo. KHÔNG có rulebook BTC hay ví dụ few-shot BTC (quy ước: eval/sqlv1/rules_spec.md).
Rubric: eval/sqlv1/AUDIT_RUBRIC.md; dòng thinking thêm tiêu chí vet_suy_nghi.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "outputs" / "synth_sql_v1"
AUD = WORK / "audit_q35"
PKG = controller_root() / "build"
V11_FIRST, V11_LAST = 4626, 4785  # augment_summary.new_ids: the 160 questions added in v1.1 (60 unanswerable + 100 long price)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_rows(d):
    return [json.loads(x) for s in ("train", "dev") for x in (d / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def sample(a):
    rows = read_rows(a.build)
    sel = {x["id"]: x for x in map(json.loads, (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines()) if x.get("id")}
    v11 = [r for r in rows if V11_FIRST <= int(r["metadata"]["question_id"].split(":q")[1]) <= V11_LAST and not r["thinking_mode"]] if a.v11 else []
    p1 = sorted([r for r in rows if (r["metadata"].get("offline") or {}).get("prompt_variant") == "p1" and not r["thinking_mode"] and r not in v11
                 and r["metadata"]["derivation"] != "offline_program_template"],  # templates of raw-table slots were regenerated (retemplate.py)
                key=lambda r: h("p1|" + r["id"]))[: a.p1]
    th = sorted([r for r in rows if r["thinking_mode"]], key=lambda r: h("th|" + r["id"]))[: a.thinking]
    rt = sorted([r for r in rows if r["metadata"]["derivation"] == "offline_program_template"
                 and "_item_in" not in r["completion"] and " IN (SELECT " in r["completion"].split("```sql")[-1]],
                key=lambda r: h("rt|" + r["id"]))[: a.retemplated]
    items = []
    for kind, rs in (("v1_1", v11), ("p1", p1), ("thinking", th), ("retemplated", rt)):
        for r in rs:
            m = r["metadata"]
            s = sel[m["question_id"]]
            items.append({"row": r["id"], "id": m["question_id"], "kind": kind, "business_id": r["business_id"], "split": m["split"],
                          "question": m["question"], "as_of_date": s["as_of_date"], "gold": s["gold"], "slot": s["spec"].get("slot"),
                          "completion": r["completion"], "derivation": m["derivation"], "prompt_variant": (m.get("offline") or {}).get("prompt_variant")})
    AUD.mkdir(parents=True, exist_ok=True)
    (AUD / f"sample_{a.tag}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    # packets by shop, about a.per items each, so one auditor reads each shop's ctx once
    by = collections.defaultdict(list)
    for x in items:
        by[x["business_id"]].append(x)
    packets, cur = [], []
    for b in sorted(by, key=lambda b: -len(by[b])):
        cur += by[b]
        if len(cur) >= a.per:
            packets.append(cur)
            cur = []
    if cur:
        packets.append(cur)
    (AUD / f"packets_{a.tag}").mkdir(exist_ok=True)
    for i, p in enumerate(packets):
        (AUD / f"packets_{a.tag}" / f"{i:02d}.json").write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"items": len(items), "by_kind": dict(collections.Counter(x["kind"] for x in items)), "packets": len(packets),
                      "shops": len(by)}, ensure_ascii=False))


def summarize(a):
    items = {x["row"]: x for f in sorted(AUD.glob("sample_*.jsonl")) for x in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
    in_build = {r["id"] for r in read_rows(a.build)}
    gone = sorted(set(items) - in_build)  # rows the final build no longer has (producer QA, regenerated templates): not counted
    items = {k: v for k, v in items.items() if k in in_build}
    ver = {}
    for f in sorted((AUD / "verdicts").glob("*.jsonl")):
        for x in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            if x.get("row") in items:
                ver[x["row"]] = x
    out = {"auditor": "subagent Claude độc lập theo eval/sqlv1/AUDIT_RUBRIC.md (+ vet_suy_nghi cho dòng thinking); KHÔNG phải người",
           "build": str(a.build), "sample": len(items), "judged": len(ver), "missing": sorted(set(items) - set(ver))[:20],
           "sampled_rows_not_in_build": len(gone)}
    for kind in (None, "v1_1", "p1", "thinking", "retemplated"):
        vs = [v for r, v in ver.items() if kind is None or items[r]["kind"] == kind]
        k = sum(bool(v.get("pass")) for v in vs)
        lo, hi = wilson(k, len(vs))
        out[kind or "all"] = {"n": len(vs), "pass": k, "rate": round(k / max(len(vs), 1), 4), "wilson95": [round(lo, 4), round(hi, 4)],
                              "fails": [{"row": v["row"], "ly_do": v.get("ly_do")} for v in vs if not v.get("pass")]}
    (AUD / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "fails"}) for k, v in out.items()}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "summarize"])
    ap.add_argument("--build", type=Path, default=PKG / "qwen35_sft_v1" / "sft")
    ap.add_argument("--p1", type=int, default=40)
    ap.add_argument("--thinking", type=int, default=40)
    ap.add_argument("--retemplated", type=int, default=0)
    ap.add_argument("--per", type=int, default=30)
    ap.add_argument("--tag", default="all", help="sample_<tag>.jsonl + packets_<tag>/ (summarize merges every sample_*.jsonl)")
    ap.add_argument("--no-v11", dest="v11", action="store_false")
    a = ap.parse_args()
    {"sample": sample, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
