"""Đo trước token Qwen3-14B cho các dòng ứng viên (KHÔNG thay exporter SFT v2, không ghi sft/; chỉ để báo rủi ro độ dài/che nhãn).

Bước 1 (venv upstream): dựng messages đúng runtime — system = prompts.SYSTEM + "\\n\\nRULEBOOK:\\n" + rulebook BTC (bản sao cục bộ trong data_root),
user = prompts.chat(system, ctx, q) (DATE ANCHORS, gợi ý kỳ, VALUES MATCHING), assistant = completion; ghi ra /dev/shm (chỉ cục bộ).
    python eval/sqlv1/token_premeasure.py build --pkg BUILD/synthetic_sql_v1 --out /dev/shm/sqlv1/msgs.jsonl
Bước 2 (python có transformers, tokenizer đã ghim SHA như eval/check_sft_tokens.py):
    python eval/sqlv1/token_premeasure.py measure --msgs /dev/shm/sqlv1/msgs.jsonl --tokenizer TOK --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()


def _build_shop(args):
    b, cs, qs, system = args
    sys.path.insert(0, str(HERE.parents[1]))
    from sqlv1.up import hints, ingest, prompts
    from sqlv1 import shops as SH
    work = SH.SYNTH_ROOT / "outputs" / "synth_sql_v1"
    info, _ = SH.load_shop(work, b)
    db, meta, layers = SH.open_db(work, b, info)
    out = []
    for c in cs:
        q = dict(qs[c["id"]])
        q["_hits"] = ingest.entity_hits(db, meta, q["question"])
        q["_periods"] = hints.question_hints(q["question"], q["as_of_date"], db, meta)
        msgs = prompts.chat(system, layers["ctx"], q) + [{"role": "assistant", "content": c["completion"]}]
        out.append({"id": c["id"], "split": c["split"], "source": c["metadata"]["solution_source"], "messages": msgs})
    db.close()
    return out


def build(pkg: Path, out: Path, max_shops=None, procs=10):
    import concurrent.futures as cf
    sys.path.insert(0, str(HERE.parents[1]))
    from sqlv1.up import prompts
    root = pkg / "data_root"
    rulebook = (root / "rulebook.md").read_text(encoding="utf-8-sig")
    system = prompts.SYSTEM + "\n\nRULEBOOK:\n" + rulebook
    qs = {json.loads(x)["id"]: json.loads(x) for x in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    cands = [json.loads(x) for x in (pkg / "candidates" / "solutions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by = {}
    for c in cands:
        by.setdefault(c["business_id"], []).append(c)
    jobs = [(b, cs, {c["id"]: qs[c["id"]] for c in cs}, system) for b, cs in sorted(by.items())[:max_shops]]
    with cf.ProcessPoolExecutor(procs) as ex, open(out, "w", encoding="utf-8") as f:
        for rows in ex.map(_build_shop, jobs):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def measure(msgs: Path, tok_path: Path, out: Path, max_length=32768):
    sys.path.insert(0, str(HERE.parents[1]))
    from check_sft_tokens import template_tokens, verify_identity
    from transformers import AutoTokenizer
    verify_identity(tok_path)
    tok = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    rows, fails = [], []
    for line in msgs.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        full = template_tokens(tok, r["messages"], False)
        prefix = template_tokens(tok, r["messages"][:-1], True)
        comp = full[len(prefix):]
        reasons = []
        if full[:len(prefix)] != prefix:
            reasons.append("prefix_not_aligned")
        if comp.count(im_end) != 1 or (comp and comp[-1] != im_end and comp[-2:-1] != [im_end]):
            reasons.append(f"im_end_count={comp.count(im_end)}")
        labels = [-100] * len(prefix) + comp
        if not any(x != -100 for x in labels):
            reasons.append("no_label_tokens")
        if len(full) > max_length:
            reasons.append("too_long")
        if reasons:
            fails.append({"id": r["id"], "reasons": reasons})
        rows.append({"id": r["id"], "split": r["split"], "prompt": len(prefix), "completion": len(comp), "total": len(full)})

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]
    stats = {k: {"p50": q([r[k] for r in rows], 0.5), "p95": q([r[k] for r in rows], 0.95), "max": max(r[k] for r in rows),
                 "mean": round(statistics.mean(r[k] for r in rows), 1)} for k in ("prompt", "completion", "total")}
    rep = {"kind": "premeasure (not the SFT v2 exporter gate)", "tokenizer": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
           "enable_thinking": False, "add_generation_prompt_for_prefix": True, "n_rows": len(rows), "failures": len(fails), "fail_examples": fails[:10],
           "stats": stats, "longest": sorted(rows, key=lambda r: -r["total"])[:5]}
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_rows", "failures", "stats")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "measure"])
    ap.add_argument("--pkg", type=Path)
    ap.add_argument("--msgs", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-shops", type=int)
    a = ap.parse_args()
    build(a.pkg, a.out, a.max_shops) if a.cmd == "build" else measure(a.msgs, a.tokenizer, a.out)
