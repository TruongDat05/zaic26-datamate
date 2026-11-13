"""Tính lại MỌI cổng §9 trên các dòng SFT thật sự được exporter nhận (không dùng số ứng viên trước export).

    python eval/sqlv1/sft_gates.py --sft build/synthetic_sql_v1/sft --tokens build/synthetic_sql_v1/sft/token_validation.json
      → outputs/synth_sql_v1/sft_v2/gates.json

Nguồn: sft/{train,dev,rejected}.jsonl + sft_manifest.json (exporter), mutation_report.jsonl (chạy lại trên snapshot), audit
(mẫu vòng 2: audit bởi LLM độc lập = subagent Claude theo AUDIT_RUBRIC.md; audit LLM khác họ = gemini + 50 mồi), token_validation.json
(train_lora.py --validate-only với tokenizer thật đúng revision), btc_align trên snapshot mới, gói FINAL (chỉ đọc, kiểm rò rỉ).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

from sqlv1 import package as PK  # noqa: E402
from sqlv1.audit import wilson  # noqa: E402
from sqlv1.questions import WORK, fold  # noqa: E402

FOXEL = controller_root()
import os  # noqa: E402

OUT = WORK / os.environ.get("SQLV1_OUT", "sft_v2")


def jl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def dist(c: collections.Counter):
    n = sum(c.values())
    return {k: round(v / n, 4) for k, v in c.most_common()} if n else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=Path, default=FOXEL / "build" / "synthetic_sql_v1" / "sft")
    ap.add_argument("--tokens", type=Path)
    ap.add_argument("--final", type=Path, default=FOXEL / "build" / "synthetic_sql_v1_final")
    ap.add_argument("--align", type=Path, default=OUT / "btc_align_snapshot.json")
    ap.add_argument("--audit-dir", type=Path, help="audit_q35 layout (sample.jsonl + verdicts/*.jsonl, eval/sqlv1/audit_q35.py) instead of audit/ round 2")
    ap.add_argument("--btc-train", type=Path, default=FOXEL / "data" / "train", help="competition train set (read locally: hold-out texts)")
    ap.add_argument("--splits", type=Path, default=FOXEL / "splits")
    a = ap.parse_args()
    man = json.loads((a.sft / "sft_manifest.json").read_text(encoding="utf-8"))
    rows = {s: jl(a.sft / f"{s}.jsonl") for s in ("train", "dev")}
    rej = jl(a.sft / "rejected.jsonl")
    sel = {x["id"]: x for x in jl(WORK / "selected.jsonl") if x.get("id")}
    tgt = PK.mining_targets()
    g = {"build_id": man["build_id"], "counts": {}, "splits": {}}
    for s, rs in rows.items():
        qs = [sel[r["metadata"]["question_id"]] for r in rs]  # theo dòng
        uq = {r["metadata"]["question_id"] for r in rs}
        uqs = [sel[q] for q in sorted(uq)]  # theo câu (cùng định nghĩa với báo cáo trước export: eda.py)
        f = [PK.features(x) for x in uqs]
        fr = [PK.features(x) for x in qs]
        der = collections.Counter(r["metadata"]["derivation"] for r in rs)
        prov = collections.Counter((r["metadata"]["solver"]["mode"], r["metadata"]["solver"]["model_key"], r["metadata"]["solver"].get("model")) for r in rs)
        mut = [r["metadata"]["mutation"] for r in rs]
        g["splits"][s] = {
            "rows": len(rs), "questions": len(uq), "shops": len({r["business_id"] for r in rs}),
            "industry_core_row_share": round(sum(x["industry"] in PK.WB.CORE for x in qs) / max(1, len(qs)), 4),
            "period": dist(collections.Counter(ff["period"] for ff in f)), "skill": dist(collections.Counter(ff["skill"] for ff in f)),
            "type": dist(collections.Counter(ff["type"] for ff in f)), "word_bins": dist(collections.Counter(ff["words"] for ff in f)),
            "L1": {d: round(PK.l1(collections.Counter(ff[d] for ff in f), len(f), tgt[d]), 4) for d in PK.DIM_W} if f else {},
            "L1_by_row": {d: round(PK.l1(collections.Counter(ff[d] for ff in fr), len(fr), tgt[d]), 4) for d in PK.DIM_W} if fr else {},
            "period_by_row": dist(collections.Counter(ff["period"] for ff in fr)),
            "template_rows": der.get("offline_program_template", 0), "template_share": round(der.get("offline_program_template", 0) / max(1, len(rs)), 4),
            "by_solver": {f"{m}:{k}:{mod}": n for (m, k, mod), n in prov.most_common()},
            "exact_dup_question_share": round(1 - len({fold(sel[q]['question']) for q in uq}) / max(1, len(uq)), 4),
            "mutation": {"rows_discriminating": sum(m["discriminating"] for m in mut), "share_discriminating": round(sum(m["discriminating"] for m in mut) / max(1, len(mut)), 4),
                         "share_discriminating_answerable": round(sum(r["metadata"]["mutation"]["discriminating"] for r in rs if r["metadata"]["answer_type"] != "unanswerable")
                                                                  / max(1, sum(r["metadata"]["answer_type"] != "unanswerable" for r in rs)), 4),
                         "n_checked_hist": dict(collections.Counter(m["n_checked"] for m in mut))},
            "route": dict(collections.Counter(r["metadata"]["route"] for r in rs)), "answer_type": dict(collections.Counter(r["metadata"]["answer_type"] for r in rs)),
            "repaired_rows": sum(r["metadata"]["repaired"] for r in rs),
            "prompt_variant": dict(collections.Counter(((r["metadata"].get("offline") or {}).get("prompt_variant") or "p0", r["thinking_mode"]) and
                                                       f'{(r["metadata"].get("offline") or {}).get("prompt_variant") or "p0"}{"+think" if r["thinking_mode"] else ""}' for r in rs)),
            "thinking_rows": sum(r["thinking_mode"] for r in rs),
        }
    g["counts"] = {"train_rows": len(rows["train"]), "dev_rows": len(rows["dev"]), "rejected_entries": len(rej),
                   "rejected_reasons": man["rejected_reasons"], "manifest_counts": man["counts"]}
    # rò rỉ: FINAL (chỉ đọc) không chung shop/lineage/câu hỏi với SFT; câu FINAL không nằm trong prompt/completion SFT hay request API
    fq = jl(a.final / "data_root" / "questions.jsonl")
    fman = json.loads((a.final / "data_root" / "manifest.json").read_text(encoding="utf-8"))
    f_shops = {q["business_id"] for q in fq}
    f_lin = {v["parent_example_id"] for v in fman["businesses"].values()}
    all_rows = rows["train"] + rows["dev"]
    texts = {fold(q["question"]) for q in fq}
    in_rows = sum(1 for r in all_rows if fold(r["metadata"]["question"]) in texts)
    fkeys = {sel[q["id"]]["qid"] for q in fq if q["id"] in sel}  # khoá nội bộ của câu FINAL (nhãn '### id:' trong request của solver)
    in_req = 0
    for role in ("solve", "repair", "think"):
        for p in (WORK / "api" / role).glob("*/*.json"):
            u = json.loads(p.read_text(encoding="utf-8"))["messages"][-1]["content"]
            in_req += len(set(re.findall(r"### id: (\S+)\n", u)) & fkeys)
    g["leakage"] = {"final_questions": len(fq), "final_shops_in_sft": len(f_shops & {r["business_id"] for r in all_rows}),
                    "final_lineages_in_sft": len(f_lin & {r["metadata"]["parent_example_id"] for r in all_rows}),
                    "final_question_ids_in_sft": len({q["id"] for q in fq} & {r["metadata"]["question_id"] for r in all_rows}),
                    "final_items_in_solver_requests": in_req,
                    "train_dev_lineage_overlap": len(set(man["lineage"]["train"]) & set(man["lineage"]["dev"]))}
    # rò rỉ holdout BTC: câu của holdout600 / val300 / smoke6 / shop holdout không được nằm trong prompt hay completion của dòng nào
    # (ví dụ few-shot của p1 đến từ bank BTC). Chỉ đếm, không in câu BTC.
    held = set()
    for f in ("holdout600.txt", "val300.txt", "smoke6.txt"):
        held |= {x.split()[0] for x in (a.splits / f).read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
    hshops = {x.split()[0] for x in (a.splits / "holdout_shops.txt").read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
    btcq = jl(a.btc_train / "questions.jsonl")
    held_txt = {"Q: " + q["question"] + "\nTYPE:" for q in btcq if q["id"] in held or q["business_id"] in hshops}
    rows_with_examples = sum("EXAMPLES FROM OTHER SHOPS" in r["prompt_messages"][-1]["content"] for r in all_rows)
    hits = sum(any(t in r["prompt_messages"][-1]["content"] for t in held_txt) for r in all_rows if "EXAMPLES FROM OTHER SHOPS" in r["prompt_messages"][-1]["content"])
    held_q = {" ".join(q["question"].split()) for q in btcq if q["id"] in held or q["business_id"] in hshops}
    g["btc_holdout_leakage"] = {"held_out_questions": len(held_txt), "rows_with_examples": rows_with_examples, "rows_showing_a_held_out_example": hits,
                                "sft_questions_equal_to_held_out_text": sum(" ".join(r["metadata"]["question"].split()) in held_q for r in all_rows)}
    # không phải rò rỉ (khác shop, khác dữ liệu, khác đáp án) nhưng báo: câu FINAL có câu chữ trùng một câu train/dev (sau fold)
    g["final_text_twins"] = {"final_questions_with_identical_text_in_sft": len(texts & {fold(r["metadata"]["question"]) for r in all_rows}),
                             "sft_rows_with_final_text": in_rows}
    # audit vòng 2 trên các dòng còn trong SFT
    # id câu được cấp lại khi đóng gói lại sau vòng 2: khớp mục audit với dòng SFT theo (shop, câu hỏi, completion)
    kept = {(r["business_id"], r["metadata"]["question"], r["completion"]) for r in all_rows}
    key = lambda x: (x["business_id"], x["question"], x["completion"])  # noqa: E731
    samp = {x["row"]: x for x in jl(WORK / "audit" / "sample.jsonl")}
    man_v = jl(WORK / "audit" / "manual_verdicts.jsonl")
    in_sft = [v for v in man_v if key(samp[v["row"]]) in kept]
    ok = sum(v["pass"] for v in in_sft)
    llm = [v for v in jl(WORK / "audit" / "llm_verdicts.jsonl")]
    llm_s = [v for v in llm if not v.get("bait") and v["row"] in samp and key(samp[v["row"]]) in kept]
    baits = [v for v in llm if v.get("bait")]
    caught = sum(1 for v in baits if v["pass"] is False)
    llm_ok = sum(1 for v in llm_s if v["pass"])
    if a.audit_dir:
        aq = json.loads((a.audit_dir / "summary.json").read_text(encoding="utf-8"))
        al_ = aq["all"]
        in_sft = [None] * al_["n"]
        ok = al_["pass"]
    g["audit"] = {
        **({"llm_independent_subagent_q35": {k: aq[k] for k in ("auditor", "sample", "judged", "all", "v1_1", "p1", "thinking")}} if a.audit_dir else {}),
        "llm_independent_subagent": {"who": "Claude subagent (độc lập, rubric AUDIT_RUBRIC.md) — KHÔNG phải người", "sample_round2": len(man_v),
                                     "in_sft": len(in_sft), "pass": ok, "rate": round(ok / max(1, len(in_sft)), 4),
                                     "wilson95": [round(x, 4) for x in wilson(ok, len(in_sft))] if in_sft else None,
                                     "all_sample_pass": sum(v["pass"] for v in man_v), "all_sample_rate": round(sum(v["pass"] for v in man_v) / max(1, len(man_v)), 4)},
        "llm_cross_family": {"auditor": sorted({v["auditor"] for v in llm}), "baits": len(baits), "baits_caught": caught,
                             "sensitivity": round(caught / max(1, len(baits)), 4), "in_sft": len(llm_s), "pass": llm_ok,
                             "pass_rate": round(llm_ok / max(1, len(llm_s)), 4), "pass_wilson95": [round(x, 4) for x in wilson(llm_ok, len(llm_s))] if llm_s else None}}
    if a.tokens and a.tokens.exists():
        tv = json.loads(a.tokens.read_text(encoding="utf-8"))
        ck = tv["checks"]
        real = ck.get("tokenizer_alignment_real", {})
        g["tokens"] = {"tokenizer_alignment_real": real.get("status"), "revision": real.get("revision"), "rows": real.get("rows"),
                       "eos_once_rows": real.get("eos_once_rows"), "rejected_long": len(real.get("rejected_long") or []), "errors": len(real.get("errors") or []),
                       "length_quantiles": real.get("length_quantiles"), "max_len": real.get("max_len"), "supervised_tokens": real.get("supervised_tokens"),
                       "input_tokens": real.get("input_tokens"), "padding_probe": (real.get("padding_probe") or {}).get("status"),
                       "dataset_contract": ck.get("dataset_contract", {}).get("status"), "mask_logic_fake_tokenizer": ck.get("mask_logic_fake_tokenizer", {}).get("status"),
                       "config": ck.get("config", {}).get("status"), "local_base_config": ck.get("local_base_config", {}).get("status"),
                       "ready_for_real_training": tv.get("ready_for_real_training"), "blockers": tv.get("blockers_for_real_training"),
                       "gpu_checks": {k: ck.get(k, {}).get("status") for k in ("gpu_bf16", "forward_backward_one_batch", "trainable_modules_dtypes",
                                                                                 "save_load_adapter", "resume_from_checkpoint", "throughput_memory")}}
    al = json.loads(a.align.read_text(encoding="utf-8")).get("summary") if a.align.exists() else None
    g["btc_align_snapshot"] = {k: al.get(k) for k in ("ok", "applicable", "match_rate", "not_applicable")} if al else None
    tr = g["splits"]["train"]
    gates = {
        "alignment": (al or {}).get("match_rate", 0) >= 0.95,
        "audit_llm_independent": (g["audit"]["llm_independent_subagent_q35"]["all"]["rate"] >= 0.95 and g["audit"]["llm_independent_subagent_q35"]["all"]["n"] >= 150
                                  and g["audit"]["llm_independent_subagent_q35"]["all"]["wilson95"][0] >= 0.90) if a.audit_dir else
                                 (g["audit"]["llm_independent_subagent"]["rate"] >= 0.95 and len(in_sft) >= 150 and (g["audit"]["llm_independent_subagent"]["wilson95"] or [0])[0] >= 0.90),
        "audit_cross_family": g["audit"]["llm_cross_family"]["sensitivity"] >= 0.80,
        "distribution": tr["industry_core_row_share"] >= 0.90 and 0.55 <= tr["period"].get("thang_cu_the", 0) <= 0.65 and tr["L1"].get("skill", 9) <= 0.15 and tr["L1"].get("type", 9) <= 0.10,
        "leakage": not any(v for k, v in g["leakage"].items() if k != "final_questions"),
        "btc_holdout_leakage": g["btc_holdout_leakage"]["rows_showing_a_held_out_example"] == 0 and g["btc_holdout_leakage"]["sft_questions_equal_to_held_out_text"] == 0,
        "mutation": tr["mutation"]["share_discriminating_answerable"] >= 0.95,
        "dup": tr["exact_dup_question_share"] <= 0.02, "templates": tr["template_share"] <= 0.15,
        "count": tr["rows"] >= 6000 and g["splits"]["dev"]["rows"] >= 500,
        "tokenizer": (g.get("tokens") or {}).get("tokenizer_alignment_real") == "passed" and not (g.get("tokens") or {}).get("rejected_long"),
    }
    g["gates"] = gates
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gates.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"gates": gates, "counts": {k: v for k, v in g["counts"].items() if k != "manifest_counts"},
                      "train": {k: tr[k] for k in ("rows", "questions", "industry_core_row_share", "L1", "template_share", "exact_dup_question_share", "mutation")},
                      "leakage": g["leakage"], "audit": g["audit"], "tokens": g.get("tokens", {}).get("tokenizer_alignment_real")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
