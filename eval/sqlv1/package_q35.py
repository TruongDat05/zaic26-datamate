"""Đóng gói SFT cho adapter Qwen3.5-9B: build/qwen35_sft_v1/ (mixture_manifest.json, HASHES.json, README.md). Không train.

    python eval/sqlv1/package_q35.py [--pkg <controller>/build/qwen35_sft_v1]

Đọc: sft/{train,dev}.jsonl + sft_manifest.json (exporter), token_validation_qwen3_5_9b.json (train_lora.py --validate-only, tokenizer thật),
parity_report.json (scripts/check_sft_parity.py), configs/train_qwen3_5_9b_bf16.toml (bản pin chép vào gói).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

FOXEL = controller_root()
SYNTH = Path(__file__).resolve().parents[2]


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def git(repo: Path, *a) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()


def source_of(r) -> str:
    m = r["metadata"]
    if r["thinking_mode"]:
        return f"thinking_trace:{m['solver']['model_key']}:{m['solver']['model']}"
    if m["derivation"] == "offline_program_template":
        return "program_template"
    return f"api_solution:{m['solver']['model_key']}:{m['solver']['model']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, default=FOXEL / "build" / "qwen35_sft_v1")
    a = ap.parse_args()
    pkg, sft = a.pkg, a.pkg / "sft"
    man = json.loads((sft / "sft_manifest.json").read_text(encoding="utf-8"))
    rows = {s: [json.loads(x) for x in (sft / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] for s in ("train", "dev")}
    tv = json.loads((pkg / "token_validation_qwen3_5_9b.json").read_text(encoding="utf-8"))
    real = tv["checks"]["tokenizer_alignment_real"]
    par = json.loads((pkg / "parity_report.json").read_text(encoding="utf-8"))
    mix = {"kind": "datamate_sft_mixture", "build_id": man["build_id"], "base": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
           "legacy_rows": 0, "note": ("one source: syn_sales_sql_v1 (synthetic shops, oracle gold), every row verified by build_sft/2; "
                                      "no legacy sql_rft_v3 rows (not on this machine), no FINAL rows, no competition questions as rows "
                                      "(competition examples appear only inside p1 prompts, filtered bank, no loss)"),
           "splits": {}}
    for s, rs in rows.items():
        n = len(rs)
        by = collections.Counter(source_of(r) for r in rs)
        var = collections.Counter(f'{(r["metadata"].get("offline") or {}).get("prompt_variant", "p0")}{"+thinking" if r["thinking_mode"] else ""}' for r in rs)
        mix["splits"][s] = {"rows": n, "questions": len({r["metadata"]["question_id"] for r in rs}), "shops": len({r["business_id"] for r in rs}),
                            "by_source": {k: {"rows": v, "share": round(v / n, 4)} for k, v in by.most_common()},
                            "by_prompt_variant": {k: {"rows": v, "share": round(v / n, 4)} for k, v in var.most_common()},
                            "thinking_rows": sum(r["thinking_mode"] for r in rs),
                            "answer_type": dict(collections.Counter(r["metadata"]["answer_type"] for r in rs).most_common())}
    mix["tokens"] = {k: real.get(k) for k in ("supervised_tokens", "input_tokens", "length_quantiles", "max_len")}
    mix["recommendation"] = {
        "epochs": 2, "lr": 1e-4, "effective_batch": 16, "lora": "r16 / alpha32 / dropout 0, attention + Gated DeltaNet + MLP of the text model",
        "checkpoint_choice": "by scripts/run_holdout.sh on holdout600 against sql_rft_v3 (dev NLL only tracks training)",
        "why": ("~7.9k rows over ~4.7k questions, short completions (mean 165 tokens): 2 passes; p0/p1 about 50/50 as the runtime samples plain "
                "questions; 220 thinking rows (2.8% of rows, 16.5% of supervised tokens, mean 1,143 tokens) keep the adapter usable under "
                "--lora-thinking without letting traces dominate the loss"),
        "thinking_supervised_tokens": {"rows": 220, "tokens": 251428, "share": 0.165}, "plain_supervised_tokens": {"rows": 7711, "tokens": 1271095},
        "receiver_options": ["init from adapters/sql_rft_v3 (config [lora] init_adapter, regex of its adapter_config) vs a new adapter: train both, keep the better on holdout",
                             "1 vs 2 epochs: evaluate the epoch-1 checkpoint too"]}
    (pkg / "mixture_manifest.json").write_text(json.dumps(mix, ensure_ascii=False, indent=1), encoding="utf-8")
    cfg_src = FOXEL / "configs" / "train_qwen3_5_9b_bf16.toml"
    cfg = pkg / "train_qwen3_5_9b_bf16.pinned.toml"
    shutil.copy2(cfg_src, cfg)
    snap = json.loads(next((FOXEL / "build" / "sft_v2").glob("snapshot_988c65c.json")).read_text(encoding="utf-8"))
    tok = FOXEL / "build" / "sft_v2" / "qwen3_5_9b_tok"
    hashes = {"build_id": man["build_id"], "data_hash": man["inputs"]["data_hash"], "source_manifest_hash": man["inputs"]["source_manifest_hash"],
              "snapshot": {"commit": snap["commit"], "source_hash": snap["source_hash"], "loader_closure_hash": snap["loader_closure_hash"]},
              "files": {p.relative_to(pkg).as_posix(): sha(p) for p in sorted([*pkg.glob("*"), *pkg.glob("sft/*"), pkg / "offline_bundle" / "bundle.json",
                                                                                  *pkg.glob("trainer_validate_pre/*")])
                        if p.is_file() and p.name != "HASHES.json" and p.suffix in (".jsonl", ".json", ".toml", ".md")},
              "tokenizer": {"revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                            "files": {p.name: sha(p) for p in sorted(tok.iterdir()) if p.is_file()}},
              "parity": {"sample": par["sample"], "passed": par["passed"], "status": par["status"]},
              "repos": {"upstream": {"branch": git(FOXEL, "rev-parse", "--abbrev-ref", "HEAD"), "head": git(FOXEL, "rev-parse", "HEAD")},
                        "synth": {"branch": git(SYNTH, "rev-parse", "--abbrev-ref", "HEAD"), "head": git(SYNTH, "rev-parse", "HEAD")}}}
    (pkg / "HASHES.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"mixture": {s: {k: v for k, v in d.items() if k in ("rows", "by_prompt_variant", "thinking_rows")} for s, d in mix["splits"].items()},
                      "files": len(hashes["files"])}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
