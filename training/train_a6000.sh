#!/usr/bin/env bash
# SFT LoRA BF16 of Qwen3.5-9B on the SFT bundle in data/ (data/data/train.jsonl + dev.jsonl + sft_manifest.json), set for one
# RTX A6000 48 GB (any bf16 GPU with >= 40 GB). Four steps: data check -> fla kernels -> memory preflight -> training.
# Running it again resumes the training from the newest checkpoint in OUT (same data and settings: run_identity.json checks it).
#
#   python3 -m venv .venv-train && .venv-train/bin/pip install -r training/requirements_bf16.txt
#   .venv-train/bin/pip install --no-deps fla-core==0.6.0
#   .venv-train/bin/hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/Qwen3.5-9B
#   bash training/train_a6000.sh
#
# Env: MODEL_DIR (models/Qwen3.5-9B), DATA_DIR (data/data), OUT (runs/lora/qwen35_sft_a6000), PYTHON (.venv-train/bin/python),
#      REPORT_TO (tensorboard | none | wandb), ONLY=data (data check only: tokenizer, no GPU), SKIP_PREFLIGHT=1,
#      INIT_ADAPTER=adapters/sql_rft_v3 (continue that adapter at lr 5e-5 instead of a new one; holdout is then no longer clean),
#      NO_FLA=1 (train without the fla kernels: correct but far slower, only when training/check_fla.py fails),
#      EXTRA_ARGS (appended to the settings below, e.g. a short end-to-end check with its own OUT before the real run:
#        OUT=runs/lora/smoke EXTRA_ARGS="--limit-rows 64 --max-steps 4 --eval-steps 2 --save-steps 2 --dev-max-rows 16" bash training/train_a6000.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv-train/bin/python}
MODEL=${MODEL_DIR:-models/Qwen3.5-9B}
DATA=${DATA_DIR:-data/data}
OUT=${OUT:-runs/lora/qwen35_sft_a6000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # rows of 11k-16k tokens: less fragmentation
export TOKENIZERS_PARALLELISM=false

# The recipe of the bundle (data/config/train_qwen3_5_9b_bf16.toml), on this trainer and fitted to the measured data and 48 GB
ARGS=(
  --max-len 16384                     # every row is <= 15,942 tokens (p50 13.1k): all kept. A longer row is rejected, never cut
  --r 16 --alpha 32 --dropout 0       # rank / alpha / 248 modules of adapters/sql_rft_v3: vLLM serves it as it serves v3
  --lr 1e-4 --scheduler cosine --warmup-ratio 0.03 --weight-decay 0 --max-grad-norm 1.0
  --epochs 2 --micro-batch 1 --grad-accum 16   # 16 rows per optimizer step: ~398 steps per epoch on 6,363 rows, ~796 in all
  --eval-steps 50 --save-steps 50 --save-total-limit 3 --early-stop 3   # adapter/ = the checkpoint with the lowest dev loss
  --dev-max-rows 320                  # dev loss on a fixed 320 of the 1,064 dev rows: ~8% of the time between evaluations, not ~30%
  --ce-chunk 1024 --attn sdpa --seed 3407
)
[ -n "${INIT_ADAPTER:-}" ] && ARGS+=(--init-adapter "$INIT_ADAPTER" --lr 5e-5)   # same r / alpha / modules required (checked)
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)   # word-split on purpose: a list of flags; later flags win
FILES=(--train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" --model "$MODEL")
for f in "$DATA/train.jsonl" "$DATA/dev.jsonl" "$MODEL/config.json"; do
  [ -f "$f" ] || { echo "missing $f (DATA_DIR / MODEL_DIR)"; exit 1; }
done
mkdir -p "$(dirname "$OUT")"

echo "== 1/4 data check: format, chat template, masks, lengths, sft_manifest.json (tokenizer only)"
"$PY" training/train_lora_bf16.py "${FILES[@]}" "${ARGS[@]}" --out "${OUT}_data_check" --validate-only
[ "${ONLY:-}" = data ] && exit 0

echo "== 2/4 fla kernels of the 24 Gated DeltaNet layers"
if [ "${NO_FLA:-0}" = 1 ]; then
  export PYTHONPATH="$PWD/training/no_fla${PYTHONPATH:+:$PYTHONPATH}"
  echo "NO_FLA=1: training on the PyTorch reference path"
else
  "$PY" training/check_fla.py --T 4096 2>&1 | tee "${OUT}_check_fla.log" || true
  grep -q "fla check PASS" "${OUT}_check_fla.log" || {
    echo "fla check did not pass (pip install --no-deps fla-core==0.6.0 into the training venv); NO_FLA=1 trains without it, far slower"; exit 1; }
fi

echo "== 3/4 memory preflight: 2 optimizer steps of 1 row on the longest rows"
if compgen -G "$OUT/checkpoint-*" >/dev/null; then
  echo "skipped: $OUT already holds checkpoints (resuming)"
elif [ "${SKIP_PREFLIGHT:-0}" = 1 ]; then
  echo "skipped: SKIP_PREFLIGHT=1"
else
  rm -rf "${OUT}_preflight"
  # the 2 longest rows of the whole data (--limit-rows 0 undoes an EXTRA_ARGS limit): here thinking rows whose replies also fill the
  # LM head's --ce-chunk, so both peaks are measured
  "$PY" training/train_lora_bf16.py --train "$DATA/train.jsonl" --model "$MODEL" "${ARGS[@]}" --grad-accum 1 --max-steps 2 --longest-first \
      --limit-rows 0 --out "${OUT}_preflight" --report-to none
  "$PY" - "${OUT}_preflight/run_manifest.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
use = m["peak_gpu_reserved_gib"] / m["gpu_total_gib"]
print(f"preflight on {m['gpu']}: {m['peak_gpu_gib']} GiB allocated, {m['peak_gpu_reserved_gib']} GiB reserved of {m['gpu_total_gib']} GiB ({use:.0%}); "
      f"{m['seconds'] / 2:.0f} s per longest row (first step includes kernel autotuning)")
if use > 0.95:
    sys.exit("too close to the memory limit: lower --ce-chunk, or --max-len (rows above it are dropped)")
EOF
fi

echo "== 4/4 training -> $OUT (log: $OUT.log)"
"$PY" training/train_lora_bf16.py "${FILES[@]}" "${ARGS[@]}" --out "$OUT" --report-to "${REPORT_TO:-tensorboard}" --resume 2>&1 | tee -a "$OUT.log"
echo "adapter: $OUT/adapter (best dev-loss checkpoint); settings: $OUT/train_args.json; summary: $OUT/run_manifest.json"
