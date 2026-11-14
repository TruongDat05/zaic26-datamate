#!/usr/bin/env bash
# Continue qwen35_sft_4xl40s on fine_tuning_data_20261114_v2: 1 H100 SXM, 1 full epoch, effective batch 16.
# Install training/requirements_bf16.txt and fla-core as described there; model and init adapter must already be local.
#   bash training/train_h100.sh
#   RESUME=1 bash training/train_h100.sh        # resume only this new run, with unchanged data/settings
# Env: PYTHON, MODEL_DIR, DATA_DIR, INIT_ADAPTER, OUT, TRAIN_CONFIG, TOKENIZER_DIR, REPORT_TO.
# ONLY=data checks the tokenizer/data without GPU or init weights; ONLY=preflight stops after the GPU memory check.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv-train/bin/python}
MODEL=${MODEL_DIR:-models/Qwen3.5-9B}
DATA=${DATA_DIR:-data/fine_tuning_data_20261114_v2}
INIT=${INIT_ADAPTER:-runs/lora/qwen35_sft_4xl40s/adapter}
OUT=${OUT:-runs/lora/qwen35_sft_h100_v2}
CONFIG=${TRAIN_CONFIG:-training/configs/qwen35_sft_h100_v2.json}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
[ "${WORLD_SIZE:-1}" = 1 ] || { echo "Run this script directly on one GPU, without torchrun."; exit 1; }
for f in "$DATA/train.jsonl" "$DATA/dev.jsonl" "$CONFIG"; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
if [ "${ONLY:-}" != data ]; then
  for f in "$MODEL/config.json" "$INIT/adapter_config.json" "$INIT/adapter_model.safetensors"; do
    [ -f "$f" ] || { echo "missing $f (MODEL_DIR / INIT_ADAPTER)"; exit 1; }
  done
  if compgen -G "$OUT/checkpoint-*" >/dev/null && [ "${RESUME:-0}" != 1 ]; then
    echo "$OUT already holds checkpoints: use RESUME=1 for this run, or choose a new OUT."; exit 1
  fi
fi
ARGS=(--config "$CONFIG" --model "$MODEL" --init-adapter "$INIT")
[ -n "${TOKENIZER_DIR:-}" ] && ARGS+=(--tokenizer "$TOKENIZER_DIR")
mkdir -p "$(dirname "$OUT")"

echo "== 1/4 data check"
"$PY" training/train_lora_bf16.py "${ARGS[@]}" --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" --out "${OUT}_data_check" --validate-only
[ "${ONLY:-}" = data ] && exit 0

echo "== 2/4 fla kernels"
"$PY" training/check_fla.py --T 4096 2>&1 | tee "${OUT}_check_fla.log"
grep -q "fla check PASS" "${OUT}_check_fla.log" || { echo "fla check did not pass; see ${OUT}_check_fla.log"; exit 1; }

echo "== 3/4 memory preflight: largest total sequence and largest reply"
if compgen -G "$OUT/checkpoint-*" >/dev/null; then
  echo "skipped: resuming the existing run"
else
  PREFLIGHT_OUT=$(mktemp -d "${OUT}_preflight.XXXXXX")
  "$PY" training/train_lora_bf16.py "${ARGS[@]}" --train "$DATA/train.jsonl" --out "$PREFLIGHT_OUT" \
      --grad-accum 1 --max-steps 2 --longest-first --limit-rows 0 --report-to none
  "$PY" - "$PREFLIGHT_OUT/run_manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
use = m["peak_gpu_reserved_gib"] / m["gpu_total_gib"]
print(f"preflight on {m['gpu']}: {m['peak_gpu_reserved_gib']} GiB reserved / {m['gpu_total_gib']} GiB ({use:.0%})")
if use > 0.95:
    sys.exit("too close to the memory limit: lower ce_chunk before training")
PY
fi
[ "${ONLY:-}" = preflight ] && exit 0

echo "== 4/4 training -> $OUT"
RESUME_ARGS=(); [ "${RESUME:-0}" = 1 ] && RESUME_ARGS=(--resume)
"$PY" training/train_lora_bf16.py "${ARGS[@]}" --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" --out "$OUT" \
    --report-to "${REPORT_TO:-tensorboard}" "${RESUME_ARGS[@]}" 2>&1 | tee -a "$OUT.log"
echo "adapter: $OUT/adapter (best dev-loss checkpoint); checkpoints remain available for RESUME=1"
