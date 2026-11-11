#!/usr/bin/env bash
# Retrain the three LoRA adapters from the packaged SFT data, in order (each continues from the previous one). See README.md.
#   bash /code/training_code/train.sh            # inside the image (training venv /opt/venv-train, base model /code/models/Qwen3.5-9B)
# One GPU with >= 24 GB. Measured on 1x RTX PRO 4000 Blackwell 24 GB: v1 3.8 h, v2 6.9 h, v3 8.2 h.
# The BF16 step (only when data/sft_bf16 is packaged) needs a bf16 GPU with >= 40 GB (the base stays unquantized).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-/opt/venv-train/bin/python}
BASE=${BASE_MODEL:-/code/models/Qwen3.5-9B}
OUT=${OUT_DIR:-/tmp/lora}
COMMON=(--model "$BASE" --r 16 --alpha 32 --epochs 1 --grad-accum 8 --end '<|im_end|>' --no-thinking)
mkdir -p "$OUT"
# The fla kernels (Gated DeltaNet layers) must match the PyTorch reference before they are trusted for training
$PY src/check_fla.py --T 4096
$PY src/train_lora.py data/sft/sql_rft_v1.jsonl --out "$OUT/sql_rft_v1" --max-len 16384 --lr 1e-4 "${COMMON[@]}"
$PY src/train_lora.py data/sft/sql_rft_v2.jsonl --out "$OUT/sql_rft_v2" --init-adapter "$OUT/sql_rft_v1/adapter" --max-len 20000 --lr 5e-5 "${COMMON[@]}"
$PY src/train_lora.py data/sft/sql_rft_v3_notes.jsonl --out "$OUT/sql_rft_v3" --init-adapter "$OUT/sql_rft_v2/adapter" --max-len 32768 --lr 5e-5 \
    "${COMMON[@]}" --completion-logits
INFER=$OUT/sql_rft_v3/adapter
# SFT LoRA BF16 (when data/sft_bf16 is packaged, i.e. the submitted adapter is the BF16 one of notebooks/05_train_lora.ipynb).
# Every setting comes from configs/sql_bf16.json (the run's train_args.json); a run that continued v3 continues the v3 rebuilt above
if [ -f data/sft_bf16/train.jsonl ]; then
  DEV=(); [ -s data/sft_bf16/dev.jsonl ] && DEV=(--dev data/sft_bf16/dev.jsonl)
  INIT=$($PY -c 'import json; print(json.load(open("configs/sql_bf16.json")).get("init_adapter") or "")')
  case "$INIT" in
    "") INIT_ARGS=() ;;
    sql_rft_v3) INIT_ARGS=(--init-adapter "$OUT/sql_rft_v3/adapter") ;;
    *) echo "configs/sql_bf16.json: init_adapter '$INIT' is not rebuilt by this script (only none or sql_rft_v3)"; exit 1 ;;
  esac
  $PY src/train_lora_bf16.py --config configs/sql_bf16.json --train data/sft_bf16/train.jsonl "${DEV[@]}" --model "$BASE" \
      --out "$OUT/sql_bf16" "${INIT_ARGS[@]}"
  INFER=$OUT/sql_bf16/adapter
fi
echo "adapters in $OUT: $(ls "$OUT" | tr '\n' ' '); the inference adapter is $INFER (packaged one: /code/models/lora)"
