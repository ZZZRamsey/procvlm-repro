#!/usr/bin/env bash
# ProcVQA aligned-vLLM evaluation; external locations are explicit inputs.
# Required env: PROCVLM_CKPT, PROCVLM_LOG_DIR, PROCVLM_DATASET, PROCVLM_BS(optional, default 4)
set -euo pipefail


REPRO_ROOT="${PROCVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${PROCVLM_EVAL_ENV:?Set PROCVLM_EVAL_ENV}"

PROCVLM_CKPT="${PROCVLM_CKPT:?set PROCVLM_CKPT}"
export PROCVLM_CKPT
PROCVLM_LOG_DIR="${PROCVLM_LOG_DIR:?set PROCVLM_LOG_DIR}"
export PROCVLM_LOG_DIR
PROCVLM_DATASET="${PROCVLM_DATASET:?set PROCVLM_DATASET}"
export PROCVLM_DATASET
BS="${PROCVLM_BS:-4}"
export PROCVLM_BS; export PROCVLM_ROOT=$REPRO_ROOT

export EVQA_DATASET_ROOT="${EVQA_DATASET_ROOT:?}"
export PROCVLM_MEDIA_INDEX="${PROCVLM_MEDIA_INDEX:-$EVQA_DATASET_ROOT/media_index}"
export PYTHONPATH="${REPRO_ROOT}:${PYTHONPATH:-}"
export AWS_SHARED_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:?}"
export AWS_CONFIG_FILE="${AWS_CONFIG_FILE:-$HOME/.aws/config}"
export AWS_PROFILE="${AWS_PROFILE:-default}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:?}"
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:?}"
: "${PROCVLM_S3_BUCKET:?}" "${PROCVLM_S3_PREFIX:?}"
export AWS_EC2_METADATA_DISABLED=true

mkdir -p "$PROCVLM_LOG_DIR"
echo "=== ProcVQA eval === ckpt=$PROCVLM_CKPT dataset=$PROCVLM_DATASET bs=$BS log=$PROCVLM_LOG_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$REPRO_ROOT"
EXTRA_ARGS=()
if [[ "${PROCVLM_RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume --fallback_batch_size "${PROCVLM_FALLBACK_BS:-32}")
  if [[ -n "${PROCVLM_LEGACY_LOG:-}" ]]; then
    EXTRA_ARGS+=(--import_legacy_log "$PROCVLM_LEGACY_LOG")
  fi
fi
if [[ -n "${PROCVLM_TASK_TYPE:-}" ]]; then
  EXTRA_ARGS+=(--task_type "$PROCVLM_TASK_TYPE")
fi
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/procvlm-triton-${USER:-job}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/procvlm-inductor-${USER:-job}"
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/procvlm-vllm-cache-${USER:-job}"
"${ENV_PREFIX}/bin/python" evqa/eval/run_eval.py \
    --model_path "$PROCVLM_CKPT" \
    --model_type procvlm \
    --dataset_use "$PROCVLM_DATASET" \
    --batch_size "$BS" \
    --tp "${PROCVLM_TP:-1}" \
    --log_dir "$PROCVLM_LOG_DIR" "${EXTRA_ARGS[@]}"

echo '=== EVAL DONE ==='
