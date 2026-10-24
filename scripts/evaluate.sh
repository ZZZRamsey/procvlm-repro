#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROCVLM_ROOT="$ROOT/evaluation"
export PROCVLM_CKPT="${1:?Usage: bash scripts/evaluate.sh CHECKPOINT [test_full|valid_full] [BATCH=64] [TP=4]}"
export PROCVLM_DATASET="${2:-test_full}" PROCVLM_BS="${3:-64}" PROCVLM_TP="${4:-4}"
export PROCVLM_RESUME=1 PROCVLM_FALLBACK_BS=32
: "${PROCVLM_LOG_DIR:?Use a new evaluation output directory}"
export PYTHONPATH="$PROCVLM_ROOT/runtime_compat:$PROCVLM_ROOT"
exec bash "$PROCVLM_ROOT/scripts/procvlm_eval_entrypoint.sh"
