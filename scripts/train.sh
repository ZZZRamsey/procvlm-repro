#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROCVLM_ROOT="$ROOT/training"
export PROCVLM_NNODES="${PROCVLM_NNODES:-4}" PROCVLM_NPROC="${PROCVLM_NPROC:-4}"
export PROCVLM_GRAD_ACCUM="${PROCVLM_GRAD_ACCUM:-32}" PROCVLM_SAVE_TOTAL_LIMIT="${PROCVLM_SAVE_TOTAL_LIMIT:-1}"
export PYTHONPATH="$PROCVLM_ROOT"
exec bash "$PROCVLM_ROOT/scripts/procvlm_train_stage.sh" "${1:?Usage: bash scripts/train.sh 1|2}"
