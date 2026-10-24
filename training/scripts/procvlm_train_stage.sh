#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${PROCVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV="${PROCVLM_ENV:?Set PROCVLM_ENV to the training environment}"
STAGE="${1:-1}"
export PATH="$ENV/bin:$PATH" PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EVQA_DATASET_ROOT="${EVQA_DATASET_ROOT:?Set EVQA_DATASET_ROOT}"
export PROCVLM_MEDIA_INDEX="${PROCVLM_MEDIA_INDEX:-$EVQA_DATASET_ROOT/media_index}"
export PROCVLM_STRICT_DATA=1
export AWS_SHARED_CREDENTIALS_FILE="${PROCVLM_AWS_CREDENTIALS_FILE:-${AWS_SHARED_CREDENTIALS_FILE:?Set AWS_SHARED_CREDENTIALS_FILE}}"
test -r "$AWS_SHARED_CREDENTIALS_FILE"
export AWS_PROFILE="${AWS_PROFILE:-default}" AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:?}" AWS_EC2_METADATA_DISABLED=true
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:?}"
: "${PROCVLM_S3_BUCKET:?}" "${PROCVLM_S3_PREFIX:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=8 PYTHONUNBUFFERED=1
export NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_IB_AR_THRESHOLD=0
export TMPDIR="${TMPDIR:-/tmp}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PROCVLM_RUN_ROOT:?}/torch_extensions/${HOSTNAME}}"
RUN_ROOT="${PROCVLM_RUN_ROOT:?Set PROCVLM_RUN_ROOT to a unique persistent output directory}"
mkdir -p "$RUN_ROOT"
cd "$ROOT"
if [[ "$STAGE" == 1 ]]; then
  MODEL="${PROCVLM_BASE_MODEL:?Set PROCVLM_BASE_MODEL}"
  LR=1e-5; MLP=1e-5; VISION=5e-6; WARMUP=0.03; VL=0.2; DROP=0; NOISE=0.03
  EPOCHS=1.2; DS=configs/finetune/ds_config_zero2.json
  DATASET="${PROCVLM_STAGE1_DATASET:-train_s1_8k_available%100}"
elif [[ "$STAGE" == 2 ]]; then
  MODEL="$RUN_ROOT/stage1"
  test -s "$MODEL/procvlm_extra/procvlm_head.pt"
  LR=5e-6; MLP=4e-6; VISION=2e-6; WARMUP=0.05; VL=0.3; DROP=0.05; NOISE=0.01
  EPOCHS=4; DS=configs/finetune/ds_config_zero2_offload.json
  DATASET=train_s2_8k%100
else
  echo "Stage must be 1 or 2" >&2; exit 2
fi
export VLM_PATH_QWEN_TRAIN="$MODEL"
NNODES="${PROCVLM_NNODES:-1}"
NPROC="${PROCVLM_NPROC:-8}"
if [[ "$NNODES" == 1 ]]; then
  LAUNCH=(--standalone --nnodes=1 --nproc_per_node="$NPROC")
else
  NODE_RANK="${PROCVLM_NODE_RANK:?Multi-node launch requires an explicit node rank}"
  [[ "$NODE_RANK" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES ))
  LAUNCH=(--nnodes="$NNODES" --nproc_per_node="$NPROC" --node_rank="$NODE_RANK"
    --master_addr="${PROCVLM_MASTER_ADDR:?}" --master_port="${PROCVLM_MASTER_PORT:?}")
fi
echo "$(date -Is) stage=$STAGE nodes=$NNODES local_gpus=$NPROC node_rank=${PROCVLM_NODE_RANK:-0} grad_accum=${PROCVLM_GRAD_ACCUM:-64} base_model=$MODEL"
exec "$ENV/bin/torchrun" "${LAUNCH[@]}" \
  evqa/train/train_qwen.py \
  --model_name_or_path "$MODEL" --tune_mm_llm True --tune_mm_vision True --tune_mm_mlp True \
  --dataset_use "$DATASET" --output_dir "$RUN_ROOT/stage$STAGE" \
  --cache_dir "$RUN_ROOT/hf_cache" --bf16 True --tf32 True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps "${PROCVLM_GRAD_ACCUM:-64}" \
  --learning_rate "$LR" --mm_projector_lr "$MLP" --vision_tower_lr "$VISION" \
  --optim adamw_torch --model_max_length 8192 --data_flatten False --data_packing True \
  --max_pixels 262144 --min_pixels 1024 --video_fps 2 --video_max_frames 512 --video_min_frames 2 \
  --num_train_epochs "$EPOCHS" --max_steps "${PROCVLM_MAX_STEPS:--1}" \
  --warmup_ratio "$WARMUP" --lr_scheduler_type cosine --weight_decay 0.01 \
  --logging_steps 1 --logging_nan_inf_filter False --save_strategy steps \
  --save_steps "${PROCVLM_SAVE_STEPS:-100}" --save_total_limit "${PROCVLM_SAVE_TOTAL_LIMIT:-3}" --report_to none \
  --eval_strategy no --gradient_checkpointing True --ddp_find_unused_parameters False \
  --value_loss_weight "$VL" --value_dropout "$DROP" --value_noise_std "$NOISE" \
  --dataloader_num_workers 4 --dataloader_prefetch_factor 2 --dataloader_persistent_workers True \
  --seed 42 --data_seed 42 --deepspeed "$DS"
