# ProcVLM reproduction record

**Snapshot date:** 2026-10-24 (Asia/Shanghai). Historical experiments and this release's validation are separate evidence sets. Exact historical filesystem locations remain in the owner's private operations record; public commands use configurable paths.

## Summary

- Stage 1 finished at **6,284 steps**, actual epoch **1.2001886**; Stage 2 finished at **2,724 steps / 4 epochs** on 2026-10-09 (Asia/Shanghai).
- Formal results use the **2026-10-15 aligned original-vLLM protocol**: 8 checkpoints, ID/OOD, 16 complete evaluations, 292,408 predictions and 112 metric values.
- This release preserves separate archival training and evaluation source trees. `results/source_manifest.json` records original/release hashes and intentional portability differences.
- No checkpoints, optimizer states, datasets, credentials or private services are distributed. There is no public download of the reproduced weights in this release.

## Source and path map

| Role | Configurable location / release path |
|---|---|
| Repository | `$REPO` (root of this clone) |
| Training implementation | `$REPO/training` |
| Evaluation implementation | `$REPO/evaluation` |
| Training Python environment | `$PROCVLM_ENV` |
| Evaluation Python environment | `$PROCVLM_EVAL_ENV` |
| Annotation/data root | `$EVQA_DATASET_ROOT` |
| Read-only media index | `$PROCVLM_MEDIA_INDEX` |
| Base checkpoint | `$PROCVLM_BASE_MODEL` (Qwen3-VL-2B-Instruct) |
| Completed S1 inference model | `$STAGE1_MODEL` |
| Completed S2 inference model | `$STAGE2_MODEL` |
| New training outputs | `$PROCVLM_RUN_ROOT/stage1` or `stage2` |
| New evaluation outputs | `$PROCVLM_LOG_DIR` |
| Standalone result page | `$REPO/visualization/index.html` |

Training source: `procvlm-paper-stage2-from-s1-6284-4x4-20261006-132128/source`; shared training core was cross-checked against the final Stage 1 resume snapshot `procvlm-paper-resume-s1-4700to6284-resilient-4x4-20261004-204749/source`. Evaluation source: `procvlm_eval_aligned_20261015/code`. These names identify archives, not directories required in a new clone.

The development repository HEAD was `5c4875d6b0acf38efc07d0029c7abf36bbf3f610`, but contained local modifications. HEAD alone does not identify the experiment implementation; use the per-file manifest.

## Environment

Training and evaluation must run in **different processes/environments** with only their own source root on `PYTHONPATH`.

| Component | Training | Aligned evaluation |
|---|---|---|
| Python | 3.10 | 3.10.12 |
| PyTorch | 2.8.0 | 2.10.0 |
| torchvision | 0.23.0 | 0.25.0 |
| Transformers | 4.57.3 | 4.57.3 |
| FlashAttention | 2.8.3 | not required for the vLLM chain |
| DeepSpeed | 0.18.9 | not required for evaluation |
| vLLM | not required | 0.18.1 |
| Hardware used | 4 nodes × 4 A800 80 GB | TP=4 |

The train environment historically used CUDA 12.8 wheels. Select wheels/builds compatible with the GPU driver; metadata version alone does not establish binary equivalence. Requirements files are **observed direct dependency pins**, not a verified portable lock or a claim that fresh installation was tested.

Illustrative environment setup (outside the repository):

```bash
python3.10 -m venv /absolute/path/to/train-env
/absolute/path/to/train-env/bin/pip install -r requirements/train.txt
# Build/install the matching FlashAttention wheel after torch is installed:
/absolute/path/to/train-env/bin/pip install flash-attn==2.8.3 --no-build-isolation

python3.10 -m venv /absolute/path/to/eval-env
/absolute/path/to/eval-env/bin/pip install -r requirements/eval.txt
```

Large CUDA wheels and compiler/toolkit setup are external prerequisites. The actual release checks reuse the existing environments without modifying them; a clean dependency-resolver install is not part of those checks. `evaluation/runtime_compat/sitecustomize.py` preserves a historical process-local compatibility shim reporting huggingface-hub as 0.34.0 to version checks (installed metadata: 0.36.0). It does not change installed files. The local `jsonlines`/`imagehash` shims are also part of the archived evaluation runtime; do not accidentally omit this path.

## Data

Official sources: [ProcVQA-20M annotations](https://huggingface.co/datasets/ce-amtic/ProcVQA-20M-annotations), [ProcVLM repository](https://github.com/RUCKBReasoning/ProcVLM). The annotation revision used here is `54591c4bad106f617b44140b2b91c7618b03990b`.

Required external layout:

```text
$EVQA_DATASET_ROOT/
  jsons/stage1/trun8192_pack.json
  jsons/stage2/trun8192_pack.json
  jsons/test/full.jsonl
  jsons/valid/full.jsonl
  missing_audit/allowed_group_ids.npy
  media_index/{paths.npy,hashes.npy,locations.npy}
  filelist.txt
  READY.json
```

The media reader fetches JPEG/PNG tar members via **S3 Range requests into RAM**. Media shards, media indexes and `filelist.txt` must correspond exactly. Point `PROCVLM_S3_BUCKET` and `PROCVLM_S3_PREFIX` at your authorized mirror; keys are `<prefix>/shards/shard-NNNNNN.tar`. The index arrays are external preparation artifacts, not generated or downloaded by this source release. A public annotation download alone is insufficient to run the S3 reader. Put AWS credentials/config in external files; never copy their content into this repository.

Training may build `.mmap_idx.npy` and `.mmap_meta.json` beside annotations on first use. Use a writable preparation copy, or precompute/verify these caches before mounting annotations read-only. Smoke tests use an isolated data directory and never rebuild indexes beside the archival annotations.

### Data coverage

| Item | Count |
|---|---:|
| Original Stage 1 packed groups | 2,681,858 |
| Excluded groups with unavailable media | 1,508 (0.05623%) |
| Retained Stage 1 groups | 2,680,350 |
| QA in excluded groups | 6,687 |
| Image references in excluded groups | 51,827 |
| Missing reference occurrences | 22,032 |
| Unique missing media paths | 9,238 |
| Stage 2 groups (all retained) | 348,657 |
| ID `test_full` QA | 18,973 |
| OOD `valid_full` QA | 17,578 |

The authorized S1 group filter is `missing_audit/allowed_group_ids.npy`; `train_s1_8k_available%100` uses it. Selecting `train_s1_8k%100` instead changes the experiment and may fail on missing media. The excluded groups count is not the number of excluded QA.

## Training configuration

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Dataset | `train_s1_8k_available%100` | `train_s2_8k%100` |
| Initial model | Qwen3-VL-2B-Instruct | completed Stage 1 |
| Epoch target | 1.2 | 4 |
| Historical terminal steps | 6,284 | 2,724 |
| LLM / projector / vision LR | 1e-5 / 1e-5 / 5e-6 | 5e-6 / 4e-6 / 2e-6 |
| Warmup ratio | 0.03 | 0.05 |
| Value loss / dropout / noise | 0.2 / 0 / 0.03 | 0.3 / 0.05 / 0.01 |
| DeepSpeed | ZeRO-2 | ZeRO-2 + CPU optimizer offload |

Common: full finetuning, 16 GPUs, per-device batch 1, accumulation **32**, global batch **512**, context 8192, bf16/tf32, FlashAttention 2, AdamW, cosine LR, weight decay 0.01, seed/data seed 42, gradient checkpointing, pixel area min/max **1024 / 262144**, video 2 FPS and 2–512 frames. Save every 100 optimizer steps; actual experiment wrapper retention **1** (the archived generic launcher defaults to 3).

`16 × 1 × 32 = 512`. Using accumulation 64 with 16 GPUs doubles global batch to 1024; 8 GPUs × accumulation 64 gives 512 but does not establish optimizer-shard resume compatibility. This release does not promise cross-topology DeepSpeed resume.

### Stage 1

Load `.env.local` as in README. Allocate four nodes with four GPUs each, then on **each node**:

```bash
# Use your scheduler's rank/master; rank differs for each node.
export PROCVLM_NODE_RANK="$NODE_RANK"
export PROCVLM_MASTER_ADDR="$MASTER_ADDR"
export PROCVLM_MASTER_PORT=29702
export PROCVLM_MAX_STEPS=6284
bash scripts/train.sh 1
```

`PROCVLM_RUN_ROOT` must be a new, shared, writable directory. The wrapper sets the historical 4×4/accumulation32/retention1 defaults. No scheduler submission command containing private infrastructure is embedded here. Do not run a multi-node command on an unallocated login host.

### Stage 2

On a **new Stage 2 run directory**, point `stage1` at the completed Stage 1 *inference* directory, not a partially written checkpoint:

```bash
export PROCVLM_RUN_ROOT=/absolute/path/to/NEW-stage2-run
mkdir -p "$PROCVLM_RUN_ROOT"
# Refuses to replace an existing path; verify STAGE1_MODEL beforehand.
ln -s "$STAGE1_MODEL" "$PROCVLM_RUN_ROOT/stage1"
export PROCVLM_MAX_STEPS=2724
# Set the same node rank/master variables on every allocated node.
bash scripts/train.sh 2
```

Stage 2 only reads that Stage 1 link. Its outputs go into `$PROCVLM_RUN_ROOT/stage2`.

### Resume safely

For an incomplete run, choose a complete checkpoint **outside the new output directory**. Preserve the original global scheduler horizon and topology:

```bash
export PROCVLM_RUN_ROOT=/absolute/path/to/NEW-stage1-resume
export PROCVLM_RESUME_CHECKPOINT=/absolute/path/to/old-stage1/checkpoint-N
export PROCVLM_MAX_STEPS=6284
export PROCVLM_RESUME_TARGET_STEPS=6284
bash scripts/train.sh 1
```

Use a checkpoint with `.complete`, `trainer_state.json`, model/value-head files, scheduler/optimizer state and all 16 RNG/shard files. The archived selector chooses the latest complete checkpoint already in the output if present, otherwise the explicit input. It rejects an explicit input nested under the output being rotated. A final inference-only model is **not** an optimizer-preserving resume point.

For Stage 2, use its original horizon 2724 and stage `2`, and prepare the Stage 1 input link. A final checkpoint already at its target is complete; do not set a resume target equal to the saved step and call it continued training.

**Historical interruptions:** S1 initially stopped at an explicit 3600-step cap (~0.688 epoch), not because 1,508 groups were filtered. Disk exhaustion later interrupted near step 4751; the latest complete point was 4700, then resumed to 6284. The scheduler repair updates restored LR for the intended horizon without deleting moments/counters. It does not retroactively erase the early zero-LR endpoint or guarantee bitwise equivalence to uninterrupted full training.

## Checkpoint mapping

| Result ID | Inference checkpoint identity | Resume point |
|---|---|---|
| `official` | upstream ProcVLM-2B | not this experiment |
| `full_s1_6284` | final S1 of the 2026-10-04 resilient run | `checkpoint-6284` |
| `full_s2_6284_2724` | final S2 of the 2026-10-06 run initialized from S1-6284 | `checkpoint-2724` |
| `full_s1_3600` | historical capped S1 | historical only |
| `full_s2_3600_2724` | historical S2 initialized from capped S1-3600 | historical only |
| `subset5_s2`, `subset10_s2`, `subset20_s2` | historical subset Stage 2 experiments | historical only |

Final inference directories contain `pytorch_model.bin`, `procvlm_extra/procvlm_head.pt` and tokenizer/processor/config files. The two final resume directories were historically verified with `.complete`, scheduler, 16 optimizer shards and 16 RNG files; this statement is not permission to delete them and not a guarantee of future availability. Distinguish checkpoints by their full run identity, not just `2724` (shared by multiple historical runs).

## Evaluation

Original chain: `evqa/eval/run_eval.py` → `evqa.model.batch_chat_with_vllm` → `core.models.qwenvl`. Formal protocol disables the value-head branch, preserves frame/text/image interleaving, temperature 0, max new tokens 8192, max model length 32768, TP4, outer batch64 (OOM fallback32), one preprocessing worker, pipeline chunk8, GPU memory utilization0.70. Each checkpoint loads its **own** processor.

On one allocated four-GPU host:

```bash
export PROCVLM_LOG_DIR=/absolute/path/to/NEW-id-eval
bash scripts/evaluate.sh "$STAGE2_MODEL" test_full 64 4
export PROCVLM_LOG_DIR=/absolute/path/to/NEW-ood-eval
bash scripts/evaluate.sh "$STAGE2_MODEL" valid_full 64 4
```

Outputs:

```text
$PROCVLM_LOG_DIR/metrics.resume.json
$PROCVLM_LOG_DIR/<dataset>.predictions.jsonl
$PROCVLM_LOG_DIR/<dataset>.predictions.progress.json
$PROCVLM_LOG_DIR/results_procvlm_<checkpoint>_vllm.log
```

The JSONL includes immutable metadata, ordered sample digests and predictions. Reusing an output directory resumes only compatible predictions. Never import older value-head/processor protocols into these aligned results. Optional `PROCVLM_TASK_TYPE=segmentation|progress|planning` filters an evaluation; it is not a full-split result.

### Aligned results

All numbers below are rounded to four decimals from `results/aligned_metrics.json`. ID=`test_full` (18,973 QA), OOD=`valid_full` (17,578 QA). **mMAE ↓; the other six ↑.**

#### ID

| Model | BF1@5 ↑ | mMAE ↓ | MCC@95 ↑ | CCC ↑ | EPR@50 ↑ | VOC ↑ | KT ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official ProcVLM-2B | 0.6849 | 0.6042 | 0.8599 | 0.7454 | 5.3807 | 0.8004 | 0.6643 |
| Full S1 · 6284 | 0.6944 | 0.5489 | 0.8713 | 0.7289 | 5.3002 | 0.7630 | 0.6306 |
| Full S2 · S1 6284 → S2 2724 | 0.7201 | 0.5827 | 0.8856 | 0.7880 | 5.6491 | 0.8158 | 0.6911 |
| Historical S1 · 3600 | 0.7056 | 0.5614 | 0.8652 | 0.6889 | 5.2650 | 0.7340 | 0.6034 |
| Historical S2 · S1 3600 | 0.6609 | 0.5852 | 0.8768 | 0.7677 | 5.3972 | 0.8028 | 0.6728 |
| 5% subset S2 | 0.5797 | 0.6375 | 0.8391 | 0.4899 | 4.5232 | 0.5535 | 0.4185 |
| 10% subset S2 | 0.5950 | 0.5985 | 0.7969 | 0.5656 | 4.8268 | 0.6431 | 0.5078 |
| 20% subset S2 | 0.6440 | 0.5485 | 0.8282 | 0.6435 | 5.1033 | 0.7146 | 0.5798 |

#### OOD

| Model | BF1@5 ↑ | mMAE ↓ | MCC@95 ↑ | CCC ↑ | EPR@50 ↑ | VOC ↑ | KT ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official ProcVLM-2B | 0.5921 | 0.5064 | 0.8105 | 0.6574 | 5.5443 | 0.6869 | 0.5534 |
| Full S1 · 6284 | 0.5992 | 0.4569 | 0.8037 | 0.6004 | 5.5827 | 0.6510 | 0.5318 |
| Full S2 · S1 6284 → S2 2724 | 0.5964 | 0.4874 | 0.8140 | 0.6148 | 5.8419 | 0.6389 | 0.5221 |
| Historical S1 · 3600 | 0.5908 | 0.4195 | 0.7951 | 0.6052 | 5.4915 | 0.6675 | 0.5422 |
| Historical S2 · S1 3600 | 0.6248 | 0.5161 | 0.7941 | 0.6182 | 5.5421 | 0.6414 | 0.5218 |
| 5% subset S2 | 0.5255 | 0.3889 | 0.7523 | 0.5269 | 4.7389 | 0.6004 | 0.4526 |
| 10% subset S2 | 0.5071 | 0.3706 | 0.7930 | 0.6019 | 5.0589 | 0.6347 | 0.4843 |
| 20% subset S2 | 0.5426 | 0.3889 | 0.7757 | 0.6382 | 5.2348 | 0.6744 | 0.5350 |

mMAE is calculated on matched boundaries, so interpret it together with BF1. These seven scores are not overall QA accuracy. Planning QA were included in inference (ID 8,112; OOD 7,480), but this scorer has no dedicated planning metric in the formal seven-score set. EPR is not a generic accuracy substitute. Not all metrics exceed the official checkpoint.

## Visualization

`visualization/index.html` is a self-contained static dashboard generated **only** from the aligned 16-row JSON. It supports split selection and includes all seven metrics. No CDN, tracking, credentials or private service dependency.

```bash
python3 scripts/render_metrics.py
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/visualization/` locally, or open the HTML file directly. GitHub renders its source unless separately hosted; this repository does not configure GitHub Pages. The historical video viewer (9 videos, 7 model choices) is separate from this formal 8×2 matrix and is not a live training monitor. Videos/private service addresses are not redistributed.

## Reproduction boundaries

1. Stage 1 filters unavailable-media groups; it is not the exact unfiltered official dataset.
2. Training processor pixel areas 1024/262144 differ from official 65536/16777216. These are areas, not edge lengths. Evaluation also loads checkpoint-specific processors; observed differences cannot be attributed only to weights.
3. Historical interruptions/capped scheduling are recorded, not hidden by the final epoch count.
4. RoboFAC measurements in the private historical materials refer to the **official release model**, not this reproduced Full S2. No Full S2 RoboFAC/LIBERO/real-robot success-rate claim is made.
5. Only ProcVLM two-stage training and aligned vLLM inference/scoring are supported. Archived optional API/InternVL/other-backend branches are not validated or supported by this curated release.
6. Short GPU smoke comparisons establish only their measured scope, not bitwise equivalence of an entire multi-day retraining. See [VALIDATION.md](VALIDATION.md).
7. Original datasets/media, models, full predictions and deployment services remain external and subject to their own access and licensing terms.
