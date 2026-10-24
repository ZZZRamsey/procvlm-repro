# Release validation

Status: **GPU inference validated; GPU training/save/resume not validated**. Historical training/evaluation success is not substituted for checks of this release copy.

| Check | Current evidence |
|---|---|
| Archival source provenance | Per-file SHA-256 manifest; portability differences listed |
| Stage 1/2 CLI parity | Captured original/release launchers: matching numerical configuration; filesystem paths normalized |
| Entry-point imports | Training and evaluation `--help` both exit 0, GPUs disabled, isolated release paths |
| Media portability | 6 cases / 25 real frames: exact original-vs-release decoded RGB pixels |
| Static viewer | Desktop and 390px mobile renders reviewed; split filters, responsive layout and relative documentation links pass |
| Original evaluation regression suite | 25 tests passed in the existing evaluation environment |
| Release / resume contracts | 15 standard-library tests passed on macOS and Linux; production sources unchanged |
| 16 prediction-set rescoring | Passed: all 292,408 predictions, ordered sample digests and 112 scores match original and stored metrics; [machine-readable report](../results/rescore_validation.json) |
| GPU inference | Passed on SCO `gpu8`: 2 nodes × 8 A800, TP=4; original repeated twice and release once on 6 fixed ID/OOD samples; input digests, text and frame boundaries matched |
| GPU train/save/resume | Not run: this release check did not mutate training outputs; full training still requires 16 GPUs, batch1/accum32 and unchanged LR horizons |
| Public remote verification | Published to `ZZZRamsey/procvlm-repro` on `main`; local and remote refs matched |

## Commands

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
EVQA_DATASET_ROOT="$EVQA_DATASET_ROOT" PYTHONPATH="$PWD/evaluation/runtime_compat:$PWD/evaluation" "$PROCVLM_EVAL_ENV/bin/python" -m pytest evaluation/tests -q -p no:cacheprovider
```

`tests/validation/rescore.py` is a read-only audit: read JSONL records directly, verify counts/order/item digests, compute the original and release scorer outputs, and compare to the formal snapshot (`atol=1e-12`, `rtol=1e-10`). It does not instantiate the writable prediction store.

`tests/validation/inference_probe.py` executes six preselected real samples covering segmentation/progress/planning in ID/OOD, using the same Full S2 model/processor and TP4 protocol for original and release. Selection is fixed before comparison; a mismatch is not resolved by changing the sample.

`tests/validation/train_probe.py` instruments the unchanged trainer identically for both copies: fixed real packed data, same seeds and numerical configuration, stop/save after update1, reload optimizer/scheduler/RNG, resume to update2. The global 6284/2724-step LR horizons remain unchanged. Short diagnostic data changes epoch length and is **not** the full historical training run. Batch hashes, finite losses, parameter groups, counters, model/head tensors, optimizer/RNG shards are compared by `compare_training.py`.

No checks run on occupied debug-host GPUs. All new outputs/caches/checkpoints live under a task-owned temporary root, separate from archival code/data/models. Only that task's artifacts may be removed after evidence is collected. Fresh-environment installation, full retraining, all-model fresh full-split inference, optional backends and robot benchmarks are not claimed as release checks.

## Packaging observations

macOS AppleDouble metadata from the initial transfer was removed **only from the task-owned staging copy**. The Linux syntax/contract suite then passed all 15 tests. All 44 archival production/test source hashes remained unchanged by this cleanup. Optional archived API backends retain Python invalid-escape deprecation warnings; those backends are outside this release's supported scope.

## GPU allocation and inference outcome — 2026-10-25

The task `pt-f2ijfs26` received two `gpu8` workers, each with 8 A800 80 GB GPUs. A fixed six-sample TP=4 probe ran the original evaluation twice and the release evaluation once. The two original outputs were identical, and the release output matched them exactly, including input digests and generated text. This establishes the release inference path for the measured six-sample scope. A separate earlier one-shot comparison produced one wording difference on the light-switch sample; the controlled repeat showed the original run itself can produce that wording variation, so it was recorded as generation nondeterminism rather than used to weaken the comparator.

The first repeat attempt exposed an IPC socket path-length limit caused by the long task root; the rerun used a task-local short cache path and completed successfully. This was an execution-environment issue, not a source change. No GPU training, checkpoint save or optimizer/RNG resume was run in this validation task.

## Earlier allocation outcome — 2026-10-24

The earlier 4-node × 4-A800 smoke received no nodes after more than 25 minutes. A separate 1-node × 4-A800 TP4 inference attempt also remained unallocated for the bounded 120-second observation window. The model pool reported only 6 idle GPUs in aggregate; the alternate 16-GPU pool reported none. Aggregate capacity does not guarantee a schedulable four-GPU worker. Those task-owned attempts were stopped; no archival job, source, data, environment or checkpoint was changed.

Temporary-artifact cleanup is partial. Local headless-browser profiles were removed after both inference jobs reached SUSPENDED and all workers were Deleted. Cleanup jobs for `/data/users/zhulingsi/procvlm-release-smoke-20261025-002454` failed before exposing readable worker logs, so remote deletion is **not confirmed**. Compact test evidence remains outside the public source tree. External Stage 1 model symlink targets were verified unchanged. A future GPU attempt must stage a fresh isolated copy.
