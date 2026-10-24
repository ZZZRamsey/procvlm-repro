"""Safety checks for resuming ProcVLM runs across a bounded max_steps restart."""
from __future__ import annotations

import json
import math
from pathlib import Path


def _underlying_scheduler(scheduler):
    """Return the torch scheduler from Accelerate's scheduler wrappers."""
    current = scheduler
    seen = set()
    while hasattr(current, "scheduler") and id(current) not in seen:
        seen.add(id(current))
        current = current.scheduler
    return current


def align_scheduler(scheduler, global_step: int, target_steps: int, warmup_steps: int) -> None:
    """Set restored optimizer LR to the target cosine schedule at global_step.

    Some bounded runs save optimizer LR=0 at their old endpoint. Preserve all
    optimizer moments and scheduler counters, but restore each param-group LR
    to its cosine value for the new target before the next optimizer update.
    """
    inner = _underlying_scheduler(scheduler)
    optimizer = getattr(scheduler, "optimizer", None) or getattr(inner, "optimizer", None)
    if optimizer is None:
        raise ValueError("cannot locate optimizer behind restored scheduler")
    if not (0 <= global_step < target_steps):
        raise ValueError(f"resume step {global_step} must be before target {target_steps}")
    if getattr(inner, "last_epoch", None) != global_step:
        raise ValueError(
            f"scheduler counter {getattr(inner, 'last_epoch', None)} does not match global step {global_step}"
        )
    base_lrs = getattr(inner, "base_lrs", None)
    if base_lrs is None or len(base_lrs) != len(optimizer.param_groups):
        raise ValueError("scheduler base LR groups do not match optimizer groups")
    if warmup_steps < 0 or warmup_steps >= target_steps:
        raise ValueError("warmup steps must be in [0, target_steps)")

    def cosine_lr(base_lr: float) -> float:
        if global_step < warmup_steps:
            return base_lr * global_step / max(1, warmup_steps)
        progress = (global_step - warmup_steps) / max(1, target_steps - warmup_steps)
        return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lrs = [cosine_lr(float(lr)) for lr in base_lrs]
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = lr
    # Trainer logs scheduler.get_last_lr(); keep that view consistent too.
    if hasattr(inner, "_last_lr"):
        inner._last_lr = list(lrs)


def _complete_checkpoint(path: Path) -> bool:
    if not (path / ".complete").is_file() or not (path / "trainer_state.json").is_file():
        return False
    try:
        state = json.loads((path / "trainer_state.json").read_text())
        return int(state["global_step"]) == int(path.name.rsplit("-", 1)[1])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def select_resume(output_dir, explicit_checkpoint=None) -> Path | None:
    """Prefer the latest complete output checkpoint, otherwise exact input checkpoint."""
    output_dir = Path(output_dir)
    checkpoint = Path(explicit_checkpoint) if explicit_checkpoint is not None else None
    if checkpoint is not None:
        if output_dir.resolve() == checkpoint.parent.resolve():
            raise ValueError("resume output directory must be separate from the source checkpoint directory")
        if output_dir.resolve() in checkpoint.resolve().parents:
            raise ValueError("resume checkpoint cannot be nested under the output directory being rotated")
    if checkpoint is not None and not _complete_checkpoint(checkpoint):
        raise FileNotFoundError(f"resume checkpoint is not complete: {checkpoint}")
    candidates = [p for p in output_dir.glob("checkpoint-*") if _complete_checkpoint(p)]
    if candidates:
        return max(candidates, key=lambda p: int(p.name.rsplit("-", 1)[1]))
    return checkpoint


def validate_alignment(state, dataloader_length, group_count, world_size, gradient_accumulation,
                       target_steps, target_epochs, ignore_data_skip=False):
    """Verify checkpoint epoch, batch geometry, and requested terminal horizon."""
    if ignore_data_skip:
        raise ValueError("ignore_data_skip must be False for exact continuation")
    if dataloader_length <= 0 or group_count <= 0 or world_size <= 0 or gradient_accumulation <= 0:
        raise ValueError("dataset and distributed batch geometry must be positive")
    expected_length = math.ceil(group_count / world_size)
    if dataloader_length != expected_length:
        raise ValueError(f"dataloader length {dataloader_length} != expected {expected_length}")
    updates_per_epoch = math.ceil(dataloader_length / gradient_accumulation)
    step = int(state["global_step"])
    expected_epoch = step * gradient_accumulation / dataloader_length
    saved_epoch = float(state["epoch"])
    if not math.isclose(saved_epoch, expected_epoch, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"checkpoint epoch {saved_epoch} != data-derived epoch {expected_epoch}")
    if target_steps <= step:
        raise ValueError("target steps must be greater than resume step")
    target_epoch = (target_steps // updates_per_epoch) + (
        (target_steps % updates_per_epoch) * gradient_accumulation / dataloader_length
    )
    if not math.isclose(target_epoch, target_epochs, rel_tol=0, abs_tol=0.001):
        raise ValueError(f"target {target_steps} steps gives epoch {target_epoch}, not {target_epochs}")
    return {
        "updates_per_epoch": updates_per_epoch,
        "skip_batches_in_epoch": (step % updates_per_epoch) * gradient_accumulation,
        "checkpoint_epoch": expected_epoch,
        "target_epoch": target_epoch,
    }
