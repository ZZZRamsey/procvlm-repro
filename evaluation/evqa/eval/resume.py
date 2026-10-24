"""Durable, single-writer evaluation resume. No dependencies on model/GPU code."""
import ast
import fcntl
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_identity(path):
    root = Path(path).resolve(strict=True)
    files = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and (p.suffix in ('.bin', '.safetensors', '.pt', '.json') and not any(x in p.name for x in ('optimizer', 'scheduler', 'trainer_state'))):
            stat = p.stat()
            files.append([str(p.relative_to(root)), stat.st_size, stat.st_mtime_ns, file_digest(p) if p.suffix == '.json' else None])
    if not files:
        raise ValueError(f'No checkpoint files at {root}')
    return {'path': str(root), 'files': files}


def _resume_identity(value):
    """Normalize execution-only fields before comparing durable state.

    Tensor parallelism changes how inference is distributed, not which model,
    dataset, prompt policy, or sample order produced a prediction. Older
    prediction headers contain ``tp=1``; ignoring this field lets a stopped
    single-GPU run resume safely with a different TP degree.
    """
    if isinstance(value, dict):
        return {k: _resume_identity(v) for k, v in value.items() if k != 'tp'}
    if isinstance(value, list):
        return [_resume_identity(v) for v in value]
    return value


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class PredictionStore:
    """JSONL header + ordered sample records, fsync each committed batch.

    Only an unterminated final line may be discarded. Complete corruption,
    identity changes, duplicates and gaps fail closed. Batch size is not part of
    inference identity, so changing it is allowed. The lock spans inference.
    """
    def __init__(self, path, metadata, items):
        self.path = Path(path)
        self.metadata = metadata
        self.items = items
        self.predictions = []
        self.lock = None
        self.f = None
        self.fallback_limit = None
        self.progress_path = self.path.with_suffix('.progress.json')

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = open(str(self.path) + '.lock', 'a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._load()
            self.f = self.path.open('ab')
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *args):
        if self.f:
            self.f.close()
        if self.lock:
            self.lock.close()

    def _load(self):
        header = {'format': 'procvlm-eval-resume-v1', 'metadata': self.metadata, 'items_digest': digest(self.items), 'total': len(self.items)}
        if not self.path.exists():
            with self.path.open('xb') as f:
                f.write((json.dumps(header, ensure_ascii=False) + '\n').encode())
                f.flush()
                os.fsync(f.fileno())
            return
        with self.path.open('r+b') as f:
            line = f.readline()
            if not line.endswith(b'\n'):
                raise ValueError('Incomplete resume header; inspect state before continuing')
            if _resume_identity(json.loads(line)) != _resume_identity(header):
                raise ValueError('Resume checkpoint/dataset/config identity mismatch')
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    f.truncate(offset)
                    f.flush()
                    os.fsync(f.fileno())
                    break
                row = json.loads(line)
                i = len(self.predictions)
                if i >= len(self.items) or row.get('index') != i or row.get('sample_digest') != digest(self.items[i]) or not isinstance(row.get('prediction'), str):
                    raise ValueError(f'Invalid/duplicate/out-of-order resume record at {i}')
                self.predictions.append(row['prediction'])

    def append(self, predictions):
        start = len(self.predictions)
        if start + len(predictions) > len(self.items) or any(not isinstance(p, str) for p in predictions):
            raise ValueError('Invalid predictions')
        for i, pred in enumerate(predictions, start):
            row = {'index': i, 'sample_digest': digest(self.items[i]), 'prediction': pred}
            self.f.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
        self.f.flush()
        os.fsync(self.f.fileno())
        self.predictions.extend(predictions)

    def effective_batch_size(self, requested):
        if self.progress_path.exists():
            progress = json.loads(self.progress_path.read_text())
            self.fallback_limit = progress.get('oom_fallback_limit')
        return min(requested, self.fallback_limit) if self.fallback_limit else requested

    def progress(self, batch_size, status='running'):
        atomic_json(self.progress_path, {'completed': len(self.predictions), 'total': len(self.items), 'effective_batch_size': batch_size, 'oom_fallback_limit': self.fallback_limit, 'status': status})


def import_legacy(path, dataset, items):
    """Import only complete legacy sample blocks and validate their ordered prefix.

    Caller must explicitly attest the checkpoint via the original dispatch/log.
    Legacy text has no model identity; never import implicitly by filename.
    """
    text = Path(path).read_text()
    header = f'===== Evaluation Results for Dataset: {dataset} ====='
    if header not in text:
        raise ValueError(f'Legacy log has no dataset {dataset}')
    body = text.split(header, 1)[1].split('===== Evaluation Results for Dataset:', 1)[0]
    blocks = re.split(r'^----------\n', body, flags=re.M)[1:]
    predictions = []
    for i, block in enumerate(blocks):
        if not block.endswith('\n\n'):
            if i == len(blocks) - 1:
                break
            raise ValueError(f'Incomplete non-final legacy block {i}')
        match = re.fullmatch(r'Resources: (.*?)\nQuestion: (.*?)\nGround Truth: (.*?)\nPrediction: (.*)\n\n', block, re.S)
        if not match or i >= len(items):
            raise ValueError(f'Invalid legacy block {i}')
        resources, question, gt, pred = match.groups()
        item = items[i]
        expected_resources = item.get('image') or item.get('video') or None
        if ast.literal_eval(resources) != expected_resources or question != item['conversations'][0]['value'] or gt != item['conversations'][1]['value']:
            raise ValueError(f'Legacy data/order mismatch at sample {i}')
        predictions.append(pred)
    return predictions


def run_pending(store, items, generate, batch_size, fallback_batch_size=32, on_commit=None):
    if batch_size <= 0 or fallback_batch_size <= 0:
        raise ValueError('Resume batch sizes must be positive')
    bs = store.effective_batch_size(batch_size)
    store.progress(bs)
    print(f'[RESUME] completed={len(store.predictions)}/{len(items)} batch_size={bs}', flush=True)
    while len(store.predictions) < len(items):
        start = len(store.predictions)
        batch = items[start:start + bs]
        # Preserve source and identity: generators may mutate their inputs.
        import copy
        inputs = copy.deepcopy(batch)
        for item in inputs:
            item['conversations'] = item['conversations'][:1]
        oom = False
        started = time.monotonic()
        try:
            predictions = generate(inputs)
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower() or ('cuda' not in str(exc).lower() and exc.__class__.__name__ != 'OutOfMemoryError'):
                raise
            if bs <= fallback_batch_size:
                raise
            oom = True
        if oom:
            # Outside except: traceback and failed model frames can be freed.
            del inputs
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            bs = fallback_batch_size
            store.fallback_limit = bs
            store.progress(bs)
            print(f'[OOM FALLBACK] retry index={start} with batch_size={bs}', flush=True)
            continue
        if len(predictions) != len(batch):
            raise ValueError(f'Generator returned {len(predictions)} predictions for {len(batch)} samples')
        store.append(predictions)
        store.progress(bs)
        if on_commit:
            on_commit(start, predictions)
        print(f'[COMMIT] completed={len(store.predictions)}/{len(items)} batch_size={bs} batch_seconds={time.monotonic()-started:.1f}', flush=True)
    store.progress(bs, 'inference_complete')
    return store.predictions
