"""Opt-in resumable runner; existing metric-only recovery remains unchanged."""
import copy
import json
import os
from pathlib import Path
from evqa.eval.resume import (PredictionStore, atomic_json, checkpoint_identity,
                              digest, file_digest, import_legacy, run_pending)


def run_resumable_eval(args):
    # Import lazily to leave lightweight store tests independent of torch.
    from evqa.eval.run_eval import (data_list, make_abs_path, format_prompts,
                                   auto_detect_task_type, extract_episode_token,
                                   compute_accuracy_with_episode, build_batch_generator)
    names = [x.strip() for x in args.dataset_use.split(',') if x.strip()]
    if len(names) != 1:
        raise ValueError('--resume requires exactly one dataset per output directory')
    if args.model_type != 'procvlm':
        raise ValueError('Resumable evaluation requires procvlm')
    if args.import_legacy_log:
        raise ValueError('Legacy predictions cannot be imported into the aligned input policy')
    name = names[0]
    cfg = data_list(names)[0]
    task_type = getattr(args, "task_type", None)
    if task_type not in (None, "segmentation", "progress", "planning"):
        raise ValueError("Unknown task_type")
    items = []
    source_indices = []
    source_index = -1
    with open(cfg['annotation_path']) as f:
        for line in f:
            if line.strip():
                source_index += 1
                item = make_abs_path(json.loads(line), cfg['data_path'])
                gt = item['conversations'][1]['value']
                category = auto_detect_task_type(gt)
                if task_type and category != task_type:
                    continue
                item['conversations'][0]['value'] += format_prompts.get(category, '')
                source_indices.append(source_index)
                items.append(item)
    if not items:
        raise ValueError(f'No samples selected for task_type={task_type}')
    metadata = {'checkpoint': checkpoint_identity(args.model_path),
                'dataset': name, 'annotation_path': str(Path(cfg['annotation_path']).resolve()),
                'annotation_sha256': file_digest(cfg['annotation_path']),
                'data_root': str(cfg['data_path']), 'model_type': args.model_type,
                'value_head': args.enable_value_head, 'tp': args.tp,
                'max_new_tokens': 1024 if args.enable_value_head else 8192,
                'temperature': 0.0, 'format_prompts': format_prompts,
                'generation_policy': ('procvlm-vh-interleaved-patch16-v2' if args.enable_value_head
                                      else 'procvlm-official-vllm-20260917-v1'),
                'image_patch_size': 16, 'visual_order': 'interleaved',
                'backend': 'transformers' if args.enable_value_head else 'vllm'}
    if task_type:
        metadata['task_type'] = task_type
    root = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    token = Path(args.model_path).name.replace('.', '_') + ('_with_vh' if args.enable_value_head else '_vllm')
    log = root / f'results_procvlm_{token}.log'
    state = root / f'{name}.predictions.jsonl'
    header = f"===== Evaluation Results for Dataset: {name} =====\nAnnotation Path: {cfg['annotation_path']}\nData Path: {cfg['data_path']}\n\n"

    def sample_text(i, pred):
        item = items[i]
        return (f"----------\nResources: {item.get('image') or item.get('video') or None}\n"
                f"Question: {item['conversations'][0]['value']}\n"
                f"Ground Truth: {item['conversations'][1]['value']}\nPrediction: {pred}\n\n")

    if not state.exists() and log.exists() and log.stat().st_size and not args.import_legacy_log:
        raise ValueError('Existing text log: explicit --import_legacy_log required; refusing to overwrite')
    with PredictionStore(state, metadata, items) as store:
        if task_type:
            atomic_json(root / 'selection.json', {
                'dataset': name, 'task_type': task_type,
                'annotation_sha256': metadata['annotation_sha256'],
                'source_indices': source_indices, 'items_digest': digest(items),
                'sample_digests': [digest(x) for x in items]})
        if args.import_legacy_log:
            if Path(args.import_legacy_log).resolve() == log.resolve():
                raise ValueError('Import from an immutable backup, not the live output log')
            legacy = import_legacy(args.import_legacy_log, name, items)
            if store.predictions[:min(len(legacy), len(store.predictions))] != legacy[:len(store.predictions)]:
                raise ValueError('Legacy import conflicts with committed state')
            if len(store.predictions) < len(legacy):
                store.append(legacy[len(store.predictions):])
            atomic_json(root / 'legacy_import.json', {'path': str(args.import_legacy_log),
                        'sha256': file_digest(args.import_legacy_log), 'count': len(legacy),
                        'checkpoint': metadata['checkpoint']['path']})
            print(f'[LEGACY IMPORT] validated={len(legacy)} dataset={name}', flush=True)

        # The JSONL is authoritative. Rebuild text atomically to recover a crash
        # between durable prediction commit and human-readable log append.
        tmp = log.with_suffix('.log.tmp')
        with tmp.open('w') as f:
            f.write(header)
            for i, pred in enumerate(store.predictions):
                f.write(sample_text(i, pred))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, log)

        def commit(start, preds):
            with log.open('a') as f:
                for i, pred in enumerate(preds, start):
                    f.write(sample_text(i, pred))
                f.flush()
                os.fsync(f.fileno())

        # Lazy model loading: completed datasets never need to load a model.
        generate = None
        def generate_lazy(batch):
            nonlocal generate
            if generate is None:
                _, generate = build_batch_generator(args)
            return generate(batch)

        predictions = run_pending(store, items, generate_lazy, args.batch_size,
                                  args.fallback_batch_size, commit)
        metrics = compute_accuracy_with_episode(
            pred_txts=predictions,
            gt_txts=[x['conversations'][1]['value'] for x in items],
            episodes=[extract_episode_token(x) for x in items],
            include_deprecated=False)
        atomic_json(root / 'metrics.resume.json', {'dataset': name, 'completed': len(predictions),
                    'total': len(items), 'metadata_digest': digest(metadata), 'metrics': metrics})
        with log.open('a') as f:
            f.write(f'===== Evaluation Results for Dataset: {name} =====\n')
            for key, value in metrics.items():
                line = f'{key}: {value:.4f}'
                print(line, flush=True)
                f.write(line + '\n')
            f.flush()
            os.fsync(f.fileno())
        store.progress(store.effective_batch_size(args.batch_size), 'complete')
        print(f'[EVAL COMPLETE] dataset={name} samples={len(predictions)}', flush=True)
    return 0
