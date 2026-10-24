import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
os.environ.setdefault('EVQA_DATASET_ROOT', '/tmp/procvlm-test-dataset')
from evqa.eval import run_eval
from evqa.eval.resume_eval import run_resumable_eval


class RunnerTests(unittest.TestCase):
    def test_end_to_end_interrupt_resume_metrics_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ckpt = root / 'checkpoint-3600'
            ckpt.mkdir()
            (ckpt / 'config.json').write_text('{}')
            anno = root / 'data.jsonl'
            anno.write_text(''.join(json.dumps({'image':[f'{i}.jpg'],'conversations':[{'from':'human','value':f'q{i}'},{'from':'gpt','value':f'g{i}'}]})+'\n' for i in range(5)))
            args = SimpleNamespace(dataset_use='test_full',model_path=str(ckpt),model_type='procvlm',enable_value_head=True,tp=1,log_dir=str(root/'out'),batch_size=2,fallback_batch_size=1,import_legacy_log=None,resume=True)
            calls=[]
            def generate(batch):
                ids=[x['conversations'][0]['value'] for x in batch]
                calls.extend(ids)
                if 'q2' in ids:
                    raise RuntimeError('interrupted')
                return ['pred'+x for x in ids]
            with patch.object(run_eval,'data_list',return_value=[{'annotation_path':str(anno),'data_path':'/images'}]), patch.object(run_eval,'auto_detect_task_type',return_value='x'), patch.object(run_eval,'extract_episode_token',return_value='episode'), patch.object(run_eval,'compute_accuracy_with_episode',return_value={'metric':1.0}) as metrics, patch.object(run_eval,'build_batch_generator',return_value=('procvlm',generate)) as builder:
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    run_resumable_eval(args)
                metrics.assert_not_called()
                self.assertFalse((root/'out/metrics.resume.json').exists())
                calls.clear()
                def resumed(batch):
                    calls.extend(x['conversations'][0]['value'] for x in batch)
                    return ['pred'+x['conversations'][0]['value'] for x in batch]
                builder.return_value=('procvlm',resumed)
                args.batch_size=64
                run_resumable_eval(args)
                self.assertEqual(calls,['q2','q3','q4'])
                self.assertEqual(metrics.call_args.kwargs['pred_txts'],['predq'+str(i) for i in range(5)])
                builder.reset_mock()
                run_resumable_eval(args)
                builder.assert_not_called()
                log=next((root/'out').glob('results*.log')).read_text()
                self.assertEqual(log.count('\nPrediction:'),5)
                self.assertEqual(log.count('metric: 1.0000'),1)
                (ckpt/'config.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    run_resumable_eval(args)

    def test_vllm_interrupt_resume_metrics_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ckpt = root / 'checkpoint-3600'
            ckpt.mkdir()
            (ckpt / 'config.json').write_text('{}')
            anno = root / 'data.jsonl'
            anno.write_text(''.join(json.dumps({'image':[f'{i}.jpg'],'conversations':[{'from':'human','value':f'q{i}'},{'from':'gpt','value':f'g{i}'}]})+'\n' for i in range(5)))
            args = SimpleNamespace(dataset_use='test_full',model_path=str(ckpt),model_type='procvlm',enable_value_head=False,tp=1,log_dir=str(root/'out'),batch_size=2,fallback_batch_size=1,import_legacy_log=None,resume=True)
            calls=[]
            def generate(batch):
                ids=[x['conversations'][0]['value'] for x in batch]
                calls.extend(ids)
                if 'q2' in ids:
                    raise RuntimeError('interrupted')
                return ['pred'+x for x in ids]
            with patch.object(run_eval,'data_list',return_value=[{'annotation_path':str(anno),'data_path':'/images'}]), patch.object(run_eval,'auto_detect_task_type',return_value='x'), patch.object(run_eval,'extract_episode_token',return_value='episode'), patch.object(run_eval,'compute_accuracy_with_episode',return_value={'metric':1.0}) as metrics, patch.object(run_eval,'build_batch_generator',return_value=('procvlm',generate)) as builder:
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    run_resumable_eval(args)
                metrics.assert_not_called()
                self.assertFalse((root/'out/metrics.resume.json').exists())
                calls.clear()
                def resumed(batch):
                    calls.extend(x['conversations'][0]['value'] for x in batch)
                    return ['pred'+x['conversations'][0]['value'] for x in batch]
                builder.return_value=('procvlm',resumed)
                args.batch_size=64
                run_resumable_eval(args)
                self.assertEqual(calls,['q2','q3','q4'])
                self.assertEqual(metrics.call_args.kwargs['pred_txts'],['predq'+str(i) for i in range(5)])
                builder.reset_mock()
                run_resumable_eval(args)
                builder.assert_not_called()
                log=next((root/'out').glob('results*.log')).read_text()
                self.assertEqual(log.count('\nPrediction:'),5)
                self.assertEqual(log.count('metric: 1.0000'),1)
                (ckpt/'config.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    run_resumable_eval(args)

    def test_task_filter_preserves_source_rows_prompts_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); ckpt = root / 'model'; ckpt.mkdir()
            (ckpt / 'config.json').write_text('{}')
            anno = root / 'data.jsonl'
            rows = [{'image': [f'{i}.jpg'], 'conversations': [
                {'from': 'human', 'value': f'q{i}'},
                {'from': 'gpt', 'value': task}]} for i, task in enumerate(['progress', 'segmentation', 'progress', 'segmentation'])]
            anno.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            args = SimpleNamespace(dataset_use='test_full', model_path=str(ckpt),
                model_type='procvlm', enable_value_head=True, tp=1, log_dir=str(root/'out'),
                batch_size=64, fallback_batch_size=32, import_legacy_log=None,
                resume=True, task_type='segmentation')
            questions = []
            def generate(batch):
                questions.extend(x['conversations'][0]['value'] for x in batch)
                self.assertTrue(all(len(x['conversations']) == 1 for x in batch))
                return ['prediction'] * len(batch)
            with patch.object(run_eval, 'data_list', return_value=[{'annotation_path': str(anno), 'data_path': '/images'}]), patch.object(run_eval, 'auto_detect_task_type', side_effect=lambda gt: gt), patch.object(run_eval, 'extract_episode_token', return_value='e'), patch.object(run_eval, 'compute_accuracy_with_episode', return_value={'BF1@5': 0.5}) as metric, patch.object(run_eval, 'build_batch_generator', return_value=('procvlm', generate)) as builder:
                run_resumable_eval(args)
                suffix = run_eval.format_prompts['segmentation']
                self.assertEqual(questions, ['q1'+suffix, 'q3'+suffix])
                self.assertEqual(metric.call_args.kwargs['gt_txts'], ['segmentation']*2)
                manifest = json.loads((root/'out/selection.json').read_text())
                self.assertEqual(manifest['source_indices'], [1, 3])
                builder.reset_mock()
                run_resumable_eval(args)
                builder.assert_not_called()
                args.task_type = None
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    run_resumable_eval(args)

    def test_empty_task_selection_refuses_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'config.json').write_text('{}')
            anno = root/'data.jsonl'
            anno.write_text(json.dumps({'conversations':[{'value':'q'}, {'value':'progress'}]})+'\n')
            args = SimpleNamespace(dataset_use='test_full', model_path=str(root),
                model_type='procvlm', enable_value_head=True, tp=1, log_dir=str(root/'out'),
                batch_size=64, fallback_batch_size=32, import_legacy_log=None,
                resume=True, task_type='segmentation')
            with patch.object(run_eval, 'data_list', return_value=[{'annotation_path':str(anno), 'data_path':'/images'}]), patch.object(run_eval, 'auto_detect_task_type', return_value='progress'), patch.object(run_eval, 'build_batch_generator') as builder:
                with self.assertRaisesRegex(ValueError, 'No samples'):
                    run_resumable_eval(args)
                builder.assert_not_called()

if __name__=='__main__':
    unittest.main()
