import json
import tempfile
import unittest
from pathlib import Path
from evqa.eval.resume import PredictionStore, import_legacy, run_pending


def sample(i):
    return {'image': [f'/images/{i}.jpg'], 'conversations': [{'value': f'q{i}'}, {'value': f'gt{i}'}]}


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.items = [sample(i) for i in range(70)]
        self.meta = {'checkpoint': '3600', 'dataset': 'ID'}

    def store(self, meta=None):
        return PredictionStore(self.root / 'state.jsonl', meta or self.meta, self.items)

    def test_resume_and_changed_batch(self):
        with self.store() as s:
            s.append(['p0', 'p1'])
        with self.store() as s:
            calls = []
            run_pending(s, self.items, lambda batch: calls.append(len(batch)) or ['p'] * len(batch), 32)
            self.assertEqual(calls, [32, 32, 4])
            self.assertEqual(s.predictions[:2], ['p0', 'p1'])
            self.assertEqual(len(s.predictions), 70)
        with self.store() as s:
            run_pending(s, self.items, lambda _: self.fail('completed rerun generated'), 64)

    def test_requested_batch_can_increase_without_previous_oom(self):
        with self.store() as s:
            s.progress(4)
            s.append(['first'])
        with self.store() as s:
            self.assertEqual(s.effective_batch_size(64), 64)

    def test_incomplete_tail_only_is_removed(self):
        with self.store() as s:
            s.append(['first'])
        p = self.root / 'state.jsonl'
        with p.open('ab') as f:
            f.write(b'{"index":1,"prediction":"unfinished')
        with self.store() as s:
            self.assertEqual(s.predictions, ['first'])
            s.append(['second'])
        with self.store() as s:
            self.assertEqual(len(s.predictions), 2)

    def test_corrupt_complete_line_rejected(self):
        with self.store() as s:
            s.append(['first'])
        with (self.root / 'state.jsonl').open('ab') as f:
            f.write(b'not json\n')
        with self.assertRaises(ValueError):
            with self.store():
                pass

    def test_metadata_and_data_mismatch(self):
        with self.store() as s:
            s.append(['first'])
        with self.assertRaises(ValueError):
            with self.store({'checkpoint': '3200', 'dataset': 'ID'}):
                pass
        self.items[0]['conversations'][0]['value'] = 'changed'
        with self.assertRaises(ValueError):
            with self.store():
                pass

    def test_duplicate_rejected(self):
        with self.store() as s:
            s.append(['first'])
        p = self.root / 'state.jsonl'
        with p.open('ab') as f:
            f.write(p.read_bytes().splitlines(keepends=True)[1])
        with self.assertRaises(ValueError):
            with self.store():
                pass

    def test_lock(self):
        with self.store():
            with self.assertRaises((BlockingIOError, RuntimeError)):
                with self.store():
                    pass

    def test_interruption_does_not_repeat_completed(self):
        def interrupted(batch):
            if batch[0]['conversations'][0]['value'] == 'q32':
                raise RuntimeError('interrupted')
            return ['p'] * len(batch)
        with self.store() as s:
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                run_pending(s, self.items, interrupted, 32)
        with self.store() as s:
            self.assertEqual(len(s.predictions), 32)
            run_pending(s, self.items, lambda b: ['p'] * len(b), 64)
            self.assertEqual(len(s.predictions), 70)

    def test_prediction_count_mismatch_not_committed(self):
        with self.store() as s:
            with self.assertRaises(ValueError):
                run_pending(s, self.items, lambda _: ['p'], 64)
            self.assertEqual(s.predictions, [])

    def test_oom_64_to_32_and_persist(self):
        calls = []
        def generate(batch):
            calls.append(len(batch))
            if len(batch) > 32:
                raise RuntimeError('CUDA out of memory. simulated')
            return ['p'] * len(batch)
        with self.store() as s:
            run_pending(s, self.items, generate, 64, 32)
            self.assertEqual(calls, [64, 32, 32, 6])
            self.assertEqual(s.effective_batch_size(64), 32)
        with self.store() as s:
            self.assertEqual(s.effective_batch_size(64), 32)

    def test_oom_at32_not_silently_split(self):
        with self.store() as s:
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                run_pending(s, self.items, lambda _: (_ for _ in ()).throw(RuntimeError('CUDA out of memory.')), 32, 32)
            self.assertFalse(s.predictions)

    def test_legacy_multiline_and_partial_tail(self):
        p = self.root / 'legacy.log'
        p.write_text("===== Evaluation Results for Dataset: ID =====\n----------\nResources: ['/images/0.jpg']\nQuestion: q0\nGround Truth: gt0\nPrediction: line1\nline2\n\n----------\nResources: ['/images/1.jpg']\nQuestion: q1\nGround Truth: gt1\nPrediction: half")
        self.assertEqual(import_legacy(p, 'ID', self.items), ['line1\nline2'])
        self.items[0]['conversations'][0]['value'] = 'wrong'
        with self.assertRaises(ValueError):
            import_legacy(p, 'ID', self.items)

    def test_non_oom_error_not_retried(self):
        with self.store() as s:
            with self.assertRaisesRegex(RuntimeError, 'media missing'):
                run_pending(s, self.items, lambda _: (_ for _ in ()).throw(RuntimeError('media missing')), 64, 32)
            self.assertEqual(s.effective_batch_size(64), 64)

if __name__ == '__main__':
    unittest.main()
