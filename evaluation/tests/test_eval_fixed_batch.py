import unittest
from unittest.mock import Mock
import torch
from evqa.model import ProcVLMWithValueHead


class FixedBatchTests(unittest.TestCase):
    def test_actual_64_no_dynamic_split_and_no_internal_oom_halving(self):
        model = Mock()
        model.parameters.side_effect = lambda: iter([torch.zeros(1)])
        model._chunk_batch_by_vram.return_value = []
        model._infer_chunk.return_value = ['p'] * 64
        processor = Mock()
        processor.tokenizer.padding_side = 'left'
        items = [{} for _ in range(64)]
        out = ProcVLMWithValueHead.batch_infer(model, items, processor, fixed_batch=True)
        self.assertEqual(len(out), 64)
        model._chunk_batch_by_vram.assert_not_called()
        self.assertIs(model._infer_chunk.call_args.args[0], items)
        self.assertFalse(model._infer_chunk.call_args.kwargs['allow_oom_split'])

if __name__ == '__main__':
    unittest.main()
