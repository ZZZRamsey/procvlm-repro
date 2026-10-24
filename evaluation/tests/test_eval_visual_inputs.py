"""Protect the training/eval contract for numbered multi-image sequences."""
import copy
import os
import unittest
from unittest.mock import patch, Mock
from PIL import Image
import evqa.model as module

class VisualInputTests(unittest.TestCase):
    def test_frame_indices_stay_adjacent_to_images_and_gt_is_not_in_prompt(self):
        a, b = Image.new('RGB',(32,32),'red'), Image.new('RGB',(32,32),'blue')
        item = {'image':[a,b], 'conversations':[{'from':'human','value':'<frame 0>\n<image>\n<frame 1>\n<image>\nSegment.'}, {'from':'gpt','value':'SECRET GT'}]}
        got=module._build_qwen_messages([item])[0]
        content=got[0]['content']
        self.assertEqual([x['type'] for x in content],['text','image','text','image','text'])
        self.assertEqual([x['text'] for x in content if x['type']=='text'],['<frame 0>','<frame 1>','Segment.'])
        self.assertIs(content[1]['image'],a); self.assertIs(content[3]['image'],b)
        self.assertNotIn('SECRET GT',str(got))
        self.assertIn('<image>',item['conversations'][0]['value'])

    def test_original_vllm_path_preserves_interleaving_and_patch16(self):
        import sys
        import types
        import core.models.qwenvl as q
        processor=Mock()
        processor.apply_chat_template.return_value='prompt'
        backend=types.ModuleType('core.backends.vllm')
        backend.SamplingParams=Mock()
        backend.get_vllm_processor=Mock(return_value=processor)
        images=[Image.new('RGB',(32,32),'red'),Image.new('RGB',(32,32),'blue')]
        item={'image':images,'conversations':[{'from':'human','value':'<frame 0> <image> <frame 1> <image> Segment.'}]}
        with patch.dict(sys.modules, {'core.backends.vllm':backend}), patch.object(q,'process_vision_info',return_value=(images,None,{})) as vision:
            q.process_batch_chat_vllm([item],'/model',max_tokens=8192)
        content=processor.apply_chat_template.call_args.args[0][0]['content']
        self.assertEqual([x['type'] for x in content],['text','image','text','image','text'])
        self.assertEqual([x['text'] for x in content if x['type']=='text'],['<frame 0>','<frame 1>','Segment.'])
        self.assertIs(content[1]['image'],images[0])
        self.assertIs(content[3]['image'],images[1])
        self.assertEqual(vision.call_args.kwargs['image_patch_size'],16)

    def test_insufficient_images_fail_instead_of_silently_discarding_markers(self):
        with self.assertRaisesRegex(ValueError,'<image>'):
            module._build_qwen_messages([{'image':[], 'conversations':[{'value':'<image> hello'}]}])

    def test_video_and_image_positions_are_preserved(self):
        item={'video':'clip.mp4','image':Image.new('RGB',(32,32)), 'conversations':[{'value':'First <video> then <image> done'}]}
        item['image']=[item['image']]
        got=module._build_qwen_messages([item])[0][0]['content']
        self.assertEqual([x['type'] for x in got],['text','video','text','image','text'])

    def test_no_placeholder_media_remains_available(self):
        image=Image.new('RGB',(32,32))
        content=module._build_qwen_messages([{'image':[image],'conversations':[{'value':'Describe.'}]}])[0][0]['content']
        self.assertEqual(sum(x['type']=='image' for x in content),1)
        self.assertTrue(any(x.get('text')=='Describe.' for x in content))

    def test_both_batch_and_value_head_preprocessing_use_qwen3_patch_size(self):
        processor=Mock()
        processor.apply_chat_template.return_value='prompt'
        processor.return_value={}
        with patch.object(module,'process_vision_info',return_value=(None,None)) as vision:
            module._prepare_chunk_inputs([{'conversations':[{'value':'Text only'}]}],processor,need_per_sample_visual=True)
        self.assertEqual(vision.call_count,2)
        for call in vision.call_args_list:
            self.assertEqual(call.kwargs.get('image_patch_size'),16)

if __name__=='__main__': unittest.main()
