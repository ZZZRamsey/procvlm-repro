"""CPU contracts for the actual archived resume implementation (not newer code)."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archived_resume',ROOT/'training/evqa/train/resume_alignment.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ResumeAlignmentTests(unittest.TestCase):
    def test_resume_repairs_lr_not_optimizer_moments_or_counters(self):
        moments={'exp_avg':[1,2,3],'step':3600}
        opt=SimpleNamespace(param_groups=[{'lr':0.0},{'lr':0.0}],state=moments)
        sched=SimpleNamespace(optimizer=opt,last_epoch=3600,base_lrs=[1e-5,5e-6],_last_lr=[0,0])
        m.align_scheduler(sched,3600,6284,189)
        progress=(3600-189)/max(1,6284-189)
        ratio=max(0.0,0.5*(1.0+math.cos(math.pi*progress)))
        self.assertEqual(sched.last_epoch,3600);self.assertIs(opt.state,moments)
        self.assertEqual(sched._last_lr,[1e-5*ratio,5e-6*ratio])
        self.assertEqual([g['lr'] for g in opt.param_groups],sched._last_lr)
    def test_counter_mismatch_and_terminal_resume_fail(self):
        opt=SimpleNamespace(param_groups=[{'lr':0}]);sch=SimpleNamespace(optimizer=opt,last_epoch=3599,base_lrs=[1e-5])
        with self.assertRaisesRegex(ValueError,'counter'):m.align_scheduler(sch,3600,6284,189)
        with self.assertRaises(ValueError):m.align_scheduler(sch,6284,6284,189)
    def test_latest_complete_checkpoint_and_incomplete_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in [10,20,30]:
                p=root/f'checkpoint-{i}';p.mkdir();(p/'trainer_state.json').write_text(json.dumps({'global_step':i}))
                if i!=30:(p/'.complete').write_text(str(i))
            self.assertEqual(m.select_resume(root),root/'checkpoint-20')
    def test_explicit_input_cannot_be_under_rotated_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'separate'):m.select_resume(tmp,Path(tmp)/'checkpoint-1')
    def test_historical_epoch_alignment(self):
        n=2680350;length=math.ceil(n/16);epoch=4700*32/length
        got=m.validate_alignment({'global_step':4700,'epoch':epoch},length,n,16,32,6284,1.2)
        self.assertEqual(got['updates_per_epoch'],5236)
        self.assertAlmostEqual(got['target_epoch'],1.2001886346,places=8)
        with self.assertRaisesRegex(ValueError,'ignore_data_skip'):m.validate_alignment({'global_step':4700,'epoch':epoch},length,n,16,32,6284,1.2,True)

if __name__=='__main__':unittest.main()
