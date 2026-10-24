"""Public-release contracts; run with standard-library unittest."""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_no_machine_specific_defaults(self):
        for name in ['training/scripts/procvlm_train_stage.sh', 'evaluation/scripts/procvlm_eval_entrypoint.sh', 'training/evqa/data/aoss_media.py', 'evaluation/evqa/data/aoss_media.py']:
            text = (ROOT / name).read_text()
            for forbidden in ['/data/users/', '/data/share_data/', 'aoss-internal.', 'daily_backup']:
                self.assertNotIn(forbidden, text, name)
    def test_media_backend_is_explicit(self):
        for part in ['training','evaluation']:
            text=(ROOT/part/'evqa/data/aoss_media.py').read_text()
            for key in ['AWS_ENDPOINT_URL', 'AWS_DEFAULT_REGION', 'PROCVLM_S3_BUCKET', 'PROCVLM_S3_PREFIX']:
                self.assertIn('os.environ[\''+key+'\']',text)
    def test_scripts_parse(self):
        for p in ROOT.rglob('*.sh'):
            subprocess.run(['bash','-n',str(p)],check=True)
    def test_python_parses(self):
        for p in ROOT.rglob('*.py'):
            ast.parse(p.read_text(),filename=str(p))
    def test_release_launch_defaults(self):
        p=ROOT/'scripts/train.sh'
        self.assertTrue(p.exists())
        text=p.read_text()
        for item in ['PROCVLM_NNODES:-4','PROCVLM_NPROC:-4','PROCVLM_GRAD_ACCUM:-32','PROCVLM_SAVE_TOTAL_LIMIT:-1']:
            self.assertIn(item,text)
    def _capture_wrapper(self, stage=None):
        with tempfile.TemporaryDirectory(prefix='procvlm-wrapper-test-') as tmp:
            base=Path(tmp); bindir=base/'env/bin';bindir.mkdir(parents=True)
            capture=base/'capture.json'
            stub='#!'+sys.executable+'\nimport os,sys,json\nfrom pathlib import Path\nPath(os.environ["CAPTURE"]).write_text(json.dumps({"argv":sys.argv[1:],"pythonpath":os.environ["PYTHONPATH"]}))\n'
            for name in ['python','torchrun']:
                (bindir/name).write_text(stub);(bindir/name).chmod(0o700)
            (bindir/'nvidia-smi').write_text('#!/bin/sh\nprintf "test transport: no GPU accessed\\n"\n');(bindir/'nvidia-smi').chmod(0o700)
            credentials=base/'empty-test-credentials';credentials.touch()
            run=base/'outputs';(run/'stage1/procvlm_extra').mkdir(parents=True)
            (run/'stage1/procvlm_extra/procvlm_head.pt').write_bytes(b'test-only sentinel')
            env={k:v for k,v in os.environ.items() if not k.startswith(('PROCVLM_','AWS_'))}
            env.update(CAPTURE=str(capture),PROCVLM_ENV=str(bindir.parent),PROCVLM_EVAL_ENV=str(bindir.parent),PROCVLM_RUN_ROOT=str(run),PROCVLM_LOG_DIR=str(base/'eval'),PROCVLM_BASE_MODEL=str(base/'base-model'),EVQA_DATASET_ROOT=str(base/'data'),AWS_SHARED_CREDENTIALS_FILE=str(credentials),AWS_DEFAULT_REGION='test-region',AWS_ENDPOINT_URL='https://example.invalid',PROCVLM_S3_BUCKET='test-bucket',PROCVLM_S3_PREFIX='test-prefix',PROCVLM_NODE_RANK='0',PROCVLM_MASTER_ADDR='127.0.0.1',PROCVLM_MASTER_PORT='29702',PATH=str(bindir)+os.pathsep+os.environ['PATH'],TMPDIR=str(base))
            command=['bash',str(ROOT/'scripts/evaluate.sh'),str(base/'checkpoint'),'test_full']
            if stage is not None:
                env['PROCVLM_MAX_STEPS']=str(6284 if stage==1 else 2724)
                command=['bash',str(ROOT/'scripts/train.sh'),str(stage)]
            subprocess.run(command,env=env,cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            return json.loads(capture.read_text())
    def test_evaluation_wrapper_executes_expected_protocol(self):
        got=self._capture_wrapper();args=got['argv'];self.assertEqual(args[0],'evqa/eval/run_eval.py')
        for name,value in {'--model_type':'procvlm','--dataset_use':'test_full','--batch_size':'64','--tp':'4','--fallback_batch_size':'32'}.items():
            self.assertEqual(args[args.index(name)+1],value)
        self.assertIn('--resume',args);self.assertIn(str(ROOT/'evaluation/runtime_compat'),got['pythonpath'])
    def test_training_wrappers_execute_historical_topology(self):
        for stage in [1,2]:
            got=self._capture_wrapper(stage);args=got['argv']
            for value in ['--nnodes=4','--nproc_per_node=4','--node_rank=0']:
                self.assertIn(value,args)
            expected={'--per_device_train_batch_size':'1','--gradient_accumulation_steps':'32','--save_total_limit':'1','--max_steps':str(6284 if stage==1 else 2724),'--num_train_epochs':'1.2' if stage==1 else '4','--learning_rate':'1e-5' if stage==1 else '5e-6'}
            for name,value in expected.items():self.assertEqual(args[args.index(name)+1],value)
            self.assertIn(str(ROOT/'training'),got['pythonpath'])
    def test_aligned_results_are_complete(self):
        p=ROOT/'results/aligned_metrics.json'; self.assertTrue(p.exists())
        rows=json.loads(p.read_text())['results']; self.assertEqual(len(rows),16)
        self.assertEqual(sum(x['total'] for x in rows),292408)
        self.assertEqual(len({(r['model_id'],r['dataset']) for r in rows}),16)
        for r in rows:
            self.assertEqual(r['total'],r['completed']); self.assertEqual(len(r['metrics']),7)
            self.assertNotIn('path',r)
    def test_included_source_hashes(self):
        p=ROOT/'results/source_manifest.json'; self.assertTrue(p.exists())
        for r in json.loads(p.read_text())['files']:
            self.assertEqual(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest(),r['release_sha256'],r['path'])
    def test_visualization_is_self_contained(self):
        p=ROOT/'visualization/index.html'; self.assertTrue(p.exists())
        text=p.read_text(); self.assertIn('0.7201',text); self.assertNotIn('https://',text); self.assertNotIn('http://',text)

if __name__ == '__main__': unittest.main()
