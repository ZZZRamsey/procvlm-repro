"""Read-only scoring audit. Does not open PredictionStore or alter predictions."""
import argparse, importlib.util, json, math
from pathlib import Path
from evqa.eval.run_eval import make_abs_path, format_prompts, auto_detect_task_type
from evqa.eval.resume import digest
from evqa.eval.compute_metrics import compute_accuracy_with_episode, extract_episode_token

def main():
    p=argparse.ArgumentParser(); p.add_argument('--results-root',required=True); p.add_argument('--data',required=True)
    p.add_argument('--snapshot',required=True); p.add_argument('--original-scorer',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); spec=importlib.util.spec_from_file_location('original_scorer',a.original_scorer)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    report=[]
    for row in json.loads(Path(a.snapshot).read_text())['results']:
        split=row['dataset']; root=Path(a.results_root)/row['model_id']/split
        with (root/(split+'.predictions.jsonl')).open() as f:
            header=json.loads(next(f)); records=[json.loads(s) for s in f if s.strip()]
        data_root=header['metadata']['data_root']
        items=[]
        with (Path(a.data)/'jsons'/('test' if split=='test_full' else 'valid')/'full.jsonl').open() as f:
            for line in f:
                if not line.strip():continue
                it=make_abs_path(json.loads(line),data_root)
                it['conversations'][0]['value']+=format_prompts.get(auto_detect_task_type(it['conversations'][1]['value']),'')
                items.append(it)
        assert len(records)==len(items)==row['total']
        assert header['total']==len(items) and header['items_digest']==digest(items)
        assert digest(header['metadata'])==row['metadata_digest']
        for i,(rec,it) in enumerate(zip(records,items)):
            assert rec['index']==i and rec['sample_digest']==digest(it),(row['model_id'],split,i)
        kw=dict(pred_txts=[x['prediction'] for x in records],gt_txts=[x['conversations'][1]['value'] for x in items],
                episodes=[extract_episode_token(x) for x in items],include_deprecated=False)
        original=old.compute_accuracy_with_episode(**kw); release=compute_accuracy_with_episode(**kw)
        assert original.keys()==release.keys()==row['metrics'].keys()
        for k in release:
            assert math.isclose(release[k],original[k],abs_tol=1e-12,rel_tol=1e-10),(row['model_id'],split,k)
            assert math.isclose(release[k],row['metrics'][k],abs_tol=1e-12,rel_tol=1e-10),(row['model_id'],split,k,release[k],row['metrics'][k])
        report.append({'model_id':row['model_id'],'dataset':split,'count':len(items),'metrics':release,'passed':True})
        print(row['model_id'],split,len(items),'PASS',flush=True)
        Path(a.output).write_text(json.dumps({'atol':1e-12,'rtol':1e-10,'results':report},indent=2))
if __name__=='__main__':main()
