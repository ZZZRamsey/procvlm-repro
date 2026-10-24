"""Run a preselected real-ID/OOD smoke batch through the unchanged eval chain."""
import argparse, copy, hashlib, json
from pathlib import Path
from types import SimpleNamespace
from evqa.eval.run_eval import build_batch_generator, make_abs_path, format_prompts, auto_detect_task_type
from evqa.eval.resume import digest
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--samples',required=True); p.add_argument('--model',required=True)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    records=json.loads(Path(a.samples).read_text()); items=[]
    for r in records:
        it=make_abs_path(copy.deepcopy(r['item']),a.data)
        it['conversations'][0]['value']+=format_prompts.get(auto_detect_task_type(it['conversations'][1]['value']),'')
        it['conversations']=it['conversations'][:1]; items.append(it)
    args=SimpleNamespace(model_type='procvlm',model_path=a.model,enable_value_head=False,tp=4)
    _,generate=build_batch_generator(args); preds=generate(items)
    assert len(preds)==len(items)
    out=[{'split':r['split'],'task':r['task'],'source_index':r['source_index'],
          'input_digest':digest(it),'prediction':pr} for r,it,pr in zip(records,items,preds)]
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2))
    from core.backends.vllm import shutdown_vllm
    shutdown_vllm()
if __name__=='__main__': main()
