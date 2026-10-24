"""Instrumentation only: stop/save at a boundary without shortening the LR horizon.
Run through torchrun, with PROCVLM_PROBE_SOURCE and PROCVLM_PROBE_STOP set.
Production source is imported unchanged; identical hooks apply to both copies.
"""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
import torch
from transformers import TrainerCallback
root=Path(os.environ['PROCVLM_PROBE_SOURCE']).resolve()
sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'evqa/train'))
spec=importlib.util.spec_from_file_location('procvlm_probed_train',root/'evqa/train/train_qwen.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
stop=int(os.environ['PROCVLM_PROBE_STOP'])

def hash_value(x,h):
    if torch.is_tensor(x):
        t=x.detach().cpu().contiguous(); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(x,dict):
        for k in sorted(x,key=str): h.update(str(k).encode()); hash_value(x[k],h)
    elif isinstance(x,(tuple,list)):
        for y in x: hash_value(y,h)
    else: h.update(repr(x).encode())

class Probe(TrainerCallback):
    def on_step_end(self,args,state,control,**kw):
        if state.global_step >= stop:
            control.should_save=True; control.should_training_stop=True
        return control
    def on_train_begin(self,args,state,control,optimizer=None,**kw):
        params=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]
        out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
        (out/f'probe-start{stop}-rank{args.process_index}.json').write_text(json.dumps({
            'step':state.global_step,'epoch':state.epoch,'world_size':args.world_size,
            'accumulation':args.gradient_accumulation_steps,'horizon':state.max_steps,
            'parameter_groups':params},default=str,indent=2))

original_init=m.Qwen3VLTrainer.__init__
def init(self,*a,**kw):
    original_init(self,*a,**kw); self.add_callback(Probe()); self._probe_batches=[]
m.Qwen3VLTrainer.__init__=init
original_step=m.Qwen3VLTrainer.training_step
def step(self,model,inputs,*a,**kw):
    h=hashlib.sha256(); hash_value(inputs,h)
    out=original_step(self,model,inputs,*a,**kw)
    self._probe_batches.append({'input_sha256':h.hexdigest(),'loss':float(out.detach().cpu())})
    path=Path(self.args.output_dir)/f'probe-batches{stop}-rank{self.args.process_index}.json'
    path.write_text(json.dumps(self._probe_batches))
    return out
m.Qwen3VLTrainer.training_step=step
m.train(attn_implementation='flash_attention_2')
