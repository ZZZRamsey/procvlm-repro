"""Compare actual tensors, optimizer shards, RNG states and recorded microbatches."""
import argparse, hashlib, json, math
from pathlib import Path
import torch

def fingerprint(x,h=None):
    if h is None:h=hashlib.sha256()
    if torch.is_tensor(x):
        t=x.detach().cpu().contiguous(); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(x,dict):
        for k in sorted(x,key=str):h.update(str(k).encode()); fingerprint(x[k],h)
    elif isinstance(x,(list,tuple)):
        for y in x:fingerprint(y,h)
    elif hasattr(x,'tobytes'):h.update(x.tobytes())
    else:h.update(repr(x).encode())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--release',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); roots=[Path(a.original),Path(a.release)]; report={'passed':False,'files':[]}
    for stop in [1,2]:
        for rank in range(16):
            for stem in ['probe-start','probe-batches']:
                name=f'{stem}{stop}-rank{rank}.json'; v=[json.loads((r/name).read_text()) for r in roots]
                assert v[0]==v[1],name
                if stem=='probe-start':
                    assert v[0]['step']==stop-1 and v[0]['world_size']==16 and v[0]['accumulation']==32
                else:
                    assert len(v[0])==32 and all(math.isfinite(x['loss']) for x in v[0])
    for r in roots:
        c=r/'checkpoint-2'; assert (c/'.complete').is_file()
        state=json.loads((c/'trainer_state.json').read_text()); assert state['global_step']==2
        assert len(list(c.rglob('*optim_states.pt')))==16
        assert len(list(c.glob('rng_state_*.pth')))==16
    rels=['pytorch_model.bin','procvlm_extra/procvlm_head.pt']
    ck=roots[0]/'checkpoint-2'
    rels += [str(x.relative_to(roots[0])) for x in ck.rglob('*') if x.is_file() and (x.suffix in ['.pt','.pth'] or x.name=='scheduler.pt')]
    for name in sorted(set(rels)):
        hashes=[]
        for r in roots:
            x=torch.load(r/name,map_location='cpu',weights_only=False); hashes.append(fingerprint(x)); del x
        assert hashes[0]==hashes[1],name
        report['files'].append({'path':name,'tensor_sha256':hashes[0]});print('MATCH',name,flush=True)
    for r in roots:
        report.setdefault('counters',[]).append({k:json.loads((r/'checkpoint-2/trainer_state.json').read_text())[k] for k in ['global_step','epoch','max_steps']})
    assert report['counters'][0]==report['counters'][1]
    report['passed']=True;Path(a.output).write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
