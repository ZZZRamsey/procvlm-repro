"""Tiny local jsonlines compatibility subset."""
from __future__ import annotations
import json
from pathlib import Path
class _Reader:
    def __init__(self, fp): self.fp = fp
    def __iter__(self):
        for line in self.fp:
            if line.strip(): yield json.loads(line)
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.fp.close()
class _Writer:
    def __init__(self, fp): self.fp = fp
    def write(self, obj): self.fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    def write_all(self, objs):
        for obj in objs: self.write(obj)
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.fp.close()
def open(path, mode="r", **kwargs):
    if mode not in ("r", "w", "a"): raise ValueError(f"unsupported jsonlines mode: {mode}")
    encoding = kwargs.pop("encoding", "utf-8")
    if kwargs: raise TypeError(f"unsupported arguments: {sorted(kwargs)}")
    fp = Path(path).open(mode, encoding=encoding)
    return _Reader(fp) if mode == "r" else _Writer(fp)
