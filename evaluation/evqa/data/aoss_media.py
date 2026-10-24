"""Read indexed JPEG/PNG tar members with S3 Range requests into RAM only."""
from collections import OrderedDict
import hashlib
import io
import json
import mmap
import os
from pathlib import Path

import boto3
from botocore.config import Config
import numpy as np
from PIL import Image

class MediaReader:
    def __init__(self, index_root, client=None, cache_bytes=128*1024**2):
        self.root=Path(index_root)
        self.paths=np.load(self.root/'paths.npy',mmap_mode='r')
        self.hashes=np.load(self.root/'hashes.npy',mmap_mode='r')
        if self.hashes.dtype != np.dtype('uint64') or not self.hashes.flags.c_contiguous:
            raise ValueError('Media hashes must be a contiguous uint64 index')
        self.locations=np.load(self.root/'locations.npy',mmap_mode='r')
        with (self.root.parent/'filelist.txt').open('rb') as f:
            self.names=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        self.client=client or boto3.client('s3',region_name=os.environ['AWS_DEFAULT_REGION'],
            endpoint_url=os.environ['AWS_ENDPOINT_URL'],
            config=Config(max_pool_connections=16,retries={'max_attempts':8,'mode':'standard'},read_timeout=120))
        self.cache=OrderedDict()
        self.cache_bytes=cache_bytes
        self.used=0
        self.bytes_read=0
        self.requests=0

    def lookup(self, path):
        name=path.encode('utf-8')
        # A Python int below 2**63 makes NumPy promote uint64+int64 to float64,
        # copying the entire ~478 MB table on EACH search. Keep the query dtype
        # identical to the index, preserving exact 64-bit comparisons as well.
        key=np.uint64(int.from_bytes(hashlib.blake2b(name,digest_size=8).digest(),'little'))
        begin=int(np.searchsorted(self.hashes,key,side='left'))
        end=int(np.searchsorted(self.hashes,key,side='right'))
        for record in self.paths[begin:end]:
            off=int(record['offset'])
            finish=self.names.find(b'\n',off)
            if finish<0:finish=len(self.names)
            if self.names[off:finish].rstrip(b'\r')==name:
                idx=int(record['id'])
                if self.locations[idx]['size']:
                    return idx
        raise FileNotFoundError(f'Media path unavailable in verified S3 index: {path}')

    def _put(self, idx, data):
        if len(data)>self.cache_bytes:return
        if idx in self.cache:self.used-=len(self.cache.pop(idx))
        self.cache[idx]=data
        self.used+=len(data)
        while self.used>self.cache_bytes:
            _,removed=self.cache.popitem(last=False)
            self.used-=len(removed)

    def read_many(self, paths):
        ids=[self.lookup(path) for path in paths]
        payloads={i:self.cache[i] for i in set(ids) if i in self.cache}
        for idx in payloads:self.cache.move_to_end(idx)
        missing=sorted((int(self.locations[i]['shard']),int(self.locations[i]['offset']),int(self.locations[i]['size']),i)
                       for i in set(ids) if i not in payloads)
        groups=[]
        for entry in missing:
            shard,off,size,idx=entry
            if groups and shard==groups[-1][0][0] and off-(groups[-1][-1][1]+groups[-1][-1][2])<=65536 and off+size-groups[-1][0][1]<=8*1024**2:
                groups[-1].append(entry)
            else:groups.append([entry])
        for group in groups:
            shard,start,_,_=group[0]
            end=max(off+size for _,off,size,_ in group)-1
            obj=self.client.get_object(Bucket=os.environ['PROCVLM_S3_BUCKET'],
                Key=f"{os.environ['PROCVLM_S3_PREFIX'].rstrip('/')}/shards/shard-{shard:06d}.tar",
                Range=f'bytes={start}-{end}')
            body=obj['Body']
            try:data=body.read()
            finally:body.close()
            if len(data)!=end-start+1 or not obj.get('ContentRange','').startswith(f'bytes {start}-{end}/'):
                raise IOError('S3 did not return the requested tar range')
            self.requests+=1
            self.bytes_read+=len(data)
            for _,off,size,idx in group:
                payload=data[off-start:off-start+size]
                payloads[idx]=payload
                self._put(idx,payload)
        result=[]
        for idx in ids:
            with Image.open(io.BytesIO(payloads[idx])) as image:
                result.append(image.convert('RGB'))
        return result

_reader=None
_reader_pid=None
def read_images(paths):
    global _reader,_reader_pid
    if _reader is None or _reader_pid!=os.getpid():
        _reader=MediaReader(os.environ['PROCVLM_MEDIA_INDEX'])
        _reader_pid=os.getpid()
    return _reader.read_many(paths)
