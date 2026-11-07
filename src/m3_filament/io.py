"""Small artifact helpers: immutable output by default."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_text(path,text):
    """Atomically publish a new file; never replace an existing artifact."""
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=p.parent,delete=False) as f:
            tmp=Path(f.name);f.write(text);f.flush();os.fsync(f.fileno())
        os.link(tmp,p)
    finally:
        if tmp is not None:tmp.unlink(missing_ok=True)


def write_json(path,value):
    write_text(path,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_jsonl(path,records):
    write_text(path,''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))


def read_jsonl(path):
    with Path(path).open(encoding='utf8') as f:
        for line_number,line in enumerate(f,1):
            if line.strip():
                try:yield json.loads(line)
                except json.JSONDecodeError as e:raise ValueError(f'{path}:{line_number}: invalid JSON') from e
