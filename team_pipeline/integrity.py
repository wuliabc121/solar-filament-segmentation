"""Explicit checks shared by all three model loaders and import inventories."""
import json
from pathlib import Path
from . import ROOT
from m3_filament.io import sha256

def verified_checkpoint(info,root=ROOT):
 path=Path(root)/info['checkpoint'];actual=sha256(path)
 if actual!=info['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch: '+str(path))
 return path,actual

def write_import_manifest(root=ROOT):
 root=Path(root);target=root/'team/import_manifest.json'
 entries={str(p.relative_to(root)):sha256(p) for p in sorted((root/'team').rglob('*')) if p.is_file() and p!=target and '__pycache__' not in p.parts}
 target.write_text(json.dumps(entries,indent=2,ensure_ascii=False)+'\n')
 return target
