"""Recover native delivered-prompt logs whose names use participant IDs."""
from __future__ import annotations
import argparse,json,shutil,tarfile
from pathlib import Path,PurePosixPath
from .kara_download import ROOT,PARTICIPANTS
from .download import sha256_file

def recover(root,participant):
    if participant not in PARTICIPANTS:raise ValueError('Unknown participant')
    destination=(root/'extracted'/participant).resolve();archive_path=root/'archives'/f'{participant}.tar.bz2'
    present=list(destination.rglob('*.txt'))
    if not any(f.name.endswith('_p.txt') for f in present):
        with tarfile.open(archive_path,'r|bz2') as archive:
            for member in archive:
                if not member.isfile():continue
                name=PurePosixPath(member.name.replace('\\','/'))
                if name.suffix.lower()!='.txt' or member.size>100000:continue
                if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe log path')
                output=(destination/Path(*name.parts)).resolve()
                if not output.is_relative_to(destination):raise ValueError('Unsafe extraction')
                output.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as source,output.open('wb') as stream:shutil.copyfileobj(source,stream)
                present.append(output)
    receipt={'participant':participant,'source_archive_sha256':sha256_file(archive_path),
             'logs':[{'path':str(f),'sha256':sha256_file(f),'bytes':f.stat().st_size} for f in present]}
    target=root/'delivery_metadata'/f'{participant}.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    for f in present: print(f.name,repr(f.read_text(errors='replace')[:1000]),flush=True)
    return present

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--participant',default='MM05');a=p.parse_args()
    recover(a.root,a.participant)
if __name__=='__main__':main()
