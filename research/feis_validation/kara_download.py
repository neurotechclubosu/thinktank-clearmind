"""Acquire the official Kara pilot on D:, preserving integrity and release terms."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.request
from .download import sha256_file

BASE = 'https://www.cs.toronto.edu/~complingweb/data/karaOne/'
ROOT = Path('D:/ResearchData/thinktank-clearmind/KaraOne')
PARTICIPANTS = ['MM05','MM08','MM09','MM10','MM11','MM12','MM14','MM15','MM16','MM18','MM19','MM20','MM21','P02']

def fetch_part(url, target, start, end, etag):
    need = end-start+1
    have = target.stat().st_size if target.exists() else 0
    if have > need:
        raise ValueError('Unexpected existing part size')
    for attempt in range(4):
        have = target.stat().st_size if target.exists() else 0
        if have == need:
            return
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={start+have}-{end}', 'If-Range': etag})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start+have}-{end}/{TOTALS[url]}':
                    raise ValueError('Server did not honor the verified byte range')
                with target.open('ab') as stream:
                    shutil.copyfileobj(response, stream, 1024*1024)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2*(attempt+1))
    if target.stat().st_size != need:
        raise ValueError('Incomplete download part')

TOTALS = {}

def acquire(participant, root):
    if participant not in PARTICIPANTS:
        raise ValueError('Unknown official participant archive')
    url=BASE+participant+'.tar.bz2'
    target=root/'archives'/f'{participant}.tar.bz2'
    target.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=40) as response:
        if response.status !=206:
            raise ValueError('Byte-range source audit unavailable')
        total=int(response.headers['Content-Range'].split('/')[-1])
        etag=response.headers.get('ETag','')
        modified=response.headers.get('Last-Modified')
    TOTALS[url]=total
    if not target.exists():
        parts=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[]
            for i in range(4):
                start=total*i//4; end=total*(i+1)//4-1
                part=target.with_suffix(target.suffix+f'.part{i}')
                parts.append(part)
                futures.append(pool.submit(fetch_part,url,part,start,end,etag))
            for f in futures: f.result()
        temporary=target.with_suffix(target.suffix+'.joining')
        with temporary.open('wb') as stream:
            for part in parts:
                with part.open('rb') as source: shutil.copyfileobj(source,stream,1024*1024)
        if temporary.stat().st_size != total: raise ValueError('Size mismatch')
        temporary.replace(target)
        for part in parts: part.unlink()  # Four named download intermediates only.
    if target.stat().st_size != total: raise ValueError('Existing archive size mismatch')
    records=[]
    destination=(root/'extracted'/participant).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(target,'r|bz2') as archive:
        for member in archive:
            if not member.isfile(): continue
            name=PurePosixPath(member.name.replace('\\','/'))
            if name.is_absolute() or '..' in name.parts: raise ValueError('Unsafe archive member')
            wanted = name.suffix.lower() in ('.cnt','.mat') or name.name in ('ID.txt','ID_p.txt','labels.txt','ID_ind.txt',participant+'.txt',participant+'_p.txt',participant+'_ind.txt')
            if not wanted: continue
            output=(destination/Path(*name.parts)).resolve()
            if not output.is_relative_to(destination): raise ValueError('Extraction outside dataset')
            output.parent.mkdir(parents=True,exist_ok=True)
            source=archive.extractfile(member)
            with output.open('wb') as stream: shutil.copyfileobj(source,stream,1024*1024)
            records.append({'path':str(output),'bytes':output.stat().st_size,'sha256':sha256_file(output)})
    return {'participant':participant,'source_url':url,'archive_path':str(target),'bytes':total,
            'etag':etag,'last_modified':modified,'sha256':sha256_file(target),'extracted':records,
            'publisher_checksum_available':False,'checked_utc':datetime.now(timezone.utc).isoformat()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--participants',nargs='+',default=['MM05']);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    root=a.root.resolve()
    if not root.is_relative_to(Path('D:/ResearchData/thinktank-clearmind').resolve()):
        raise ValueError('This acquisition is confined to the authorized D: research directory')
    root.mkdir(parents=True,exist_ok=True)
    for path in ['karaOne.html','src/split_data.m','src/get_all_features.m','src.zip']:
        target=root/'source_metadata'/path;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            with urllib.request.urlopen(BASE+path,timeout=40) as response,target.open('wb') as stream: shutil.copyfileobj(response,stream)
    manifest_path=root/'download_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'dataset':'Kara One official release','release_version':'not provided','terms':'Academic non-profit; cite Zhao and Rudzicz (2015). Retain official release page.','participants':{}}
    if not 1<=a.workers<=2:raise ValueError('At most two independent participant downloads')
    def job(participant):
        print('Downloading/auditing',participant,flush=True)
        return acquire(participant,root)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for participant,record in zip(a.participants,pool.map(job,a.participants)):
            manifest['participants'][participant]=record
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print('Verified local archive and extracted EEG',participant,flush=True)

if __name__=='__main__': main()
