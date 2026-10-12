"""Inspect small sidecars and bounded file headers, not complete EEG recordings."""
import base64
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
from inspect_public_sources import get, ROOT, HEADERS

records=[]

def blob(name, item):
    record={'name':name,'source_path':item['path'],'url':item['url']}
    try:
        raw,status,final=get(item['url'])
        obj=json.loads(raw)
        content=base64.b64decode(obj['content'])
        (ROOT/(name+'.txt')).write_bytes(content)
        record.update(status=status,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),git_blob_sha=obj['sha'],preview=content.decode('utf-8',errors='replace')[:1600])
    except Exception as exc: record['error']=str(exc)
    records.append(record)

sync=json.loads((ROOT/'synchronous_tree.json').read_text())['tree']
for name,suffix,prefix in [
    ('sync_eeg_events','_events.tsv','sub-02/ses-01/eeg/'),
    ('sync_fmri_events','_events.tsv','sub-02/ses-01/func/'),
    ('sync_eeg_sidecar','_eeg.json','sub-02/ses-01/eeg/'),
]:
    match=next(x for x in sync if x['path'].startswith(prefix) and x['path'].endswith(suffix))
    blob(name,match)
feis=json.loads((ROOT/'feis_tree.json').read_text())['tree']
blob('feis_archive_readme',next(x for x in feis if x['path']=='README.txt'))
files=[x for x in feis if x['path'].lower().endswith('.csv') and 'thinking' in x['path'].lower() and not x['path'].startswith('B059691/')]
if files:
    chosen=files[0]
    url='https://raw.githubusercontent.com/scottwellington/FEIS/v1.1/'+urllib.parse.quote(chosen['path'])
    record={'name':'feis_csv_header','source_path':chosen['path'],'url':url,'complete_file_bytes':chosen['size'],'scope':'first 4096 bytes only; incomplete sample, not a downloaded recording'}
    try:
        request=urllib.request.Request(url,headers={**HEADERS,'Range':'bytes=0-4095'})
        with urllib.request.urlopen(request,timeout=20) as response:
            content=response.read(4096)
            record.update(status=response.status,bytes_read=len(content),preview=content.decode('utf-8',errors='replace')[:1200],sha256=hashlib.sha256(content).hexdigest())
        (ROOT/'feis_csv_header.partial.txt').write_bytes(content)
    except Exception as exc: record['error']=str(exc)
    records.append(record)
url='https://www.cs.toronto.edu/~complingweb/data/karaOne/MM05.tar.bz2'
record={'name':'kara_one_archive_head','url':url,'scope':'HEAD only; no body'}
try:
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD',headers=HEADERS),timeout=20) as response:
        record.update(status=response.status,content_length=response.headers.get('Content-Length'),content_type=response.headers.get('Content-Type'),final_url=response.geturl())
except Exception as exc: record['error']=str(exc)
records.append(record)
(ROOT/'example_checks.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
for record in records: print(record['name'],record.get('status',record.get('error')))
