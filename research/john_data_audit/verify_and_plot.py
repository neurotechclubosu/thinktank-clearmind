"""Independently verify audit receipts and plot fixed, unselected example trials."""
from pathlib import Path
import collections, csv, hashlib, json
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('D:/ResearchData/thinktank-clearmind/John/2026-10-05')
OUT=Path(__file__).resolve().parents[2]/'paper_revisions/john_data_audit'
def readcsv(name):
    with (ROOT/'audit'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
records=readcsv('recordings.csv');events=readcsv('events.csv')
s=json.loads((ROOT/'audit/summary.json').read_text(encoding='utf-8'))
assert len(records)==s['recordings']==704
assert len(events)==s['events']==3510
assert len({(r['subject'],r['phoneme_index'],r['block']) for r in records})==704
assert len({e['full_task_eeg_sha256'] for e in events})==s['unique_full_task_signals']
assert sum(r['matches_original_lf']=='True' for r in records)==44
assert sum(int(r['paired_events']) for r in records)==len(events)
for sub,blocks in [('GTP000',7),('GTP003',3),('GTP004',3),('GTP007',3)]:
    rr=[r for r in records if r['subject']==sub]
    assert {int(r['phoneme_index']) for r in rr}==set(range(44))
    assert set(collections.Counter(r['phoneme_index'] for r in rr).values())=={blocks}
manifest=json.loads((ROOT/'extraction_manifest.json').read_text(encoding='utf-8'))
assert len(manifest['files'])==705
for item in manifest['files']:
    p=ROOT/'extracted'/Path(item['member'].replace('\\','/'))
    assert p.stat().st_size==item['bytes']
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']

fig,axes=plt.subplots(4,2,figsize=(11,10),constrained_layout=True)
examples=[]
for i,(sub,alias) in enumerate([('GTP000','DLR'),('GTP003','GTP003'),('GTP004','GTP004'),('GTP007','GT007')]):
    file=f'{alias}_0_1.txt'
    record=next(r for r in records if r['file']==file)
    p=ROOT/record['path'];a=np.loadtxt(p)
    e=next(e for e in events if e['file']==file and e['marker']=='1')
    start,end=int(e['start_row']),int(e['end_row_exclusive']);x=a[start:end,1:17]
    assert end>start
    assert hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()==e['full_task_eeg_sha256']
    x=x-x.mean(axis=0)
    axes[i,0].plot(np.arange(len(x))/250,x[:,0],lw=.6)
    axes[i,0].set(title=f'{sub}: index 0, block 1, marker 1, column 1',ylabel='Mean removed (recorded units)',xlabel='Nominal seconds')
    f,ps=welch(x,fs=250,nperseg=250,axis=0,detrend='constant')
    axes[i,1].semilogy(f,np.median(ps,axis=1),lw=1)
    axes[i,1].set(xlim=(1,100),title='Median PSD across 16 candidate EEG columns',xlabel='Nominal frequency (Hz)',ylabel='Recorded units²/Hz')
    axes[i,1].axvline(50,color='gray',ls=':',lw=.7);axes[i,1].axvline(60,color='gray',ls=':',lw=.7)
    examples.append(dict(subject=sub,file=file,marker=1))
fig.suptitle('Fixed raw-data examples; no filtering or calibration\nThese examples do not establish neural or imagined-speech specificity',fontsize=12)
fig.savefig(OUT/'raw_examples.png',dpi=160)
fig.savefig(OUT/'raw_examples.pdf')
checks=dict(status='PASS',recordings=704,paired_events=3510,extracted_file_hashes_verified=705,original_lf_matches=44,example_marker_hashes_verified=4,examples=examples,notes='All marker intervals are counted descriptively, not approved for biological analysis.')
(OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks))
