"""Descriptive audit of John's October 2026 delivery; never executes supplied code.

Raw/extracted participant data and file/event receipts stay on D:. Output summaries
do not confer acquisition validity, ethics approval, or speech-specificity.
"""
from pathlib import Path, PurePosixPath
import argparse, ast, collections, csv, datetime, hashlib, json, re
import shutil, subprocess, zipfile, xml.etree.ElementTree as ET
import numpy as np
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[2]
UNRAR = Path('C:/Program Files/WinRAR/UnRAR.exe')
ATTACH = Path('C:/Users/Aniket/.codex/codex-remote-attachments/01a0815d-99ed-72a2-b27f-3ee5acfafe94/1C0D44C8-B869-40AC-9BE0-E1CC7D737CED')
FS = 250  # nominal command ~6; measured timing reported separately

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def csvwrite(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def safe_member(name, root):
    n=PurePosixPath(name.replace('\\','/'))
    if n.is_absolute() or '..' in n.parts or any(':' in part for part in n.parts):
        raise ValueError(f'Unsafe archive member: {name}')
    dest=(root/Path(*n.parts)).resolve()
    if not dest.is_relative_to(root.resolve()): raise ValueError(name)
    return dest

def prepare(root):
    manifest=json.loads((root/'download_manifest.json').read_text())
    extracted=root/'extracted';extracted.mkdir(exist_ok=True)
    receipts=[]
    for item in manifest['files']:
        archive=root/item['name']
        if digest(archive)!=item['sha256']: raise ValueError(f'Archive checksum changed: {archive}')
        if archive.suffix=='.zip':
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None: raise ValueError('ZIP CRC failure')
                if sum(i.file_size for i in z.infolist())>5_000_000_000: raise ValueError('Unexpected expanded size')
                for i in z.infolist():
                    dest=safe_member(i.filename,extracted)
                    if i.is_dir(): dest.mkdir(parents=True,exist_ok=True);continue
                    if dest.exists(): raise FileExistsError(dest)
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(i) as src,dest.open('xb') as out: shutil.copyfileobj(src,out)
                    receipts.append(dict(archive=archive.name,member=i.filename,bytes=dest.stat().st_size,sha256=digest(dest)))
        else:
            names=subprocess.check_output([str(UNRAR),'lb',str(archive)],text=True).splitlines()
            for name in names:
                dest=safe_member(name,extracted)
                if dest.exists(): raise FileExistsError(dest)
            subprocess.run([str(UNRAR),'t','-inul',str(archive)],check=True)
            # Names were checked before extraction; -o- also forbids overwriting.
            subprocess.run([str(UNRAR),'x','-inul','-o-',str(archive),str(extracted)+'\\'],check=True)
            for name in names:
                dest=safe_member(name,extracted)
                if dest.is_file(): receipts.append(dict(archive=archive.name,member=name,bytes=dest.stat().st_size,sha256=digest(dest)))
    meta=root/'metadata';meta.mkdir(exist_ok=True)
    for name in ['1-Photo-1.jpg','2-Photo-2.jpg','3-stimulus.pptx','4-StimPres.py']:
        dest=meta/name
        if dest.exists(): raise FileExistsError(dest)
        shutil.copy2(ATTACH/name,dest)
    dump(root/'extraction_manifest.json',dict(files=receipts,total_bytes=sum(i['bytes'] for i in receipts),attachments=[dict(name=p.name,sha256=digest(p)) for p in meta.iterdir()]))
    print(json.dumps(dict(extracted_files=len(receipts),expanded_bytes=sum(i['bytes'] for i in receipts))),flush=True)

def deck(p):
    slides=[]
    with zipfile.ZipFile(p) as z:
        names=sorted([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)],key=lambda n:int(re.search(r'slide(\d+)',n).group(1)))
        for n in names:
            xml=ET.fromstring(z.read(n))
            texts=[e.text for e in xml.iter() if e.tag.endswith('}t') and e.text]
            slides.append(dict(slide=int(re.search(r'slide(\d+)',n).group(1)),text=' | '.join(texts)))
    return dict(sha256=digest(p),slides=slides,count=len(slides))

def distribution(vals):
    x=np.asarray(vals,dtype=float);x=x[np.isfinite(x)]
    if not len(x): return None
    return dict(n=len(x),min=float(x.min()),p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()))

def audit(root, output):
    output.mkdir(parents=True,exist_ok=True)
    script=root/'metadata/4-StimPres.py'
    tree=ast.parse(script.read_text())
    slide_spec=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='slides' for t in n.targets))
    supplied=deck(root/'metadata/3-stimulus.pptx')
    recovered=deck(REPO/'paper_revisions/round2_work/sources/stimulus.pptx')
    labels=json.loads((REPO/'paper_revisions/round2_revision/inputs/stimulus_labels.json').read_text(encoding='utf-8'))
    mapping=[dict(index=i,script_label=s[0],recovered_slide=i+3,recovered_label=labels[str(i)],status='Recovered public slide label; applicability to delivered recordings requires John confirmation') for i,s in enumerate(slide_spec)]
    csvwrite(output/'stimulus_index_map.csv',mapping)
    dump(output/'stimulus_provenance.json',dict(supplied=supplied,recovered=recovered,script_sha256=digest(script),script_expected_max_slide=91))
    rows=[];events=[];errors=[];extra=[]
    for p in sorted((root/'extracted').rglob('*.txt')):
        m=re.fullmatch(r'(DLR|GTP003|GTP004|GT007)_(\d+)_(\d+)\.txt',p.name)
        if not m:
            info=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p),lf_sha256=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest())
            try:
                other=np.loadtxt(p)
                info.update(rows=len(other),columns=other.shape[1],finite=bool(np.isfinite(other).all()))
                if other.shape[1]==32:
                    info['marker_counts']={str(int(v)):int(np.sum(other[:,31]==v)) for v in np.unique(other[:,31]) if v!=0}
                    info['span_seconds']=float(other[-1,30]-other[0,30])
            except (ValueError,IndexError):info['format']='not a numeric matrix'
            extra.append(info);continue
        alias,k,block=m.groups();subject={'DLR':'GTP000','GT007':'GTP007'}.get(alias,alias)
        raw=p.read_bytes();a=np.loadtxt(p)
        row=dict(subject=subject,source_alias=alias,file=p.name,path=str(p.relative_to(root)),phoneme_index=int(k),block=int(block),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),rows=len(a),columns=a.shape[1] if a.ndim==2 else 0,finite=bool(np.isfinite(a).all()),matches_original_lf=False)
        lf=raw.replace(b'\r\n',b'\n');row['lf_sha256']=hashlib.sha256(lf).hexdigest()
        if row['columns']!=32 or not row['finite']:
            errors.append(dict(file=p.name,error='shape or nonfinite'));rows.append(row);continue
        ts=a[:,30];dt=np.diff(ts);count=a[:,0];steps=np.mod(np.diff(count),256)
        row['full_eeg_sha256']=hashlib.sha256(np.ascontiguousarray(a[:,1:17],dtype='<f8').tobytes()).hexdigest()
        row.update(timestamp_start=float(ts[0]),timestamp_end=float(ts[-1]),span_seconds=float(ts[-1]-ts[0]),timestamp_nonpositive_diffs=int(np.sum(dt<=0)),timestamp_max_gap=float(dt.max()),timestamp_median_diff=float(np.median(dt)),counter_nonunit_steps=int(np.sum(steps!=1)),counter_values_integer=bool(np.all(count==np.floor(count))),counter_range=f'{count.min():g}:{count.max():g}',eeg_std_min=float(a[:,1:17].std(axis=0).min()),aux_nonzero_columns=','.join(str(i) for i in range(17,30) if np.any(a[:,i]!=0)))
        marker=a[:,31];unique=np.unique(marker[marker!=0]);row['marker_values']=','.join(f'{v:g}' for v in unique);valid=0
        if set(unique)!=set(range(1,6)):
            errors.append(dict(file=p.name,error='Missing or unexpected marker IDs',observed=[float(v) for v in unique]))
        for v in unique:
            ix=np.flatnonzero(marker==v)
            if len(ix)!=2 or v not in range(1,6):
                errors.append(dict(file=p.name,error=f'marker {v:g} count {len(ix)}'));continue
            s,e=map(int,ix);seconds=float(ts[e]-ts[s]);n=e-s
            ev=dict(subject=subject,file=p.name,phoneme_index=int(k),block=int(block),marker=int(v),start_row=s,end_row_exclusive=e,samples=n,seconds=seconds,implied_rate=n/seconds if seconds>0 else None,counter_nonunit_steps=int(np.sum(steps[s:e-1]!=1)),timestamp_nonpositive_diffs=int(np.sum(dt[s:e-1]<=0)),first_250_available=bool(n>=250))
            x=a[s:e,1:17]
            ev['full_task_eeg_sha256']=hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
            if n>=250:
                ev['first250_eeg_sha256']=hashlib.sha256(np.ascontiguousarray(x[:250],dtype='<f8').tobytes()).hexdigest()
                std=x.std(axis=0);xc=x-x.mean(axis=0);total=float(np.var(xc,axis=0).mean())
                ev.update(eeg_std_min=float(std.min()),eeg_std_median=float(np.median(std)),eeg_std_max=float(std.max()),eeg_abs_max=float(np.max(np.abs(x))),flat_channels=int(np.sum(std==0)),common_mode_variance_ratio=float(np.var(xc.mean(axis=1))/total) if total>0 else 0)
                f,ps=welch(x,fs=FS,nperseg=min(250,n),axis=0,detrend='constant')
                def fraction(lo,hi):
                    den=ps[(f>=1)&(f<=100)].sum(axis=0);num=ps[(f>=lo)&(f<=hi)].sum(axis=0)
                    return float(np.median(np.divide(num,den,out=np.zeros_like(num),where=den>0)))
                ev['power_fraction_58_62_median']=fraction(58,62);ev['power_fraction_48_52_median']=fraction(48,52);ev['power_fraction_30_100_median']=fraction(30,100)
            events.append(ev);valid+=1
        row['paired_events']=valid
        old=REPO/'data/eeg_recordings'/p.name
        row['matches_original_lf']=bool(old.exists() and old.read_bytes().replace(b'\r\n',b'\n')==lf)
        rows.append(row)
        if len(rows)%100==0:print(f'audited {len(rows)} recordings',flush=True)
    def dups(items,key):
        groups=collections.defaultdict(list)
        for r in items:
            if key in r:groups[r[key]].append(f"{r['file']}"+(f":marker{r['marker']}" if 'marker' in r else ''))
        return [v for v in groups.values() if len(v)>1]
    duplicate_files=dups(rows,'lf_sha256');duplicate_trials=dups(events,'first250_eeg_sha256')
    duplicate_full_eeg=dups(rows,'full_eeg_sha256');duplicate_full_task=dups(events,'full_task_eeg_sha256')
    for item in extra:
        item['matches_named_files']=[r['file'] for r in rows if r['lf_sha256']==item['lf_sha256']]
    subjects=[]
    for subject in sorted(set(r['subject'] for r in rows)):
        rr=[r for r in rows if r['subject']==subject];ee=[e for e in events if e['subject']==subject]
        counts=collections.Counter(r['phoneme_index'] for r in rr)
        subjects.append(dict(subject=subject,recordings=len(rr),phonemes=len(counts),blocks=sorted(set(r['block'] for r in rr)),files_per_phoneme=dict(collections.Counter(counts.values())),paired_events=len(ee),samples=sum(r['rows'] for r in rr),recording_hours=sum(r.get('span_seconds',0) for r in rr)/3600,original_matches=sum(r['matches_original_lf'] for r in rr),event_duration=distribution([e['seconds'] for e in ee]),event_samples=distribution([e['samples'] for e in ee]),event_rate=distribution([e['implied_rate'] for e in ee if e['implied_rate']]),line60=distribution([e['power_fraction_58_62_median'] for e in ee if 'power_fraction_58_62_median' in e]),line50=distribution([e['power_fraction_48_52_median'] for e in ee if 'power_fraction_48_52_median' in e]),gamma=distribution([e['power_fraction_30_100_median'] for e in ee if 'power_fraction_30_100_median' in e]),eeg_std=distribution([e['eeg_std_median'] for e in ee if 'eeg_std_median' in e]),common_mode=distribution([e['common_mode_variance_ratio'] for e in ee if 'common_mode_variance_ratio' in e]),counter_nonunit_steps=sum(r.get('counter_nonunit_steps',0) for r in rr),files_with_timestamp_nonpositive=sum(r.get('timestamp_nonpositive_diffs',0)>0 for r in rr),flat_trial_channels=sum(e.get('flat_channels',0) for e in ee)))
    csvwrite(root/'audit/recordings.csv',rows);csvwrite(root/'audit/events.csv',events)
    dump(root/'audit/exceptions.json',errors);dump(root/'audit/extra_files.json',extra)
    dump(root/'audit/duplicates.json',dict(file_groups=duplicate_files,full_eeg_groups=duplicate_full_eeg,first250_groups=duplicate_trials,full_task_groups=duplicate_full_task))
    summary=dict(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),nominal_fs=FS,recordings=len(rows),events=len(events),subject_count=len(subjects),subjects=subjects,exceptions=errors,extra_files=extra,duplicate_file_groups=len(duplicate_files),duplicate_full_eeg_groups=len(duplicate_full_eeg),duplicate_trial_groups=len(duplicate_trials),duplicate_full_task_groups=len(duplicate_full_task),unique_full_task_signals=len(set(e['full_task_eeg_sha256'] for e in events)),supplied_deck_slides=supplied['count'],recovered_deck_slides=recovered['count'],source_code_sha256=digest(Path(__file__)),notes=['16 candidate EEG columns (1..16); physical montage, gain, reference and units need confirmation.','Paired marker endpoints indicate instructed intervals; duration is measured from timestamp column30.','Metrics descriptive, unfiltered, at nominal250Hz, and not automatic quality rejection thresholds.','Auxiliary columns not identified as independent EMG/EOG or physiological recordings.','File suffix is a block identifier; session/day semantics require verification.'])
    dump(root/'audit/summary.json',summary);dump(output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','audit']);ap.add_argument('--root',type=Path,default=Path('D:/ResearchData/thinktank-clearmind/John/2026-10-05'));ap.add_argument('--output',type=Path,default=REPO/'paper_revisions/john_data_audit');args=ap.parse_args()
    if args.phase=='prepare':prepare(args.root)
    else:audit(args.root,args.output)
