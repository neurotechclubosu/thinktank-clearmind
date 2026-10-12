"""Fixed exploratory cross-session index repeatability. See PROTOCOL.md.

Three stages keep input extraction/model scoring/reporting separately auditable.
All epochs/predictions/receipts stay on D:; only reports and figures go in repo.
"""
from pathlib import Path
import argparse, collections, csv, datetime, hashlib, json, platform, shutil
import numpy as np
import scipy
from scipy import signal
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import sklearn

REPO=Path(__file__).resolve().parents[2]
AUDIT=REPO/'paper_revisions/john_data_audit'
DATA=Path('D:/ResearchData/thinktank-clearmind/John/2026-10-05')
RUN=DATA/'analysis/phoneme_repeatability_v1'
OUT=REPO/'paper_revisions/john_phoneme_analysis'
PROTOCOL=Path(__file__).with_name('PROTOCOL.md')
FS=250;CLASSES=np.arange(44);CHANCE=1/44
BANDS=((1,4),(4,8),(8,13),(13,30),(30,45))
ORDER='Fp1 Fp2 F7 F3 F4 F8 T3 C3 C4 T4 T5 T6 P3 P4 O1 O2'.split()

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def dump(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def csvread(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def transform(x,car=True):
    """channels x samples, independent phase only, no learned filter parameters."""
    x=signal.detrend(np.asarray(x,dtype=np.float64),axis=-1,type='linear')
    if car:x=x-x.mean(axis=0,keepdims=True)
    return signal.sosfiltfilt(signal.butter(4,[1,45],fs=FS,btype='bandpass',output='sos'),x,axis=-1)

def spectral(x,harmonic=False):
    f,p=signal.welch(x,fs=FS,window='hann',nperseg=250,noverlap=0,detrend='constant',axis=-1)
    mask=np.ones(len(f),dtype=bool)
    if harmonic:
        for h in [10,20,30,40]:mask &= np.abs(f-h)>1
    total=p[...,mask&(f>=1)&(f<45)].sum(axis=-1)
    if np.any(total<=0):raise ValueError('Zero spectral energy')
    parts=[]
    for lo,hi in BANDS:
        chosen=mask&(f>=lo)&(f<hi)
        if not chosen.any():raise ValueError('Harmonic mask empties band')
        rel=p[...,chosen].sum(axis=-1)/total
        parts.append(np.log(np.maximum(rel,np.finfo(float).tiny)))
    return np.stack(parts,axis=-1).reshape(len(x),-1)

def covariance(x):
    x=np.asarray(x,dtype=float);x=x-x.mean(axis=-1,keepdims=True)
    cov=np.einsum('nct,ndt->ncd',x,x)/(x.shape[-1]-1)
    scale=np.trace(cov,axis1=1,axis2=2)/16
    if np.any(scale<=0):raise ValueError('Zero covariance')
    cov=.9*cov/scale[:,None,None]+.1*np.eye(16)
    val,vec=np.linalg.eigh(cov)
    logged=np.einsum('nci,ni,ndi->ncd',vec,np.log(val.clip(1e-12)),vec)
    ii,jj=np.triu_indices(16)
    return logged[:,ii,jj]*np.where(ii==jj,1,np.sqrt(2))

def prediction(train,y,query):
    if set(y)!=set(CLASSES):raise ValueError('Training must retain all44classes')
    scaler=StandardScaler().fit(train)
    z=scaler.transform(train);q=scaler.transform(query)
    centers=np.stack([z[y==k].mean(axis=0) for k in CLASSES])
    distances=cdist(q,centers,'sqeuclidean')
    pred=distances.argmin(axis=1)
    nearest=np.partition(distances,1,axis=1)[:,:2]
    margin=np.abs(nearest[:,1]-nearest[:,0])
    return pred,distances,margin

def ba(y,p):
    # Fixed44-class denominator; absent test classes make the sensitivity invalid.
    if set(y)!=set(CLASSES):raise ValueError('Query must retain all44classes')
    return float(np.mean([np.mean(p[y==k]==k) for k in CLASSES]))

def alignment_null(y,p,session,seed=17,n=999):
    rng=np.random.default_rng(seed)
    perms=np.stack([rng.permutation(44) for _ in range(n)])
    scores=np.zeros(n)
    for s in np.unique(session):
        ids=session==s;target=y[ids];guess=p[ids]
        scores+=np.stack([np.mean(perms[:,guess[target==k]]==k,axis=1) for k in CLASSES]).mean(axis=0)
    return scores/len(np.unique(session)),perms

def upper_limits(metrics):
    mid=np.median(metrics,axis=0);mad=np.median(np.abs(metrics-mid),axis=0)
    return mid+6*1.4826*np.maximum(mad,1e-8)

def extract(run):
    if run.exists():raise FileExistsError(f'{run}: preserve existing run; do not overwrite')
    run.mkdir(parents=True)
    shutil.copy2(PROTOCOL,run/'PROTOCOL.md')
    shutil.copy2(Path(__file__),run/'analyze_snapshot.py')
    for name in ['complete_session_recordings.csv','author_metadata_2026-10-06.json','confirmed_channel_order.csv']:
        shutil.copy2(AUDIT/name,run/name)
    rec=csvread(run/'complete_session_recordings.csv')
    ev=csvread(DATA/'audit/events.csv')
    byfile=collections.defaultdict(list)
    for e in ev:byfile[e['file']].append(e)
    meta=[];task=[];wait=[];no_car=[];quality=[];receipts=[]
    for j,r in enumerate(rec):
        p=DATA/r['path']
        if sha(p)!=r['sha256']:raise ValueError(f'Changed source {p}')
        a=np.loadtxt(p);es=sorted(byfile[r['file']],key=lambda e:int(e['marker']))
        assert len(es)==5
        first_rows=[];last_rows=[]
        for e in es:
            s,end=int(e['start_row']),int(e['end_row_exclusive']);first_rows.append(s);last_rows.append(end)
            b=np.ascontiguousarray(a[s:end,1:17],dtype='<f8')
            if hashlib.sha256(b.tobytes()).hexdigest()!=e['full_task_eeg_sha256']:raise ValueError('Changed event')
            previous=last_rows[-2] if len(last_rows)>1 else 0
            if s-250<previous:raise ValueError(f'Insufficient wait {p.name}:marker{e["marker"]}')
            if end-s<375:raise ValueError('Short task window')
            tx=a[s:end,1:17].T;wx=a[s-250:s,1:17].T
            task.append(transform(tx)[:,125:375]);wait.append(transform(wx));no_car.append(transform(tx,False)[:,125:375])
            central=signal.detrend(tx[:,125:375],axis=-1,type='linear')
            metrics=np.log([np.sqrt(np.mean(central**2)),np.max(np.abs(np.diff(central,axis=-1)))])
            quality.append(metrics)
            meta.append(dict(subject=r['subject'],session=int(r['block']),file=r['file'],index=int(r['phoneme_index']),marker=int(e['marker']),task_start=s+125,task_end=s+375,wait_start=s-250,wait_end=s,full_task_hash=e['full_task_eeg_sha256'],central_task_hash=hashlib.sha256(np.ascontiguousarray(a[s+125:s+375,1:17],dtype='<f8').tobytes()).hexdigest(),start_timestamp=float(a[s,30]),file_timestamp=float(r['timestamp_start']),duration=float(e['seconds']),samples=int(e['samples']),counter=float(a[s,0]),counter_nonunit_steps=int(e['counter_nonunit_steps']),raw_rms=float(np.exp(metrics[0])),raw_derivative_max=float(np.exp(metrics[1]))))
        receipts.append(dict(file=r['file'],sha256=r['sha256'],phase_windows=5,rows=len(a)))
        if (j+1)%100==0:print(f'extracted {j+1}/{len(rec)} source files',flush=True)
    task=np.asarray(task);wait=np.asarray(wait);no_car=np.asarray(no_car)
    assert task.shape==wait.shape==no_car.shape==(2860,16,250)
    assert len({m['full_task_hash'] for m in meta})==2860
    timing=np.empty((len(meta),5))
    for sub in sorted(set(m['subject'] for m in meta)):
        for ses in sorted(set(m['session'] for m in meta if m['subject']==sub)):
            ids=[i for i,m in enumerate(meta) if m['subject']==sub and m['session']==ses]
            ordered=sorted({meta[i]['file']:meta[i]['file_timestamp'] for i in ids}.items(),key=lambda z:z[1])
            rank={name:i for i,(name,_) in enumerate(ordered)}
            assert len(rank)==44
            for i in ids:
                m=meta[i];timing[i]=[rank[m['file']]/43,m['marker']/5,m['duration'],m['samples']/500,m['counter']/255]
    features=dict(task_spectral=spectral(task),task_covariance=covariance(task),wait_spectral=spectral(wait),wait_covariance=covariance(wait),no_car_spectral=spectral(no_car),no_car_covariance=covariance(no_car),harmonic_spectral=spectral(task,True),timing=timing,quality=np.asarray(quality))
    if not all(np.isfinite(x).all() for x in features.values()):raise ValueError('Nonfinite features')
    np.savez_compressed(run/'features.npz',**features)
    np.savez_compressed(run/'epochs.npz',task=task.astype(np.float32),wait=wait.astype(np.float32))
    csvwrite(run/'epochs.csv',meta);csvwrite(run/'source_receipts.csv',receipts)
    dump(run/'extraction.json',dict(status='PASS',sources=len(rec),epochs=len(meta),train_epochs=880,test_epochs=1980,channel_order=ORDER,fs=250,shape=list(task.shape),protocol_sha256=sha(PROTOCOL),code_sha256=sha(Path(__file__)),source_manifest_sha256=sha(run/'complete_session_recordings.csv'),features_sha256=sha(run/'features.npz'),meta_sha256=sha(run/'epochs.csv'),verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    print('Extraction complete; no classifier scores computed.',flush=True)

def evaluate(run):
    if (run/'scores.csv').exists():raise FileExistsError('Already evaluated; preserve the fixed run')
    receipt=json.loads((run/'extraction.json').read_text())
    if receipt['code_sha256']!=sha(Path(__file__)):raise ValueError('Code changed since extraction')
    if receipt['protocol_sha256']!=sha(PROTOCOL):raise ValueError('Protocol changed')
    if receipt['features_sha256']!=sha(run/'features.npz') or receipt['meta_sha256']!=sha(run/'epochs.csv'):raise ValueError('Cache changed')
    m=csvread(run/'epochs.csv');f=np.load(run/'features.npz')
    subject=np.array([r['subject'] for r in m]);ses=np.array([int(r['session']) for r in m]);y=np.array([int(r['index']) for r in m])
    specs=[('task','spectral','task_spectral'),('task','covariance','task_covariance'),('wait','spectral','wait_spectral'),('wait','covariance','wait_covariance'),('no_car','spectral','no_car_spectral'),('no_car','covariance','no_car_covariance'),('harmonic','spectral','harmonic_spectral'),('timing','timing','timing'),('screened','spectral','task_spectral'),('screened','covariance','task_covariance')]
    scores=[];predictions=[];statuses=[];null_arrays={};null_perms={};screen_rows=[]
    for c,(condition,family,key) in enumerate(specs):
        x=f[key]
        for sub in np.unique(subject):
            tr=np.flatnonzero((subject==sub)&(ses==1));te=np.flatnonzero((subject==sub)&(ses!=1))
            assert len(tr)==220 and len(te)>0
            if {m[i]['file'] for i in tr}&{m[i]['file'] for i in te}:raise AssertionError('Source leakage')
            if {m[i]['full_task_hash'] for i in tr}&{m[i]['full_task_hash'] for i in te}:raise AssertionError('Duplicate leakage')
            if condition=='screened':
                thresholds=upper_limits(f['quality'][tr])
                keep=np.all(f['quality']<=thresholds,axis=1)
                tr=tr[keep[tr]];te=te[keep[te]]
                screen_rows.append(dict(subject=sub,family=family,train_kept=len(tr),test_kept=len(te),log_rms_upper=float(thresholds[0]),log_derivative_upper=float(thresholds[1])))
            if set(y[tr])!=set(CLASSES) or any(set(y[te[ses[te]==s]])!=set(CLASSES) for s in np.unique(ses[te])) or not len(te):
                statuses.append(dict(condition=condition,family=family,subject=sub,status='unestimable_missing_class_after_screen',train=len(tr),test=len(te)));continue
            pred,dist,margin=prediction(x[tr],y[tr],x[te])
            for pos,i in enumerate(te):
                predictions.append(dict(condition=condition,family=family,subject=sub,session=int(ses[i]),file=m[i]['file'],marker=int(m[i]['marker']),true=int(y[i]),prediction=int(pred[pos]),margin=float(margin[pos]),epoch_index=int(i)))
            for s in np.unique(ses[te]):
                q=ses[te]==s
                file_preds=[];file_truth=[]
                for k in CLASSES:
                    kk=q&(y[te]==k);file_preds.append(int(dist[kk].mean(axis=0).argmin()));file_truth.append(k)
                scores.append(dict(condition=condition,family=family,subject=sub,session=int(s),train=len(tr),test=int(q.sum()),balanced_accuracy=ba(y[te][q],pred[q]),file_mean_accuracy=ba(np.asarray(file_truth),np.asarray(file_preds))))
            if condition=='task':
                null,perms=alignment_null(y[te],pred,ses[te],17+np.flatnonzero(np.unique(subject)==sub)[0])
                null_arrays[f'{family}_{sub}']=null;null_perms[f'{family}_{sub}']=perms
            statuses.append(dict(condition=condition,family=family,subject=sub,status='PASS',train=len(tr),test=len(te)))
        print(f'completed fixed condition {condition}/{family}',flush=True)
    # Different-day John diagnostic: no selected source overlaps, no parameter tuning.
    for family in ['spectral','covariance']:
        tr=np.flatnonzero((subject=='GTP000')&np.isin(ses,[1,3,4]));te=np.flatnonzero((subject=='GTP000')&np.isin(ses,[5,6,7]))
        pred,dist,margin=prediction(f['task_'+family][tr],y[tr],f['task_'+family][te])
        for pos,i in enumerate(te):predictions.append(dict(condition='different_day',family=family,subject='GTP000',session=int(ses[i]),file=m[i]['file'],marker=int(m[i]['marker']),true=int(y[i]),prediction=int(pred[pos]),margin=float(margin[pos]),epoch_index=int(i)))
        for s in [5,6,7]:
            q=ses[te]==s
            fps=np.array([int(dist[q&(y[te]==k)].mean(axis=0).argmin()) for k in CLASSES])
            scores.append(dict(condition='different_day',family=family,subject='GTP000',session=s,train=len(tr),test=int(q.sum()),balanced_accuracy=ba(y[te][q],pred[q]),file_mean_accuracy=ba(CLASSES,fps)))
    csvwrite(run/'scores.csv',scores);csvwrite(run/'predictions.csv',predictions);csvwrite(run/'model_status.csv',statuses);csvwrite(run/'screen_thresholds.csv',screen_rows)
    np.savez_compressed(run/'alignment_nulls.npz',**null_arrays);np.savez_compressed(run/'label_permutations.npz',**null_perms)
    dump(run/'evaluation.json',dict(code_sha256=sha(Path(__file__)),protocol_sha256=sha(PROTOCOL),verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),score_rows=len(scores),prediction_rows=len(predictions),status='PASS',primary_train_sources=176,primary_test_sources=396,classes=44))
    print(f'Evaluation saved:{len(scores)} scores/{len(predictions)} predictions',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['extract','evaluate']);ap.add_argument('--run',type=Path,default=RUN);a=ap.parse_args()
    if a.phase=='extract':extract(a.run)
    else:evaluate(a.run)
