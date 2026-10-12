"""Audit supplied Kara event/prompt alignment; never infer imaginary phoneme onsets."""
from __future__ import annotations
import os
os.environ.setdefault('MNE_LOGGING_LEVEL','WARNING')
import json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import loadmat
import mne
from .download import sha256_file
from .kara_download import ROOT

WORDS=['gnaw','knew','pat','pot']
NONSCALP={'M1','M2','VEO','HEO','EKG','EMG','Trigger'}
SCALP=['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']

def epoch_pairs(array,n_times):
    pairs=[]
    for cell in np.asarray(array).ravel():
        pair=np.asarray(cell).ravel()
        if pair.size!=2 or not np.isfinite(pair).all() or not np.array_equal(pair,np.floor(pair)):
            raise ValueError('Unsupported MATLAB epoch-index cell')
        start,end=map(int,pair)
        if not 1<=start<=end<=n_times: raise ValueError('Epoch outside native CNT')
        pairs.append((start-1,end))  # MATLAB inclusive one-based -> Python exclusive zero-based.
    out=np.array(pairs,dtype=np.int64)
    if np.any(np.diff(out[:,0])<=0) or np.any(out[:-1,1]>out[1:,0]):
        raise ValueError('Interleaved, duplicated or unordered imagery intervals')
    return out

def word_split(y,ids):
    train=[];test=[]
    for label in range(4):
        positions=np.flatnonzero(y==label)
        positions=positions[np.argsort(ids[positions])]
        if len(positions)<8: raise ValueError('Fewer than eight usable word trials')
        cut=int(np.floor(2*len(positions)/3));fit=positions[:cut]
        train.extend(fit);test.extend(positions[cut:])
    train=np.array(sorted(train));test=np.array(sorted(test))
    inner=class_inner_folds(y,ids,train)
    if set(train)&set(test): raise AssertionError('Kara trial leakage')
    for fit,val in inner:
        if set(fit)&set(test) or set(val)&set(test) or set(fit)&set(val): raise AssertionError('Inner leakage')
        for label in range(4):
            if max(ids[fit[y[fit]==label]])>=min(ids[val[y[val]==label]]): raise AssertionError('Within-class chronology violation')
    return train,test,inner

def class_inner_folds(y,ids,train):
    folds=[([],[]),([],[])]
    for label in range(4):
        fit=train[y[train]==label];fit=fit[np.argsort(ids[fit])]
        if len(fit)<3:raise ValueError('Insufficient class-specific inner training')
        for j in range(2):
            stop=len(fit)-2+j;folds[j][0].extend(fit[:stop]);folds[j][1].append(fit[stop])
    return [(np.array(sorted(a)),np.array(sorted(b))) for a,b in folds]

def fit_eye_regression(x,eye,train):
    e=eye[train].transpose(1,0,2).reshape(eye.shape[1],-1).astype(np.float64)
    z=x[train].transpose(1,0,2).reshape(x.shape[1],-1).astype(np.float64)
    gram=e@e.T;ridge=.01*np.trace(gram)/len(gram)
    coefficients=np.linalg.solve(gram+max(ridge,1e-30)*np.eye(len(gram)),e@z.T).T
    corrected=x-np.einsum('ce,net->nct',coefficients,eye)
    return corrected.astype(np.float32),coefficients

def delivered_prompt_mapping(delivered,accepted):
    """Require a unique monotone alignment; repeats are never silently reindexed."""
    if delivered==accepted:return list(range(len(accepted)))
    # Earliest and latest feasible subsequence matches must agree everywhere.
    earliest=[];cursor=0
    for label in accepted:
        while cursor<len(delivered) and delivered[cursor]!=label:cursor+=1
        if cursor==len(delivered):raise ValueError('Accepted prompts not a subsequence of delivered prompts')
        earliest.append(cursor);cursor+=1
    latest=[];cursor=len(delivered)-1
    for label in accepted[::-1]:
        while cursor>=0 and delivered[cursor]!=label:cursor-=1
        if cursor<0:raise ValueError('Accepted prompts not a subsequence of delivered prompts')
        latest.append(cursor);cursor-=1
    latest=latest[::-1]
    if earliest!=latest:raise ValueError('Repeated/missing delivered prompts give ambiguous trial alignment')
    return earliest

def sensor_prompt_mapping(sensor,thinking,clearing,actual,accepted,task_mask=None):
    """Resolve repeats from task-colour sensor epochs, never from EEG scores."""
    sensor=np.asarray(sensor,dtype=np.float64)
    if not np.isfinite(sensor).all():raise ValueError('Nonfinite task-colour sensor')
    think_level=float(np.median([np.median(sensor[a:b]) for a,b in thinking]))
    clear_level=float(np.median([np.median(sensor[a:b]) for a,b in clearing]))
    if think_level==clear_level:raise ValueError('No distinct sensor level for imagery phase')
    threshold=(think_level+clear_level)/2
    medians=np.median(sensor[:len(sensor)//100*100].reshape(-1,100),axis=1)
    inside=medians<threshold if think_level<clear_level else medians>threshold
    edges=np.diff(np.r_[False,inside,False].astype(int))
    starts=np.flatnonzero(edges==1)*100;ends=np.flatnonzero(edges==-1)*100
    valid=((ends-starts)>=4800)&((ends-starts)<=5100)
    starts=starts[valid];ends=ends[valid]
    # Complete five-second sensor levels can also occur before the trial task.
    # Use supplied phase bounds, not a hand-picked offset or EEG score.
    in_task=(starts>=int(clearing[0,0]))&(ends<=int(thinking[-1,1])+250)
    outside=np.column_stack([starts[~in_task],ends[~in_task]]).tolist()
    starts=starts[in_task];ends=ends[in_task]
    if len(starts)!=len(actual):raise ValueError(f'Task sensor found {len(starts)} complete imagery phases for {len(actual)} delivered prompts')
    nearest=np.abs(thinking[:,0,None]-starts[None]).argmin(axis=1)
    deviation=np.abs(thinking[:,0]-starts[nearest])
    end_deviation=np.abs(thinking[:,1]-ends[nearest])
    task_mask=np.ones(len(accepted),dtype=bool) if task_mask is None else np.asarray(task_mask,dtype=bool)
    if not task_mask.any() or len(task_mask)!=len(accepted):raise ValueError('Invalid target-task marker mask')
    if len(np.unique(nearest))!=len(nearest) or np.max(deviation[task_mask])>150 or np.max(end_deviation[task_mask])>250:
        raise ValueError('Supplied epochs do not map uniquely onto task sensor phases')
    if [actual[i] for i in nearest]!=accepted:raise ValueError('Sensor-linked delivered labels disagree with released accepted prompts')
    receipt={'method':'task-colour sensor, 100ms block medians; threshold midway between supplied imagery/clearing median levels',
        'thinking_level':think_level,'clearing_level':clear_level,'threshold':threshold,
        'sensor_imagery_phase_count':len(starts),'native_delivered_prompt_count':len(actual),
        'max_supplied_onset_deviation_seconds':float(deviation.max()/1000),
        'max_supplied_end_deviation_seconds':float(end_deviation.max()/1000),
        'max_target_word_onset_deviation_seconds':float(deviation[task_mask].max()/1000),
        'max_target_word_end_deviation_seconds':float(end_deviation[task_mask].max()/1000),
        'unmatched_sensor_intervals_outside_supplied_task_bounds':outside,
        'unused_nonword_timing_outlier_indices':np.flatnonzero(~task_mask&((deviation>150)|(end_deviation>250))).tolist(),
        'unretained_attempt_indices':sorted(set(range(len(actual)))-set(nearest.tolist())),
        'internal_speech_onset_measured':False}
    return nearest.tolist(),receipt

def transform_variable_phase(x,fs=1000.,line=60.,common=True):
    x=np.asarray(x,dtype=np.float64)
    if not 4.8*fs<=x.shape[-1]<=5.1*fs: raise ValueError('Imagery interval not close to supplied five-second protocol')
    x=x-x.mean(axis=-1,keepdims=True)
    if common:x=x-x.mean(axis=0,keepdims=True)
    b,a=signal.iirnotch(line,30,fs=fs);x=signal.filtfilt(b,a,x,axis=-1)
    x=signal.sosfiltfilt(signal.butter(4,[1,45],fs=fs,btype='bandpass',output='sos'),x,axis=-1)
    samples=int(round(4*fs));start=(x.shape[-1]-samples)//2;x=x[:,start:start+samples]
    x=signal.resample_poly(x,16,125,axis=-1)
    if x.shape[-1]!=512:raise ValueError('Unexpected resampled size')
    return x.astype(np.float32)

def prepare(root,participant):
    manifest=json.loads((root/'download_manifest.json').read_text())
    receipt=manifest['participants'][participant]
    base=root/'extracted'/participant
    cnts=list(base.rglob('*.cnt'));mats=list(base.rglob('epoch_inds.mat'));labels=list(base.rglob('labels.txt'))
    if len(cnts)!=1 or len(mats)!=1 or len(labels)!=1: raise ValueError('Ambiguous/missing native EEG, indices or labels')
    original_hash={Path(e['path']).resolve():e['sha256'] for e in receipt['extracted']}
    for path in [cnts[0],mats[0],labels[0]]:
        if sha256_file(path)!=original_hash[path.resolve()]:raise ValueError('Source hash changed')
    metadata=list(base.rglob('all_features_noICA.mat')) or list(base.rglob('all_features_simple.mat'))
    if len(metadata)!=1:raise ValueError('Cannot corroborate prompt order against released integrated feature metadata')
    logs=list(base.rglob('*_p.txt'))
    if not logs:raise ValueError('Missing native delivered-prompt log')
    log_hashes={str(p):sha256_file(p) for p in logs}
    if len(set(log_hashes.values()))!=1:raise ValueError('Different native delivered-prompt logs are ambiguous')
    logs.sort(key=lambda p:('kinect_data' not in p.parts,str(p)))
    if sha256_file(metadata[0])!=original_hash[metadata[0].resolve()]:raise ValueError('Integrated metadata hash mismatch')
    fingerprint={'code_sha256':sha256_file(Path(__file__)),'mne_version':mne.__version__,
        'source_sha256':{str(p):sha256_file(p) for p in [cnts[0],mats[0],labels[0],metadata[0],logs[0]]}}
    cache=root/'processed'/f'{participant}_audited.npz';receipt=cache.with_suffix('.json')
    if cache.exists() and receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get('fingerprint')==fingerprint and saved.get('cache_sha256')==sha256_file(cache):
            with np.load(cache,allow_pickle=False) as loaded:
                x,y,ids,ptp,eye=[loaded[k] for k in ('x','y','ids','ptp','eye')]
            train,test,folds=word_split(y,ids)
            return x,y,ids,ptp,eye,train,test,folds,saved['audit']
    features=loadmat(metadata[0],squeeze_me=True,struct_as_record=False)['all_features']
    prompts=[str(v).strip() for v in np.asarray(features.prompts).ravel()]
    delivered=[v.strip() for v in labels[0].read_text().splitlines() if v.strip()]
    if prompts!=delivered:raise ValueError('Integrated metadata and label file disagree; no guess at repeated prompts')
    actual=[v.strip() for v in logs[0].read_text().splitlines() if v.strip()]
    if sha256_file(metadata[0])!=original_hash[metadata[0].resolve()]:raise ValueError('Integrated metadata hash mismatch')
    count=np.asarray(features.eeg_features.thinking_feats).size
    del features
    raw=mne.io.read_raw_cnt(str(cnts[0]),preload=False,verbose='WARNING')
    if raw.info['sfreq']!=1000:raise ValueError('Unexpected native sampling rate')
    scalp=[n for n in raw.ch_names if n not in NONSCALP]
    if scalp!=SCALP:raise ValueError('Scalp channel order differs from audited pilot; revise parser before scoring')
    mat=loadmat(mats[0]);intervals=epoch_pairs(mat['thinking_inds'],raw.n_times)
    if len(intervals)!=len(prompts) or count!=len(prompts): raise ValueError('Prompt, event and integrated EEG trial counts differ')
    marker_audit={'method':'direct or unique monotone delivered-prompt match'}
    try:mapping=delivered_prompt_mapping(actual,prompts)
    except ValueError:
        clearing=epoch_pairs(mat['clearing_inds'],raw.n_times)
        sensor=raw.get_data(picks=['EMG'])[0]
        mapping,marker_audit=sensor_prompt_mapping(sensor,intervals,clearing,actual,prompts,[p in WORDS for p in prompts])
    durations=(intervals[:,1]-intervals[:,0])/1000
    # Audit every supplied interval, including unused syllable trials.
    if np.any((durations<4.8)|(durations>5.1)):raise ValueError('Imagined intervals deviate from declared task duration')
    eyes=[n for n in ('VEO','HEO') if n in raw.ch_names]
    rows=[];xs=[];ys=[];ids=[];ptps=[];eye_xs=[]
    for ordinal,((start,end),prompt) in enumerate(zip(intervals,prompts)):
        row={'trial_id':ordinal,'prompt':prompt,'start_sample':int(start),'end_sample_exclusive':int(end),
             'duration_seconds':float((end-start)/1000),'used_four_word_task':prompt in WORDS}
        rows.append(row)
        if prompt not in WORDS:continue
        x=raw.get_data(picks=SCALP,start=int(start),stop=int(end))
        eye=raw.get_data(picks=eyes,start=int(start),stop=int(end))
        if not np.isfinite(x).all() or np.sum(np.ptp(x,axis=-1)<=1e-15)==len(SCALP):
            raise ValueError('Nonfinite/completely flat imagery trial; audit rather than silent exclusion')
        xs.append(transform_variable_phase(x));ys.append(WORDS.index(prompt));ids.append(ordinal);ptps.append(np.ptp(x,axis=-1))
        eye_xs.append(transform_variable_phase(eye,common=False))
    raw.close()
    x=np.stack(xs);y=np.array(ys);ids=np.array(ids);ptp=np.stack(ptps);eye=np.stack(eye_xs)
    train,test,folds=word_split(y,ids)
    audit={'participant':participant,'native_sampling_rate_hz':1000,'scalp_channels':SCALP,
        'native_bad_channels':raw.info['bads'],'native_recording_datetime_available':raw.info['meas_date'] is not None,
        'eye_channels':eyes,'scalp_channel_count':len(SCALP),'total_supplied_imagery_trials':len(prompts),
        'imagery_duration_seconds_range':[float(durations.min()),float(durations.max())],
        'word_counts':dict(zip(WORDS,np.bincount(y,minlength=4).tolist())),
        'label_alignment':'labels.txt equals integrated all_features.prompts; thinking_feats and imagery intervals have identical trial count; native delivered prompts have direct or unique monotone alignment',
        'accepted_to_delivered_prompt_indices':mapping,'delivered_log_sha256':sha256_file(logs[0]),
        'delivered_prompt_log_copies':[{'path':p,'sha256':h} for p,h in log_hashes.items()],
        'native_delivered_prompt_count':len(actual),
        'task_marker_alignment':marker_audit,
        'source_cnt_sha256':original_hash[cnts[0].resolve()],'source_indices_sha256':original_hash[mats[0].resolve()],
        'source_label_sha256':original_hash[labels[0].resolve()],'source_integrated_metadata_sha256':original_hash[metadata[0].resolve()],
        'physical_units':'MNE CNT calibration output; recording gain not independently calibrated by us',
        'EMG_channel':'Colour sensor per official release; excluded, not muscle signal',
        'trials':rows,'train_ids':ids[train].tolist(),'test_ids':ids[test].tolist(),
        'inner_folds':[{'train_ids':ids[a].tolist(),'validation_ids':ids[b].tolist()} for a,b in folds],
        'cross_class_chronology_overlap':bool(np.max(ids[train])>=np.min(ids[test]))}
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,x=x,y=y,ids=ids,ptp=ptp,eye=eye)
    receipt.write_text(json.dumps({'fingerprint':fingerprint,'cache_sha256':sha256_file(cache),'audit':audit},indent=2),encoding='utf-8')
    return x,y,ids,ptp,eye,train,test,folds,audit

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--participant',default='MM05')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'paper_revisions/feis_validation/kara_pilot_audit.json');a=p.parse_args()
    x,y,ids,ptp,eye,train,test,folds,audit=prepare(a.root,a.participant)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    target=a.root/'processed'/f'{a.participant}_format_audit.npz';target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target,x=x,y=y,ids=ids,ptp=ptp,eye=eye)
    print('Audited',a.participant,x.shape,'No classifier results calculated.')

if __name__=='__main__':main()
