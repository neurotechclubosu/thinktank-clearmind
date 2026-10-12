"""Run the bounded exploratory FEIS follow-up; original results are untouched."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from scipy import signal
import scipy,sklearn,torch
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from .data import load_recording,chronological_split,CHANNELS,LABELS
from .download import DEFAULT_ROOT,PROJECT_ROOT,sha256_file
from .benchmark import classifier,strict_mask
from .features import band_features
from .followup_models import (signal_transform,covariance_features,covariance_classifier,
    inner_block_folds,select_covariance,select_neural,neural_predictions,permute_within,SEEDS)

OUTPUT=PROJECT_ROOT/'paper_revisions/feis_validation/results/followup_v1'

def prepare(root,participant,phase):
    manifest=json.loads((root/'download_manifest.json').read_text())
    original=root/f'extracted/experiments/{participant}/{phase}/{phase}.csv'
    if phase in ('stimuli','speaking'):
        if participant!='01': raise ValueError('Diagnostic full recording exists only for pilot01')
        original=root/'extracted/experiments/01/full_eeg/full_eeg.csv'
        entry=manifest['files']['experiments/01/full_eeg.zip']
    else: entry=manifest['files'][f'experiments/{participant}/{phase}.zip']
    expected=next(e['sha256'] for e in entry['extracted'] if Path(e['path']).name==original.name)
    checksum=sha256_file(original)
    if checksum!=expected: raise ValueError('Original CSV hash mismatch')
    source=original
    if phase in ('stimuli','speaking'):
        frame=pd.read_csv(original)
        source=root/f'processed/followup_v1/01_{phase}_derived.csv'
        source.parent.mkdir(parents=True,exist_ok=True)
        frame.loc[frame.Stage==phase].to_csv(source,index=False,float_format='%.9f')
    rec=load_recording(source,phase,timestamp_atol=1e-6)
    raw=rec.x.copy(); ptp=np.ptp(raw,axis=-1)
    f,psd=signal.welch(raw,fs=rec.fs,nperseg=int(rec.fs),noverlap=0,axis=-1)
    total=psd[..., (f>=1)&(f<=100)].sum(axis=-1).clip(1e-12)
    fractions={name:psd[...,mask].sum(axis=-1)/total for name,mask in {
        'line50_fraction':(f>=49)&(f<=51),'line60_fraction':(f>=59)&(f<=61),
        'high45_90_fraction':(f>=45)&(f<=90)}.items()}
    common_rms=np.std(raw.mean(axis=1),axis=-1)
    channel_rms=np.sqrt(np.mean(np.var(raw,axis=-1),axis=1)).clip(1e-12)
    nuisance=np.column_stack([np.log(ptp.clip(1e-12)),raw.mean(axis=-1),
                             *[v for v in fractions.values()],rec.flagged.astype(float)])
    rec.audit.update(participant=participant,original_source=str(original),original_source_sha256=checksum,
       derived_phase_source=phase in ('stimuli','speaking'),raw_ptp_median=float(np.median(ptp)),
       raw_ptp_max=float(np.max(ptp)),flagged_trials=int(rec.flagged.sum()),
       common_mode_rms_fraction_median=float(np.median(common_rms/channel_rms)),
       spectral_fractions_median={k:float(np.median(v)) for k,v in fractions.items()},
       spectral_fractions_channel_medians={k:dict(zip(CHANNELS,np.median(v,axis=0).tolist())) for k,v in fractions.items()})
    rec.x=signal_transform(raw,rec.fs)
    cache=root/f'processed/followup_v1/{participant}_{phase}.npz'
    cache.parent.mkdir(parents=True,exist_ok=True)
    cov=covariance_features(rec.x)
    np.savez_compressed(cache,x=rec.x,y=rec.y,trial_ids=rec.trial_ids,raw_ptp=ptp,nuisance=nuisance,covariance=cov)
    rec.audit['processed_cache_sha256']=sha256_file(cache)
    return rec,ptp,nuisance,cov

def run_one(root,participant,phase,output,large,null_refits):
    start=time.perf_counter()
    rec,ptp,nuisance,cov=prepare(root,participant,phase)
    train,test=chronological_split(rec)
    blocks=np.array([rec.audit['id_to_block'][int(i)] for i in rec.trial_ids])
    folds=inner_block_folds(train,blocks)
    c,c_cv=select_covariance(cov,rec.y,folds)
    epochs,nn_cv=select_neural(rec.x,rec.y,folds)
    scores=[];predictions=[]; selections={'covariance':c_cv,'eegnet_style':nn_cv}
    selected={'covariance':c,'eegnet_style':epochs}
    def record(model,pred,seed=17,screening='primary',indices=test):
        score=float(balanced_accuracy_score(rec.y[indices],pred))
        scores.append({'participant':participant,'phase':phase,'model':model,'seed':seed,
            'screening':screening,'balanced_accuracy':score,'n_train':len(train),'n_test':len(indices),
            'selected_parameter':selected.get(model)})
        for index,label in zip(indices,pred):
            predictions.append({'participant':participant,'phase':phase,'model':model,'seed':seed,
                'screening':screening,'trial_id':int(rec.trial_ids[index]),'block':int(blocks[index]),
                'true_label':int(rec.y[index]),'predicted_label':int(label)})
        return pred
    cov_pred=record('covariance',covariance_classifier(c).fit(cov[train],rec.y[train]).predict(cov[test]))
    nn_preds={}
    for seed in SEEDS:
        nn_preds[seed]=record('eegnet_style',neural_predictions(rec.x[train],rec.y[train],rec.x[test],[epochs],seed)[epochs],seed)
    context=band_features(rec.x,128)
    record('mean_power_lda',classifier().fit(context[train],rec.y[train]).predict(context[test]))
    nuisance_c,nuisance_cv=select_covariance(nuisance,rec.y,folds)
    selections['nuisance']=nuisance_cv;selected['nuisance']=nuisance_c
    record('nuisance',covariance_classifier(nuisance_c).fit(nuisance[train],rec.y[train]).predict(nuisance[test]))
    keep=strict_mask(ptp,rec.flagged,train)
    good_train=train[keep[train]];good_test=test[keep[test]]
    strict={'n_train':len(good_train),'n_test':len(good_test),'excluded_trial_ids':rec.trial_ids[~keep].tolist()}
    if len(np.unique(rec.y[good_train]))==16 and len(np.unique(rec.y[good_test]))==16:
        if keep.all():
            record('covariance',cov_pred,screening='strict')
            record('eegnet_style',nn_preds[17],screening='strict')
            strict['same_records_as_primary']=True
        else:
            record('covariance',covariance_classifier(c).fit(cov[good_train],rec.y[good_train]).predict(cov[good_test]),screening='strict',indices=good_test)
            record('eegnet_style',neural_predictions(rec.x[good_train],rec.y[good_train],rec.x[good_test],[epochs],17)[epochs],screening='strict',indices=good_test)
        for entry in scores:
            if entry['screening']=='strict': entry['n_train']=len(good_train)
    else: strict['not_scored_reason']='At least one class absent after strict screening'
    conditional=[];rng=np.random.default_rng(20261004+int(participant)+1000*['thinking','resting','stimuli','speaking'].index(phase))
    for draw in range(999):
        shuffled=permute_within(rec.y,test,blocks,rng)
        for model,pred in [('covariance',cov_pred),('eegnet_style',nn_preds[17])]:
            conditional.append({'participant':participant,'phase':phase,'model':model,'draw':draw,
                'balanced_accuracy':float(balanced_accuracy_score(shuffled[test],pred))})
    refits=[]
    for draw in range(null_refits):
        shuffled=permute_within(rec.y,train,blocks,rng)
        nc,_=select_covariance(cov,shuffled,folds)
        ne,_=select_neural(rec.x,shuffled,folds)
        for model,pred in [('covariance',covariance_classifier(nc).fit(cov[train],shuffled[train]).predict(cov[test])),
                          ('eegnet_style',neural_predictions(rec.x[train],shuffled[train],rec.x[test],[ne],17)[ne])]:
            refits.append({'participant':participant,'phase':phase,'model':model,'draw':draw,
                'balanced_accuracy':float(balanced_accuracy_score(rec.y[test],pred))})
    identifier=f'{participant}_{phase}'
    pd.DataFrame(predictions).to_csv(large/f'{identifier}_predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(output/f'{identifier}_scores.csv',index=False)
    pd.DataFrame(conditional).to_csv(output/f'{identifier}_conditional_null.csv',index=False)
    pd.DataFrame(refits,columns=['participant','phase','model','draw','balanced_accuracy']).to_csv(output/f'{identifier}_training_null.csv',index=False)
    receipt={'participant':participant,'phase':phase,'selected':selected,'selection':selections,
       'train_ids':rec.trial_ids[train].tolist(),'test_ids':rec.trial_ids[test].tolist(),
       'inner_folds':[{'train_ids':rec.trial_ids[a].tolist(),'validation_ids':rec.trial_ids[b].tolist()} for a,b in folds],
       'strict':strict,'data_audit':rec.audit,'elapsed_seconds':time.perf_counter()-start}
    (output/f'{identifier}_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(f'Finished {identifier}, {len(train)} training/{len(test)} real test, {receipt["elapsed_seconds"]:.1f}s',flush=True)
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--participants',nargs='+',default=[f'{i:02}' for i in range(1,22)])
    p.add_argument('--phases',nargs='+',default=['thinking','resting']);p.add_argument('--null-refits',type=int,default=9)
    p.add_argument('--pilot-diagnostics',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    output=a.output.resolve();large=a.root/'analysis'/output.name;output.mkdir(parents=True,exist_ok=True);large.mkdir(parents=True,exist_ok=True)
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    files=['followup.py','followup_models.py','FOLLOWUP_PROTOCOL.md','data.py','benchmark.py','features.py']
    hashes={}
    for name in files:
        source=Path(__file__).with_name(name); content=source.read_bytes();digest=sha256_file(source)
        previous=snapshot/name
        if previous.exists() and previous.read_bytes()!=content: raise ValueError('Run snapshot differs; use a fresh output name')
        previous.write_bytes(content);hashes[name]=digest
    manifest_path=output/'run_manifest.json'
    if not a.resume and manifest_path.exists(): raise ValueError('Existing run; use --resume or a new output path')
    manifest=json.loads(manifest_path.read_text()) if a.resume and manifest_path.exists() else {
        'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running','source_sha256':hashes,
        'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
        'torch':torch.__version__,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'null_refits':a.null_refits,'null_refits_scope':'participant01 thinking only','protocol_status':'FEIS exploratory; prospective follow-up implementation plan',
        'large_evidence_root':str(large),'completed':[]}
    if manifest['source_sha256']!=hashes or manifest['null_refits']!=a.null_refits: raise ValueError('Cannot resume changed design')
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    jobs=[(participant,phase) for participant in a.participants for phase in a.phases]
    if a.pilot_diagnostics: jobs.extend([('01','stimuli'),('01','speaking')])
    with threadpool_limits(limits=2):
        for participant,phase in jobs:
            identifier=f'{participant}_{phase}'
            if identifier in manifest['completed']: continue
            print('Starting',identifier,flush=True)
            refits=a.null_refits if participant=='01' and phase=='thinking' else 0
            run_one(a.root,participant,phase,output,large,refits)
            manifest['completed'].append(identifier)
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__': main()
