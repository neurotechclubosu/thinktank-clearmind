"""Independent Kara four-word evaluation following an acquisition-only pilot."""
from __future__ import annotations
import argparse,json,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from .kara_download import ROOT,PARTICIPANTS
from .kara_data import prepare,class_inner_folds,fit_eye_regression
from .download import PROJECT_ROOT,sha256_file
from .followup_models import covariance_features,covariance_classifier,select_covariance,select_neural,neural_predictions,SEEDS
from .summarize_followup import bootstrap,holm

OUTPUT=PROJECT_ROOT/'paper_revisions/feis_validation/results/kara_followup_v1'

def scaled_covariance_features(x):
    # CNT reader emits volts; microvolt numeric scale keeps fixed machine floors
    # inactive so the declared10% trace shrinkage governs regularization.
    return covariance_features(np.asarray(x,dtype=np.float64)*1e6)

def strict_mask(ptp,train):
    logs=np.log(ptp.clip(1e-30));center=np.median(logs[train],axis=0)
    spread=np.maximum(1.4826*np.median(np.abs(logs[train]-center),axis=0),.1)
    return ~((np.sum(logs>center+6*spread,axis=1)>3)|(np.sum(ptp<=1e-15,axis=1)>=2))

def evaluate(participant,data,output,large):
    x,y,ids,ptp,eye,train,test,folds,audit=data
    cov=scaled_covariance_features(x);c,c_cv=select_covariance(cov,y,folds);epochs,nn_cv=select_neural(x,y,folds)
    predictions=[];scores=[];conditional=[];chosen={'covariance':c,'eegnet_style':epochs}
    def record(model,pred,seed=17,screening='primary',fit=train,held=test):
        scores.append({'participant':participant,'model':model,'seed':seed,'screening':screening,
                       'n_train':len(fit),'n_test':len(held),'balanced_accuracy':float(balanced_accuracy_score(y[held],pred))})
        for pos,label in zip(held,pred):predictions.append({'participant':participant,'model':model,'seed':seed,
            'screening':screening,'trial_id':int(ids[pos]),'true_label':int(y[pos]),'predicted_label':int(label)})
        return pred
    cov_pred=record('covariance',covariance_classifier(c).fit(cov[train],y[train]).predict(cov[test]))
    nn={}
    for seed in SEEDS:nn[seed]=record('eegnet_style',neural_predictions(x[train],y[train],x[test],[epochs],seed)[epochs],seed)
    eye_features=scaled_covariance_features(eye);ec,ecv=select_covariance(eye_features,y,folds)
    record('eye_only',covariance_classifier(ec).fit(eye_features[train],y[train]).predict(eye_features[test]))
    nuisance=np.log(ptp.clip(1e-30));nc,ncv=select_covariance(nuisance,y,folds)
    record('amplitude_nuisance',covariance_classifier(nc).fit(nuisance[train],y[train]).predict(nuisance[test]))
    corrected,coefficients=fit_eye_regression(x,eye,train);ccov=scaled_covariance_features(corrected)
    record('covariance',covariance_classifier(c).fit(ccov[train],y[train]).predict(ccov[test]),screening='eye_regression')
    record('eegnet_style',neural_predictions(corrected[train],y[train],corrected[test],[epochs],17)[epochs],screening='eye_regression')
    keep=strict_mask(ptp,train);good_train=train[keep[train]];good_test=test[keep[test]]
    audit['strict']={'train_ids':ids[good_train].tolist(),'test_ids':ids[good_test].tolist(),'excluded_ids':ids[~keep].tolist()}
    if len(np.unique(y[good_train]))==4 and len(np.unique(y[good_test]))==4:
        if keep.all():
            record('covariance',cov_pred,screening='strict');record('eegnet_style',nn[17],screening='strict')
        else:
            record('covariance',covariance_classifier(c).fit(cov[good_train],y[good_train]).predict(cov[good_test]),screening='strict',fit=good_train,held=good_test)
            record('eegnet_style',neural_predictions(x[good_train],y[good_train],x[good_test],[epochs],17)[epochs],screening='strict',fit=good_train,held=good_test)
    else:audit['strict']['not_scored_reason']='Class coverage lost'
    cut=int(np.floor(2*len(y)/3));gtrain=np.arange(cut);gtest=np.arange(cut,len(y))
    audit['global_chronology']={'train_ids':ids[gtrain].tolist(),'test_ids':ids[gtest].tolist()}
    if len(np.unique(y[gtrain]))==4 and len(np.unique(y[gtest]))==4 and min(np.bincount(y[gtrain],minlength=4))>=3:
        gfolds=class_inner_folds(y,ids,gtrain);gc,gcv=select_covariance(cov,y,gfolds);ge,gev=select_neural(x,y,gfolds)
        record('covariance',covariance_classifier(gc).fit(cov[gtrain],y[gtrain]).predict(cov[gtest]),screening='global_chronology',fit=gtrain,held=gtest)
        record('eegnet_style',neural_predictions(x[gtrain],y[gtrain],x[gtest],[ge],17)[ge],screening='global_chronology',fit=gtrain,held=gtest)
        audit['global_chronology'].update(selected={'C':gc,'epochs':ge},selection={'covariance':gcv,'eegnet_style':gev},
            inner_folds=[{'train_ids':ids[a].tolist(),'validation_ids':ids[b].tolist()} for a,b in gfolds])
    else:audit['global_chronology']['not_scored_reason']='Insufficient chronological class coverage'
    rng=np.random.default_rng(20261004+PARTICIPANTS.index(participant))
    for draw in range(999):
        shuffled=rng.permutation(y[test])
        for model,pred in [('covariance',cov_pred),('eegnet_style',nn[17])]:conditional.append({'participant':participant,'model':model,'draw':draw,
            'balanced_accuracy':float(balanced_accuracy_score(shuffled,pred))})
    audit.update(selected=chosen,selection={'covariance':c_cv,'eegnet_style':nn_cv,'eye_only':ecv,'amplitude_nuisance':ncv},
                 eye_regression_coefficients_sha256=None)
    np.savez_compressed(large/f'{participant}_eye_regression.npz',coefficients=coefficients)
    audit['eye_regression_coefficients_sha256']=sha256_file(large/f'{participant}_eye_regression.npz')
    pd.DataFrame(predictions).to_csv(large/f'{participant}_predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(output/f'{participant}_scores.csv',index=False)
    pd.DataFrame(conditional).to_csv(output/f'{participant}_conditional_null.csv',index=False)
    (output/f'{participant}_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Finished independent participant',participant,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True);large=a.root/'analysis'/output.name;large.mkdir(parents=True,exist_ok=True)
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    names=['kara_benchmark.py','kara_data.py','followup_models.py','FOLLOWUP_PROTOCOL.md','KARA_PARSER_PLAN.md']
    for name in names:
        src=Path(__file__).with_name(name);target=snapshot/name
        if target.exists() and target.read_bytes()!=src.read_bytes():raise ValueError('Frozen parser/model differs; choose fresh run')
        target.write_bytes(src.read_bytes())
    manifest_path=output/'run_manifest.json'
    if manifest_path.exists() and not a.audit_only:raise ValueError('Do not overwrite an existing independent evaluation')
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'audit_running',
        'participants':[p for p in PARTICIPANTS if p!='MM05'],'excluded_format_pilot':'MM05',
        'source_sha256':{n:sha256_file(snapshot/n) for n in names},'python':platform.python_version(),
        'numpy':np.__version__,'torch':torch.__version__,'large_evidence_root':str(large),'completed':[]}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8');datasets={};failures={}
    with threadpool_limits(limits=2):
        for participant in manifest['participants']:
            print('Auditing independent participant before scores',participant,flush=True)
            try:
                data=prepare(a.root,participant);datasets[participant]=data
                x,y,ids,ptp,eye,_,_,_,audit=data
                target=a.root/'processed'/f'{participant}_four_word.npz';target.parent.mkdir(exist_ok=True)
                np.savez_compressed(target,x=x,y=y,ids=ids,ptp=ptp,eye=eye)
                audit['cache_sha256']=sha256_file(target)
                (output/f'{participant}_native_audit.json').write_text(json.dumps(audit,indent=2))
            except (ValueError,KeyError,FileNotFoundError) as error:failures[participant]=str(error)
        manifest['acquisition_audit_failures']=failures
        if failures or a.audit_only:
            manifest['status']='audit_failed_no_classification' if failures else 'audit_only_complete'
            manifest_path.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2));return
        manifest['status']='classification_running';manifest_path.write_text(json.dumps(manifest,indent=2))
        for participant,data in datasets.items():
            evaluate(participant,data,output,large);manifest['completed'].append(participant)
            manifest_path.write_text(json.dumps(manifest,indent=2))
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
