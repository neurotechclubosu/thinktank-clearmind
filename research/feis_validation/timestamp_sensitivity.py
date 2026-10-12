"""Audit-informed sensitivity to decimal-rounded timestamps; retain the primary run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .benchmark import RESULTS, PROTOCOL, score_one, strict_mask
from .data import load_recording, preprocess, chronological_split
from .download import DEFAULT_ROOT, sha256_file
from .features import band_features
from .summarize import mean_interval


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--results',type=Path,default=RESULTS)
    args=parser.parse_args()
    output=args.results.parent/'timestamp_rounding_sensitivity'
    output.mkdir(parents=True,exist_ok=True)
    evidence=args.root/'analysis'/'timestamp_rounding_sensitivity'
    evidence.mkdir(parents=True,exist_ok=True)
    baseline=pd.read_csv(args.results/'scores.csv',dtype={'participant':str})
    original_audits=json.loads((args.results/'data_audit.json').read_text(encoding='utf-8'))
    download_manifest=json.loads((args.root/'download_manifest.json').read_text(encoding='utf-8'))
    notes={'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha256_file(PROTOCOL),
           'primary_run_preserved':str(args.results),'participant':'04','timestamp_atol_seconds':1e-6,
           'motivation':'Source timestamps have six-decimal rounding. Maximum step deviation0.75microseconds and grid deviation0.5microseconds. No samples were missing. This audit-informed sensitivity was proposed before reading primary metrics, but is not a prespecified primary result.',
           'source_code_sha256':{p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},'status':'running'}
    (output/'run_manifest.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    audits=[]; scores=[]
    with threadpool_limits(limits=1), (evidence/'predictions.csv').open('w',encoding='utf-8',newline='') as predictions, (evidence/'donor_receipts.jsonl').open('w',encoding='utf-8') as receipts:
        predictions.write('participant,phase,screening,feature_family,method,seed,trial_id,true_label,predicted_label\n')
        for phase in ['thinking','resting']:
            path=args.root/f'extracted/experiments/04/{phase}/{phase}.csv'
            checksum=sha256_file(path)
            expected=download_manifest['files'][f'experiments/04/{phase}.zip']['extracted'][0]['sha256']
            assert checksum==expected
            rec=load_recording(path,phase,timestamp_atol=1e-6)
            ptp=np.ptp(rec.x,axis=-1)
            rec.x=preprocess(rec.x,rec.fs)
            features=band_features(rec.x,rec.fs)
            train,test=chronological_split(rec)
            original=next(x for x in original_audits if x['participant']=='04' and x['phase']==phase)
            assert rec.trial_ids[test].tolist()==original['test_trial_ids']
            cache=args.root/'processed'/f'04_{phase}_timestamp_rounding_sensitivity.npz'
            np.savez_compressed(cache,x=rec.x,y=rec.y,trial_ids=rec.trial_ids,features=features)
            rec.audit.update(source_sha256=checksum,train_trial_ids=rec.trial_ids[train].tolist(),test_trial_ids=rec.trial_ids[test].tolist())
            audits.append(rec.audit)
            scores.extend(score_one('04',phase,rec,features,train,test,'primary',predictions,receipts))
            mask=strict_mask(ptp,rec.flagged,train)
            strict_train,strict_test=train[mask[train]],test[mask[test]]
            if len(np.unique(rec.y[strict_train]))==16 and len(np.unique(rec.y[strict_test]))==16 and min(np.bincount(rec.y[strict_train],minlength=16))>=2:
                scores.extend(score_one('04',phase,rec,features,strict_train,strict_test,'strict',predictions,receipts))
            print(f'04 {phase}: retained {len(rec.y)} trials; {len(train)} train, same {len(test)} test trials',flush=True)
    new=pd.DataFrame(scores)
    combined=pd.concat([baseline[baseline.participant!='04'],new],ignore_index=True)
    new.to_csv(output/'participant04_scores.csv',index=False)
    combined.to_csv(output/'cohort_scores_with_timestamp_rounding.csv',index=False)
    (output/'data_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    grouped=combined.query("phase=='thinking' and screening=='primary' and feature_family=='multiband'").groupby(['participant','method']).balanced_accuracy.mean().unstack()
    delta=grouped.same_label_splice-grouped.same_label_feature_mixup
    mean,lo,hi=mean_interval(delta)
    notes.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
                 cohort_primary_difference=mean,ci_low=lo,ci_high=hi,
                 provisional_gate_passed=bool(mean>=.02 and lo>0),participants=len(delta),
                 predictions_path=str(evidence/'predictions.csv'),donor_receipts_path=str(evidence/'donor_receipts.jsonl'),
                 predictions_sha256=sha256_file(evidence/'predictions.csv'),donor_receipts_sha256=sha256_file(evidence/'donor_receipts.jsonl'))
    (output/'run_manifest.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print('Cohort comparison after retaining rounded-time trials:',mean,lo,hi,'gate',notes['provisional_gate_passed'],flush=True)


if __name__=='__main__':
    main()
