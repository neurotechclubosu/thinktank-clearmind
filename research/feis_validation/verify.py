"""Verify acquisition integrity and independently reconstruct scores from predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

import numpy as np
import pandas as pd

from .benchmark import RESULTS
from .download import DEFAULT_ROOT, git_blob_sha, sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--results',type=Path,default=RESULTS)
    args=parser.parse_args()
    acquisition=json.loads((args.root/'download_manifest.json').read_text(encoding='utf-8'))
    csv_count=0
    for entry in acquisition['files'].values():
        archive=Path(entry['archive_path'])
        assert archive.is_relative_to(args.root)
        assert git_blob_sha(archive)==entry['git_blob_sha']
        assert sha256_file(archive)==entry['sha256']
        if archive.suffix=='.zip':
            with zipfile.ZipFile(archive) as z:
                members={Path(x.filename).name:x for x in z.infolist() if not x.is_dir()}
                for item in entry['extracted']:
                    path=Path(item['path'])
                    assert path.is_relative_to(args.root)
                    digest=hashlib.sha256(); crc=0
                    with path.open('rb') as stream:
                        for chunk in iter(lambda:stream.read(1024*1024),b''):
                            digest.update(chunk); crc=zlib.crc32(chunk,crc)
                    assert digest.hexdigest()==item['sha256']
                    assert (crc & 0xffffffff)==members[path.name].CRC
                    csv_count+=1
    manifest=json.loads((args.results/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete'
    for field in ['predictions','donor_receipts']:
        path=Path(manifest[field+'_path'])
        assert path.is_relative_to(args.root)
        assert sha256_file(path)==manifest[field+'_sha256']
    audits=json.loads((args.results/'data_audit.json').read_text(encoding='utf-8'))
    mapped={(a['participant'],a['phase']):a for a in audits}
    truth={(a['participant'],a['phase'],t['trial_id']):t['label'] for a in audits for t in a['trials']}
    predictions=pd.read_csv(manifest['predictions_path'],dtype={'participant':str})
    for row in predictions.itertuples(index=False):
        audit=mapped[row.participant,row.phase]
        assert row.trial_id in audit['test_trial_ids']
        assert row.true_label==truth[row.participant,row.phase,row.trial_id]
    keys=['participant','phase','screening','feature_family','method','seed']
    assert not predictions.duplicated(keys+['trial_id']).any()
    predictions['correct']=predictions.true_label==predictions.predicted_label
    recovered=predictions.groupby(keys+['true_label']).correct.mean().groupby(keys).mean().rename('recovered_balanced_accuracy').reset_index()
    scores=pd.read_csv(args.results/'scores.csv',dtype={'participant':str})
    comparison=scores.merge(recovered,on=keys,validate='one_to_one')
    assert len(comparison)==len(scores)
    max_difference=float(np.max(np.abs(comparison.balanced_accuracy-comparison.recovered_balanced_accuracy)))
    assert max_difference<1e-12
    counts=predictions.groupby(keys).size().rename('recovered_n_test').reset_index()
    counted=scores.merge(counts,on=keys,validate='one_to_one')
    assert np.all(counted.n_real_test==counted.recovered_n_test)
    donor_count=0; budgets=Counter()
    with Path(manifest['donor_receipts_path']).open(encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line)
            audit=mapped[item['participant'],item['phase']]
            parents=set(item['parent_trial_ids'])
            assert parents and parents.issubset(audit['train_trial_ids'])
            assert parents.isdisjoint(audit['test_trial_ids'])
            if item['method']!='random_label_splice':
                assert all(truth[item['participant'],item['phase'],parent]==item['target_label'] for parent in parents)
            donor_count+=1
            budgets[item['participant'],item['phase'],item['screening'],item['method'],item['seed']]+=1
    for row in scores.itertuples(index=False):
        assert budgets[row.participant,row.phase,row.screening,row.method,row.seed]==row.n_synthetic_train
    for name,digest in manifest['source_code_sha256'].items():
        assert sha256_file(args.results/'source_snapshot'/name)==digest
    report={'status':'passed','source_files_verified':len(acquisition['files']),
            'csv_extractions_crc_and_sha256_verified':csv_count,
            'prediction_rows_verified':len(predictions),'score_rows_reconstructed':len(scores),
            'maximum_balanced_accuracy_reconstruction_error':max_difference,
            'donor_receipts_verified':donor_count,'donor_holdout_overlap':0,
            'synthetic_training_budgets_verified':True,'initial_source_snapshot_hashes_verified':True,
            'dataset_logical_bytes_on_D':sum(x.stat().st_size for x in args.root.rglob('*') if x.is_file())}
    (args.results/'integrity_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
