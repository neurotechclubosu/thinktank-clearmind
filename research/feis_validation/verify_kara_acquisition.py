"""Verify all acquired archive/extraction receipts without inferring publisher hashes."""
import json
from pathlib import Path
from .kara_download import ROOT
from .download import PROJECT_ROOT,sha256_file

def main():
    manifest=json.loads((ROOT/'download_manifest.json').read_text());checks=[];extracted_count=0
    if len(manifest['participants'])!=14:raise ValueError('Expected all14 official archives')
    for participant,entry in manifest['participants'].items():
        path=Path(entry['archive_path'])
        if path.stat().st_size!=entry['bytes'] or sha256_file(path)!=entry['sha256']:raise AssertionError('Archive changed')
        for item in entry['extracted']:
            file=Path(item['path'])
            if file.stat().st_size!=item['bytes'] or sha256_file(file)!=item['sha256']:raise AssertionError(f'Extraction changed:{file}')
            extracted_count+=1
        checks.append({'participant':participant,'archive_sha256_verified':True,'listed_extractions_verified':len(entry['extracted'])})
        print('Verified archive/extractions',participant,flush=True)
    additional=0
    for receipt in (ROOT/'delivery_metadata').glob('*.json'):
        data=json.loads(receipt.read_text());participant=data['participant']
        if data['source_archive_sha256']!=manifest['participants'][participant]['sha256']:raise AssertionError('Extra metadata archive provenance differs')
        for item in data['logs']:
            file=Path(item['path'])
            if sha256_file(file)!=item['sha256']:raise AssertionError('Additional delivered log changed')
            additional+=1
    result={'all_checks_pass':True,'archives':14,'archive_bytes':sum(e['bytes'] for e in manifest['participants'].values()),
        'listed_extracted_files_verified':extracted_count,'additional_delivery_metadata_receipts_verified':additional,
        'publisher_authenticated_checksums':False,'checksum_scope':'Stored local download receipts and extracted bytes; source TLS/ETag/size logged at acquisition; no publisher checksum supplied.',
        'checks':checks}
    target=PROJECT_ROOT/'paper_revisions/feis_validation/kara_acquisition_integrity.json'
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
