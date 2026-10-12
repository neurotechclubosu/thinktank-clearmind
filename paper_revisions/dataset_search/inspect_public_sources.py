"""Fetch small public metadata only; never download EEG archives or execute remote code."""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
HEADERS = {'User-Agent': 'ThinkTank-public-dataset-metadata-audit/1.0'}
LIMIT = 4_000_000
TARGETS = [
    ('chisco_tags', 'https://api.github.com/repos/OpenNeuroDatasets/ds005170/tags'),
    ('chisco_description', 'https://api.github.com/repos/OpenNeuroDatasets/ds005170/contents/dataset_description.json'),
    ('chisco_readme', 'https://api.github.com/repos/OpenNeuroDatasets/ds005170/contents/README'),
    ('chisco_tree', 'https://api.github.com/repos/OpenNeuroDatasets/ds005170/git/trees/main?recursive=1'),
    ('inner_speech_description', 'https://api.github.com/repos/OpenNeuroDatasets/ds003626/contents/dataset_description.json?ref=2.1.2'),
    ('inner_speech_tree', 'https://api.github.com/repos/OpenNeuroDatasets/ds003626/git/trees/2.1.2?recursive=1'),
    ('feis_zenodo', 'https://zenodo.org/api/records/3554128'),
    ('vocalmind_zenodo', 'https://zenodo.org/api/records/14696348'),
    ('feis_tree', 'https://api.github.com/repos/scottwellington/FEIS/git/trees/v1.1?recursive=1'),
    ('chisco2_description', 'https://api.github.com/repos/OpenNeuroDatasets/ds006317/contents/dataset_description.json?ref=1.1.1'),
    ('chisco2_readme', 'https://api.github.com/repos/OpenNeuroDatasets/ds006317/contents/README?ref=1.1.1'),
    ('synchronous_description', 'https://api.github.com/repos/OpenNeuroDatasets/ds006033/contents/dataset_description.json?ref=1.0.1'),
    ('synchronous_readme', 'https://api.github.com/repos/OpenNeuroDatasets/ds006033/contents/README?ref=1.0.1'),
    ('synchronous_tree', 'https://api.github.com/repos/OpenNeuroDatasets/ds006033/git/trees/1.0.1?recursive=1'),
]

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=20) as response:
        raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError('Metadata exceeds fixed size limit')
        return raw, response.status, response.geturl()

def main():
    records = []
    for name, url in TARGETS:
        item = {'name': name, 'url': url}
        try:
            raw, status, final = get(url)
            (ROOT / (name + '.json')).write_bytes(raw)
            obj = json.loads(raw)
            item.update(status=status, final_url=final, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            if isinstance(obj, dict) and obj.get('encoding') == 'base64':
                decoded = base64.b64decode(obj['content'])
                (ROOT / (name + '.txt')).write_bytes(decoded)
                item['git_blob_sha'] = obj['sha']
                item['decoded_preview'] = decoded.decode('utf-8')[:3500]
            if isinstance(obj, dict) and 'metadata' in obj:
                item['license'] = obj['metadata'].get('license')
                item['access_right'] = obj['metadata'].get('access_right')
                item['files'] = [{k: f.get(k) for k in ('key', 'size', 'checksum', 'links')} for f in obj.get('files', [])]
            if isinstance(obj, dict) and 'tree' in obj:
                item['tree_sha'] = obj.get('sha')
                item['tree_truncated'] = obj.get('truncated')
                item['entry_count'] = len(obj['tree'])
                item['examples'] = [f for f in obj['tree'] if f['path'].endswith(('_events.tsv', '_eeg.json', 'participants.tsv'))][:8]
            if name.endswith('tags'):
                item['tags'] = [{'name': x['name'], 'sha': x['commit']['sha']} for x in obj]
        except Exception as exc:
            item.update(error=type(exc).__name__ + ': ' + str(exc))
        records.append(item)
        print(name, item.get('status', item.get('error')), flush=True)
    result = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'public metadata; no EEG recordings downloaded', 'records': records}
    (ROOT / 'access_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
