"""Acquire versioned FEIS archives with Git blob checksums and safe extraction."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path('D:/ResearchData/thinktank-clearmind/FEIS/v1.1')
TREE_PATH = PROJECT_ROOT / 'paper_revisions/dataset_search/feis_tree.json'
SOURCE_ROOT = 'https://raw.githubusercontent.com/scottwellington/FEIS/v1.1/'


def git_blob_sha(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(item: dict, root: Path) -> dict:
    relative = PurePosixPath(item['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe source path')
    target = root / 'archives' / Path(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = SOURCE_ROOT + urllib.parse.quote(item['path'], safe='/')
    if not target.exists() or git_blob_sha(target) != item['sha']:
        temporary = target.with_suffix(target.suffix + '.partial')
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'ThinkTank-FEIS-research/1.0'})
                with urllib.request.urlopen(request, timeout=45) as response, temporary.open('wb') as stream:
                    shutil.copyfileobj(response, stream, 1024 * 1024)
                if temporary.stat().st_size != item['size'] or git_blob_sha(temporary) != item['sha']:
                    raise ValueError(f'Size or Git checksum mismatch: {relative}')
                temporary.replace(target)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    extracted = []
    if target.suffix == '.zip':
        destination = (root / 'extracted' / Path(*relative.parts[:-1]) / target.stem).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            if sum(x.file_size for x in archive.infolist()) > 2 * 1024**3:
                raise ValueError('Unexpectedly large extracted archive')
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = PurePosixPath(member.filename.replace('\\', '/'))
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError(f'Unsafe archive member: {name}')
                output = (destination / Path(*name.parts)).resolve()
                if not output.is_relative_to(destination):
                    raise ValueError('Archive member escapes destination')
                if output.exists():
                    if output.stat().st_size != member.file_size:
                        raise ValueError(f'Existing extraction has unexpected size: {output}')
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, output.open('wb') as stream:
                        shutil.copyfileobj(source, stream, 1024 * 1024)
                extracted.append({'path': str(output), 'bytes': output.stat().st_size, 'sha256': sha256_file(output)})
    record = {'source_path': item['path'], 'url': url, 'archive_path': str(target), 'bytes': target.stat().st_size,
              'git_blob_sha': item['sha'], 'sha256': sha256_file(target), 'extracted': extracted,
              'verified_utc': datetime.now(timezone.utc).isoformat()}
    print(f'Verified {relative}: {target.stat().st_size:,} bytes', flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--participants', nargs='+', default=['01'])
    parser.add_argument('--phases', nargs='+', default=['thinking', 'resting', 'full_eeg'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    tree = json.loads(TREE_PATH.read_text(encoding='utf-8'))
    indexed = {item['path']: item for item in tree['tree']}
    paths = ['README.txt', 'experiment-timeline.lua', 'experiment_scenario.mxs']
    for participant in args.participants:
        if participant not in [f'{i:02}' for i in range(1, 22)]:
            raise ValueError('Only the verified English cohort 01..21 is accepted')
        paths.extend(f'experiments/{participant}/{phase}.zip' for phase in args.phases)
    items = [indexed[path] for path in paths]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'download_manifest.json'
    existing = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'dataset': 'FEIS v1.1', 'dataset_url': 'https://zenodo.org/records/3554128', 'license': 'ODC-BY; author README describes nonprofit research use', 'tree_sha': tree['sha'], 'files': {}}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(lambda item: acquire(item, root), items):
            existing['files'][record['source_path']] = record
            manifest_path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
    print(f'Manifest: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
