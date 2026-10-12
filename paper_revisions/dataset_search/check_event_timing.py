"""Audit a small public event sample; never infer or apply a timing correction."""
import base64
import csv
import hashlib
import io
import json
from inspect_public_sources import ROOT, get

tree = json.loads((ROOT / 'synchronous_tree.json').read_text(encoding='utf-8'))['tree']
records = []
prefix = 'sub-02/ses-01/'
for suffix in ('_eeg.vmrk', '_eeg.vhdr', '_bold.json'):
    item = next(x for x in tree if x['path'].startswith(prefix) and x['path'].endswith(suffix))
    record = {'path': item['path'], 'git_blob_sha': item['sha']}
    try:
        raw, status, _ = get(item['url'])
        obj = json.loads(raw)
        body = base64.b64decode(obj['content'])
        record['git_blob_text'] = body.decode('utf-8', errors='replace')
        if item['mode'] == '120000':
            url = 'https://s3.amazonaws.com/openneuro.org/ds006033/' + item['path']
            body, status, _ = get(url)
            record['url'] = url
        else:
            record['url'] = item['url']
        name = 'timing_' + suffix.removeprefix('_') + '.txt'
        (ROOT / name).write_bytes(body)
        record.update(status=status, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), local_file=name)
    except Exception as exc:
        record['error'] = str(exc)
    records.append(record)

summary = {'scope': 'One sampled participant/session, not a complete dataset audit.', 'fetched': records}
for name in ('sync_eeg_events', 'sync_fmri_events'):
    rows = list(csv.DictReader(io.StringIO((ROOT / (name + '.txt')).read_text(encoding='utf-8')), delimiter='\t'))
    onsets = [float(r['onset']) for r in rows]
    durations = [float(r['duration']) for r in rows]
    summary[name] = {'rows': len(rows), 'first_onset_seconds': min(onsets), 'last_onset_seconds': max(onsets), 'duration_values_seconds': sorted(set(durations)), 'last_event_end_seconds': max(o+d for o,d in zip(onsets,durations))}
    short = [(i, r) for i, r in enumerate(rows) if float(r['duration']) < 0.1]
    summary[name]['events_with_duration_below_0_1_seconds'] = len(short)
    summary[name]['short_duration_row_indices_zero_based'] = [i for i, _ in short]
    summary[name]['first_event_with_duration_at_least_0_1_seconds'] = next(({'row_index_zero_based': i, **r} for i, r in enumerate(rows) if float(r['duration']) >= 0.1), None)
summary['eeg_sidecar_recording_duration_seconds'] = json.loads((ROOT / 'sync_eeg_sidecar.txt').read_text(encoding='utf-8'))['RecordingDuration']
summary['interpretation'] = 'In this sampled session, the first 140 of 319 event rows have durations below 0.1 seconds; later rows include protocol-scale 2-second task and approximately 12-second rest durations. The early rows require an explanation against the published protocol and native acquisition/task logs. Do not apply a whole-file timing rescaling. Verify units, trial completeness, stimulus duration and dummy-volume offset before analysis. No correction has been applied.'
(ROOT / 'event_timing_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
for record in records:
    print(record['path'], record.get('status', record.get('error')), record.get('bytes'))
print(json.dumps({k:v for k,v in summary.items() if k != 'fetched'}, indent=2))
