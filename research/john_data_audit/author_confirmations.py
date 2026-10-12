"""Apply the 6 October author clarification to non-destructive selection manifests.

No raw files are changed. Complete means structural coverage/markers, not neural
quality. Duplicate ownership uses timestamps, never classifier performance.
"""
from pathlib import Path
import argparse, ast, collections, csv, datetime, hashlib, json

REPO=Path(__file__).resolve().parents[2]
DATA=Path('D:/ResearchData/thinktank-clearmind/John/2026-10-05')
OUTPUT=REPO/'paper_revisions/john_data_audit'
FRONTIERS='https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2023.1306277/full'
ORDER='Fp1 Fp2 F7 F3 F4 F8 T3 C3 C4 T4 T5 T6 P3 P4 O1 O2'.split()
QUOTE="""1.

1) Slides are the phones, as dialed by presentation. 2. Yes and yes. 3. You can see the order in text (the order I name them specifically in the paragraph in frontiers paper for FT). 4. Those refer to separate sessions (entire trials for all phonemes). Remove those glitch files, and you can simply stick to complete runs. ONe run per person is enough data on its own. mapo.txt was  a dummy file I made (useless). 5. No full words. No separate eye or muscle actions.

1. What you see is what you get"""

def readcsv(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def writecsv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def old_order():
    tree=ast.parse((REPO/'paper_revisions/round2_revision/code/sphere.py').read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHANNELS' for t in n.targets))

def main(data,output):
    output.mkdir(parents=True,exist_ok=True)
    records=readcsv(data/'audit/recordings.csv');events=readcsv(data/'audit/events.csv')
    byfile={r['file']:r for r in records}
    pairs=[('DLR_27_2.txt','DLR_27_3.txt'),('GTP003_2_2.txt','GTP003_2_3.txt')]
    decisions=[];excluded={}
    for n in ['GTP003_18_3.txt','GTP004_11_2.txt']:
        assert int(byfile[n]['paired_events'])==0
        excluded[n]='Truncated file: no paired recalls; author authorized glitch exclusion.'
    for pair in pairs:
        hashes=[{e['marker']:e['full_task_eeg_sha256'] for e in events if e['file']==n} for n in pair]
        assert hashes[0]==hashes[1] and len(hashes[0])==5
        assert byfile[pair[0]]['timestamp_start']==byfile[pair[1]]['timestamp_start']
        owners=[];ranges=[]
        for n in pair:
            row=byfile[n]
            peers=[r for r in records if r['subject']==row['subject'] and r['block']==row['block'] and r['file'] not in pair and int(r['paired_events'])==5]
            lo=min(float(r['timestamp_start']) for r in peers);hi=max(float(r['timestamp_end']) for r in peers)
            inside=lo<=float(row['timestamp_start'])<=float(row['timestamp_end'])<=hi
            ranges.append(dict(file=n,peer_start=lo,peer_end=hi,duplicate_within_peer_range=inside))
            if inside:owners.append(n)
        assert len(owners)==1, 'Duplicate session ownership ambiguous; do not choose arbitrary copy.'
        keep=owners[0];discard=next(n for n in pair if n!=keep)
        excluded[discard]=f'Duplicate task data/timestamps from {keep}; outside its own session peer time range.'
        decisions.append(dict(pair=list(pair),keep=keep,exclude=discard,evidence=ranges,inference='Timestamp-consistent ownership; no task scores inspected.'))
    sessions=[]
    for subject in sorted(set(r['subject'] for r in records)):
        for block in sorted({int(r['block']) for r in records if r['subject']==subject}):
            rr=[r for r in records if r['subject']==subject and int(r['block'])==block]
            bad=[r['file'] for r in rr if r['file'] in excluded]
            indices={int(r['phoneme_index']) for r in rr if r['file'] not in excluded and int(r['paired_events'])==5}
            complete=len(rr)==44 and indices==set(range(44)) and not bad
            sessions.append(dict(subject=subject,session_suffix=block,recording_files=len(rr),usable_indices=len(indices),complete=complete,excluded_files=';'.join(bad),reason='Complete index/marker coverage; raw quality not certified' if complete else 'Incomplete after glitch or copied-recording exclusions'))
    selected={(s['subject'],str(s['session_suffix'])) for s in sessions if s['complete']}
    chosen=[r for r in records if (r['subject'],r['block']) in selected]
    chosen_names={r['file'] for r in chosen};chosen_events=[e for e in events if e['file'] in chosen_names]
    assert all(int(r['paired_events'])==5 for r in chosen)
    assert len(chosen)==572 and len(chosen_events)==2860
    assert len({e['full_task_eeg_sha256'] for e in chosen_events})==len(chosen_events)
    subset={(sub,min(int(b) for s,b in selected if s==sub)) for sub in sorted({s for s,b in selected})}
    one=[r for r in chosen if (r['subject'],int(r['block'])) in subset]
    assert len(one)==176
    prior=old_order()
    channel_rows=[dict(python_column=i+1,electrode=name,previous_sphere_label=prior[i],changed=name!=prior[i],source='Frontiers section2.4; author says paragraph order is recording order') for i,name in enumerate(ORDER)]
    writecsv(output/'confirmed_channel_order.csv',channel_rows)
    writecsv(output/'session_eligibility.csv',sessions)
    writecsv(output/'complete_session_recordings.csv',chosen)
    writecsv(output/'one_session_per_person_recordings.csv',one)
    excluded_rows=[dict(file=n,scope='file',reason=why) for n,why in excluded.items()]
    excluded_rows.append(dict(file='mapo.txt',scope='file',reason='Author identified as useless dummy file; excluded.'))
    for s in sessions:
        if not s['complete']:excluded_rows.append(dict(file=f"{s['subject']} session {s['session_suffix']}",scope='session',reason='Whole-session analysis excludes incomplete run; originals preserved.'))
    writecsv(output/'analysis_exclusions.csv',excluded_rows)
    dump(output/'duplicate_session_ownership.json',decisions)
    metadata=dict(clarification_received_local='2026-10-06',source='User relayed John response in this chat; author confirmation, not independently inspected acquisition log',verbatim_response=QUOTE,confirmed=dict(recall_seconds=2,wait_seconds=1,matching_markers='Beginning and end of each instructed recall',filename_suffix='Separate acquisition sessions covering all phonemes',electrode_order=ORDER,electrode_order_source=FRONTIERS,mapo='Dummy, excluded',full_word_recordings_available=False,separate_eye_muscle_recordings_available=False),stimulus_mapping=dict(status='Author confirms presentation defines phone order; recovered phoneme deck is operational candidate; supplied16slide deck incompatibility remains documented',full_candidate_table='stimulus_index_map.csv',exact_supplied_deck_equivalence_confirmed=False,arpabet_dialect_crosswalk_confirmed=False),not_confirmed=dict(physical_units=None,amplifier_gain=None,physical_reference_ground=None,delivered_files_raw_or_preprocessed=None,cue_only_control_availability=None,continuous_sentence_availability='No such recordings supplied; full words absent per author',subset_demographics=None,secondary_use_and_public_release_permissions=None),prior_source_order=prior,changed_channel_labels=sum(ORDER[i]!=prior[i] for i in range(16)),source_space_status='v1.27 used differing channel assignment; do not treat existing spatial outputs as confirmed. Regenerate with new order and resolve reference/model assumptions.',one_run_sufficiency='Author opinion; independent validation sufficiency not established by file counts.',complete_session_ids={sub:[s['session_suffix'] for s in sessions if s['subject']==sub and s['complete']] for sub in sorted(set(r['subject'] for r in records))},complete_sessions=len(selected),selected_files=len(chosen),selected_marked_intervals=len(chosen_events),one_session_subset_files=len(one),one_session_subset_intervals=880,selection_policy='All structurally complete sessions retained; earliest complete session per person offered as descriptive subset only. No models fit; no train/test roles assigned.',verification='PASS:44indices/fivepairs per selected file; all2860 task hashes distinct; duplicate timestamp ownership unique',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(output/'author_metadata_2026-10-06.json',metadata)
    print(json.dumps({k:metadata[k] for k in ['complete_session_ids','complete_sessions','selected_files','selected_marked_intervals','changed_channel_labels','verification']},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=DATA);ap.add_argument('--output',type=Path,default=OUTPUT);a=ap.parse_args();main(a.data,a.output)
