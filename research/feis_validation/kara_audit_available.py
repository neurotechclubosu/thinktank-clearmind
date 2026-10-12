"""Audit currently acquired participants without computing model results."""
import json
from pathlib import Path
from .kara_data import prepare
from .kara_download import ROOT
from .download import PROJECT_ROOT

def main():
    output=PROJECT_ROOT/'paper_revisions/feis_validation/kara_acquisition_audits';output.mkdir(exist_ok=True)
    available=json.loads((ROOT/'download_manifest.json').read_text())['participants']
    results={}
    for participant in available:
        print('Native audit only',participant,flush=True)
        try:
            *_,audit=prepare(ROOT,participant)
            (output/f'{participant}.json').write_text(json.dumps(audit,indent=2))
            results[participant]={'status':'pass','word_trials':sum(audit['word_counts'].values())}
        except (ValueError,KeyError,FileNotFoundError) as error:results[participant]={'status':'fail','reason':str(error)}
        print(participant,results[participant],flush=True)
        (output/'status.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
