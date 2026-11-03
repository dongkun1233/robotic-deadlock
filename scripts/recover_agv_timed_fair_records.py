"""Restore incomplete durable cache prefixes without changing trials or selection."""
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
from scripts.run_agv_timed_fair_pilot import (
    ROOT, source_hashes, initialize, candidate_rows, objective,
    read_manifest, atomic_durable,
)


def main():
    base=ROOT/'outputs/agv_timed_fair_pilot_v1'
    protocol_path=base/'protocol.json'
    p=json.loads(protocol_path.read_text())
    assert source_hashes()==p['source_sha256']
    protocol_sha=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    manifest=read_manifest(ROOT/p['main_manifest'])
    assert hashlib.sha256((ROOT/p['main_manifest']).read_bytes()).hexdigest()==p['main_manifest_sha256']
    cases={split:[s for s in manifest['scenarios'] if s['layout']=='multi' and s['split']=='main_'+split]
           for split in ['train','development']}
    report_path=base/'cache_recovery.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else dict(repairs=[],cost_limit='Original CPU cost of records lost before recovery is unavailable; original and recovered attempts are not both represented in retained CPU totals.')
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,initializer=initialize,
        initargs=(manifest['physical_config'],manifest['dt_s'],manifest['max_time_s'])) as pool:
        for split in cases:
            paths=sorted((base/split/'cache').glob('*.json'),key=lambda path:(json.loads(path.read_text())['config']['rule']!='plain',path.name))
            for path in paths:
                record=json.loads(path.read_text());spec=record['config']
                assert spec['protocol_sha256']==protocol_sha
                if len(record['rows'])==len(cases[split]) and record['status']=='complete':continue
                before=len(record['rows'])
                rows,identity,_=candidate_rows(pool,base,protocol_sha,cases[split],spec['vector'],spec['rule'],spec['threshold_s'],policy=spec['policy'],split=split)
                assert identity==path.stem
                report['repairs'].append(dict(split=split,cache=path.name,retained_before=before,retained_after=len(rows),newly_recovered_scenarios=[r['scenario_id'] for r in rows[before:]]))
                atomic_durable(report_path,report)
                print('RECOVERED',split,path.stem,before,len(rows),flush=True)
    checked=0
    for path in sorted((base/'train').glob('*.json')):
        for trial in json.loads(path.read_text())['trials']:
            record=json.loads((base/'train/cache'/(trial['cache_key']+'.json')).read_text())
            value=objective(record['rows'],manifest['max_time_s'])
            if not math.isclose(value,trial['objective'],rel_tol=0,abs_tol=1e-10):raise AssertionError(('Stored objective differs',path,trial['evaluation'],value,trial['objective']))
            checked+=1
    assert checked==270
    report.update(verified_training_objectives=checked,selection_and_trials_unchanged=True)
    atomic_durable(report_path,report)
    print('VERIFIED',checked,'unchanged objectives',flush=True)

if __name__=='__main__':main()
