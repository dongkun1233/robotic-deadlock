"""Restore missing raw evidence without changing saved optimizer observations."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from scripts.agv_timed_main_common import *

BASE = Path('outputs/agv_timed_main_v1')

def repair(task):
    label, gaps = task
    manifest = read_manifest(BASE / 'manifest.json')
    state = json.loads((BASE / 'search' / label / 'checkpoint.json').read_text())
    layout_name = state['config']['layout']
    layout = build_layout(layout_name)
    cases = training_cases(manifest, layout_name)
    for gap in gaps:
        trial = state['trials'][gap['evaluation'] - 1]
        vector = trial['vector']
        path = BASE / 'search' / label / 'cache' / (key(vector) + '.json')
        record = json.loads(path.read_text()) if path.exists() else dict(vector=vector, rows=[], status='running')
        assert record['vector'] == vector
        require_rows(record['rows'], cases, complete=False)
        for case in cases[len(record['rows']):]:
            record['rows'].append(run_case(layout, case, manifest, 'heuristic', vector))
            atomic_durable(path, record)
        value = objective(record['rows'], manifest['max_time_s'])
        if value != trial['objective']:
            raise ValueError(f'{label} evaluation {trial["evaluation"]}: recovered objective {value} != saved {trial["objective"]}')
        assert sum(r['status'] != 'completed' for r in record['rows']) == trial['incomplete']
        record.update(status='complete', objective=value)
        atomic_durable(path, record)
        print('RAW_VERIFIED', label, trial['evaluation'], flush=True)
    return label

def main():
    report = json.loads((BASE / 'recovery_verification_20261104.json').read_text())
    tasks = [(j['job'], j['raw_gaps']) for j in report['jobs'] if j['raw_gaps']]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for label in pool.map(repair, tasks):
            print('RAW_JOB_COMPLETE', label, flush=True)
    atomic_durable(BASE / 'raw_recovery_complete.json', dict(complete=True, repaired_objectives=report['raw_gaps'], optimizer_observations_changed=False, final_test_evaluated=False))

if __name__ == '__main__':
    main()
