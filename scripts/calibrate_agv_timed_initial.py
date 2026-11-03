"""Evaluate the frozen training-only bank and choose common initial weights."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from scripts.agv_timed_main_common import *

_STATE = {}


def initialize(manifest_path):
    path = Path(manifest_path)
    _STATE.update(manifest=read_manifest(path),
                  manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  output=path.parent,
                  layouts={l: build_layout(l) for l in ['single', 'double', 'multi']})


def candidate_task(layout_name, label, vector, policy):
    manifest = _STATE['manifest']
    cases = training_cases(manifest, layout_name)
    path = _STATE['output'] / 'calibration' / layout_name / (label + '.json')
    spec = dict(manifest_sha256=_STATE['manifest_sha256'], layout=layout_name,
                candidate_id=label, vector=vector, policy=policy)
    record = json.loads(path.read_text()) if path.exists() else dict(config=spec, rows=[], status='running')
    if record['config'] != spec:
        raise ValueError('Calibration cache belongs to different inputs')
    require_rows(record['rows'], cases, complete=False)
    for case in cases[len(record['rows']):]:
        row = run_case(_STATE['layouts'][layout_name], case, manifest, policy, vector)
        record['rows'].append(row)
        atomic_durable(path, record)
    require_rows(record['rows'], cases)
    record['objective'] = objective(record['rows'], manifest['max_time_s'])
    record['status'] = 'complete'
    atomic_durable(path, record)
    return layout_name, label, len(record['rows'])


def summarize(manifest_path):
    path = Path(manifest_path)
    manifest = read_manifest(path)
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    summaries = []
    selected_by_layout = {}
    all_rows = []
    for name in ['single', 'double', 'multi']:
        cases = training_cases(manifest, name)
        evaluated = []
        records = {}
        for label in [c['candidate_id'] for c in manifest['candidate_banks'][name]] + ['_strict_fcfs', '_waiting_equivalence']:
            record = json.loads((path.parent / 'calibration' / name / (label + '.json')).read_text())
            if record['status'] != 'complete' or record['config']['manifest_sha256'] != manifest_sha:
                raise ValueError('Unfinished or mismatched calibration')
            require_rows(record['rows'], cases)
            records[label] = record
            all_rows.extend(record['rows'])
        if any(a['trajectory_sha256'] != b['trajectory_sha256'] for a, b in zip(
                records['fcfs']['rows'], records['_waiting_equivalence']['rows'])):
            raise AssertionError('Pure waiting does not reproduce FCFS')
        for item in manifest['candidate_banks'][name]:
            record = records[item['candidate_id']]
            result = dict(**item, objective=record['objective'])
            evaluated.append(result)
            summaries.append(dict(layout=name, candidate_id=item['candidate_id'], source=item['source'],
                                  objective=result['objective'], incomplete=sum(r['status'] != 'completed' for r in record['rows']),
                                  mean_makespan_s=sum(r['makespan_s'] for r in record['rows']) / len(cases),
                                  **weights(item['vector'])))
        selected, fallback = choose_initial(evaluated)
        anchors = [next(c for c in evaluated if c['candidate_id'] == label) for label in ['fcfs', 'fixed4_reference']]
        selected_by_layout[name] = dict(candidates=anchors + selected, diversity_fallback_used=fallback,
                                       fcfs_equivalence_scenarios=len(cases))
    selection = dict(manifest_sha256=manifest_sha, selected_by_layout=selected_by_layout,
                     selection_split='main_train', development_evaluated=False, final_test_evaluated=False,
                     charged_calibration_calls_per_layout=24, retained_episode_records=len(all_rows),
                     completed_episode_records=sum(r['status'] == 'completed' for r in all_rows),
                     safety_checks_passed=all(r['safety_checks_passed'] for r in all_rows),
                     cpu_seconds_recorded=sum(r['episode_cpu_s'] for r in all_rows))
    atomic_durable(path.parent / 'initial_weights.json', selection)
    save_csv(path.parent / 'calibration_summary.csv', summaries)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('Use 1..8 workers')
    manifest = read_manifest(args.manifest)
    tasks = []
    for name in ['single', 'double', 'multi']:
        for item in manifest['candidate_banks'][name]:
            policy = 'fcfs' if item['candidate_id'] == 'fcfs' else 'heuristic'
            tasks.append((name, item['candidate_id'], item['vector'], policy))
        tasks.extend([(name, '_strict_fcfs', None, 'strict_fcfs'),
                      (name, '_waiting_equivalence', [1., 0., 0., 0.], 'heuristic')])
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            initializer=initialize, initargs=(str(args.manifest),)) as executor:
        futures = [executor.submit(candidate_task, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            print(*future.result(), flush=True)
    selection = summarize(args.manifest)
    print(json.dumps({k: v for k, v in selection.items() if k != 'selected_by_layout'}, indent=2))


if __name__ == '__main__':
    main()
