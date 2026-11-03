"""Resumable 72-call search using frozen training-only common initial weights."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.agv_timed_main_common import *
from scripts.run_multicross_search import search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--layout', choices=['single', 'double', 'multi'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--method', choices=['basic', 'bo', 'pso'], required=True)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    if args.seed not in manifest['optimizer_seeds']:
        raise ValueError('Optimizer seed not predeclared')
    base = args.manifest.parent
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    selection = json.loads((base / 'initial_weights.json').read_text())
    if selection['manifest_sha256'] != manifest_sha or not selection['safety_checks_passed']:
        raise ValueError('Initial bank not validated')
    initial_items = selection['selected_by_layout'][args.layout]['candidates']
    initial = np.array([c['vector'] for c in initial_items])
    if initial.shape != (6, 4):
        raise ValueError('Expected six common initial vectors')
    cases = training_cases(manifest, args.layout)
    layout = build_layout(args.layout)
    job = base / 'search' / f'{args.layout}_{args.seed}_{args.method}'
    job.mkdir(parents=True, exist_ok=True)
    spec = dict(manifest_sha256=manifest_sha, layout=args.layout, seed=args.seed, method=args.method,
                initial_weights_sha256=hashlib.sha256((base / 'initial_weights.json').read_bytes()).hexdigest())
    checkpoint = job / 'checkpoint.json'
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(config=spec, trials=[])
    if state['config'] != spec:
        raise ValueError('Checkpoint belongs to different experiment')
    for item in initial_items:
        record = json.loads((base / 'calibration' / args.layout / (item['candidate_id'] + '.json')).read_text())
        require_rows(record['rows'], cases)
    number = 0
    def evaluate(vector, source):
        nonlocal number
        number += 1
        values = list(map(float, vector))
        if number <= len(state['trials']):
            trial = state['trials'][number - 1]
            if trial['evaluation'] != number or trial['vector'] != values:
                raise ValueError('Search replay differs from saved trajectory')
            return trial['objective']
        matching = next((c for c in initial_items if c['vector'] == values), None)
        if matching is not None:
            record = json.loads((base / 'calibration' / args.layout / (matching['candidate_id'] + '.json')).read_text())
            cached = True
        else:
            path = job / 'cache' / (key(vector) + '.json')
            record = json.loads(path.read_text()) if path.exists() else dict(vector=values, rows=[], status='running')
            if record['vector'] != values:
                raise ValueError('Cache vector mismatch')
            require_rows(record['rows'], cases, complete=False)
            cached = len(record['rows']) == len(cases)
            for case in cases[len(record['rows']):]:
                record['rows'].append(run_case(layout, case, manifest, 'heuristic', vector))
                atomic_durable(path, record)
            record['status'] = 'complete'
            record['objective'] = objective(record['rows'], manifest['max_time_s'])
            atomic_durable(path, record)
        require_rows(record['rows'], cases)
        trial = dict(evaluation=number, source=source, vector=values, objective=record['objective'],
                     incomplete=sum(r['status'] != 'completed' for r in record['rows']), cache_hit=cached)
        state['trials'].append(trial)
        atomic_durable(checkpoint, state)
        print(args.layout, args.seed, args.method, number, trial['objective'], flush=True)
        return trial['objective']
    search(args.method, args.seed, manifest['search_budget'], initial, evaluate)
    if number != 72 or len(state['trials']) != 72:
        raise AssertionError('Budget mismatch')
    best = min(state['trials'], key=lambda t: (t['objective'], t['evaluation']))
    if best['objective'] > min(c['objective'] for c in initial_items) + 1e-9:
        raise AssertionError('Best search result is worse than an initial candidate')
    atomic_durable(job / 'best.json', dict(**best, layout=args.layout, seed=args.seed, method=args.method,
                                         charged_calibration_calls=24, charged_search_calls=72,
                                         total_charged_objectives=96, split='main_train'))
    atomic_durable(job / 'complete.json', dict(complete=True, trials=72, development_evaluated=False,
                                              final_test_evaluated=False))


if __name__ == '__main__':
    main()
