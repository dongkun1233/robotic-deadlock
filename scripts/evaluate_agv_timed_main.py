"""Development-only evaluation after all frozen main searches complete."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from scripts.agv_timed_main_common import *

_STATE = {}

def initialize(path):
    path = Path(path)
    _STATE.update(base=path.parent, manifest=read_manifest(path), sha=hashlib.sha256(path.read_bytes()).hexdigest(),
                  layouts={l: build_layout(l) for l in ['single','double','multi']})


def evaluate_task(layout_name, label, policy, vector):
    cases = [s for s in _STATE['manifest']['scenarios']
             if s['split'] == 'main_development' and s['layout'] == layout_name]
    if len(cases) != 48:
        raise ValueError('Expected 48 development cases/layout')
    path = _STATE['base'] / 'development' / layout_name / (label + '.json')
    spec = dict(manifest_sha256=_STATE['sha'], label=label, layout=layout_name, policy=policy, vector=vector)
    record = json.loads(path.read_text()) if path.exists() else dict(config=spec, rows=[], status='running')
    if record['config'] != spec:
        raise ValueError('Development checkpoint inputs differ')
    require_rows(record['rows'], cases, complete=False)
    for case in cases[len(record['rows']):]:
        record['rows'].append(run_case(_STATE['layouts'][layout_name], case, _STATE['manifest'], policy, vector))
        atomic_durable(path, record)
    require_rows(record['rows'], cases)
    record['status'] = 'complete'
    atomic_durable(path, record)
    return layout_name, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    base = args.manifest.parent
    selection = json.loads((base / 'initial_weights.json').read_text())
    tasks = []
    for layout in ['single','double','multi']:
        chosen = min(selection['selected_by_layout'][layout]['candidates'], key=lambda x:x['objective'])
        tasks.extend([(layout,'fcfs','fcfs',None), (layout,'fixed4_reference','heuristic',list(map(float,FIXED4))),
                      (layout,'strict_fcfs','strict_fcfs',None),
                      (layout,'warmstart_best','heuristic',chosen['vector'])])
        for seed in manifest['optimizer_seeds']:
            for method in manifest['methods']:
                job = base / 'search' / f'{layout}_{seed}_{method}'
                done = json.loads((job / 'complete.json').read_text())
                if not done['complete'] or done['trials'] != 72:
                    raise ValueError('Finish every search before development evaluation')
                fit = json.loads((job / 'best.json').read_text())
                tasks.append((layout, f'{method}_{seed}', 'heuristic', fit['vector']))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                               initargs=(str(args.manifest),)) as executor:
        futures = [executor.submit(evaluate_task,*task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            print(*future.result(),flush=True)
    atomic_durable(base / 'development_complete.json', dict(complete=True, policies_per_layout=19,
        independent_scenarios_per_layout=48, retained_episode_records=2736,
        final_test_evaluated=False, primary_comparison_family=15,
        warmstart_best_scope='descriptive separation of initialization and search effects'))


if __name__ == '__main__':
    main()
