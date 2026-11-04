"""Resumable FCFS engineering audit of a v2 draft; never train or run final tests."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from scripts.prepare_agv_capstone_v2 import ROOT, fingerprint, build_layout
from rware.agv_capstone_v2_simulation import run


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-time-s', type=float, default=1200.)
    parser.add_argument('--limit', type=int, help='Total prefix target; omit to audit all cases')
    args = parser.parse_args()
    if not math.isfinite(args.max_time_s) or args.max_time_s <= 0:
        raise ValueError('Positive finite timeout required')
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('protocol') != 'agv_capstone_v2_scenario_draft' or manifest.get('final_test_generated'):
        raise ValueError('Only the v2 train/development draft is permitted')
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source changed: {name}')
    cases = manifest['scenarios']
    ids = [s['scenario_id'] for s in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate scenario IDs')
    for s in cases:
        if s['split'] not in ('train', 'development') or fingerprint(s['missions']) != s['missions_sha256']:
            raise ValueError('Invalid split or mission checksum')
    target = len(cases) if args.limit is None else args.limit
    if not 1 <= target <= len(cases):
        raise ValueError('Invalid prefix target')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'runner.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('An audit already owns this output directory')
        spec = dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    policy='fcfs', dt_s=.01, max_time_s=args.max_time_s)
        checkpoint = args.output_dir/'checkpoint.json'
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(spec=spec, rows=[])
        if state['spec'] != spec:
            raise ValueError('Checkpoint settings/source differ; use a new directory')
        if len(state['rows']) > len(cases):
            raise ValueError('Too many checkpoint records')
        for row, case in zip(state['rows'], cases):
            if row['scenario_id'] != case['scenario_id'] or row['missions_sha256'] != case['missions_sha256']:
                raise ValueError('Checkpoint scenario order differs')
        layouts = {}
        for case in cases[len(state['rows']):target]:
            started = time.perf_counter()
            try:
                name = case['layout']
                if name not in layouts:
                    layouts[name] = build_layout(name)
                result = run(layouts[name], case['missions'], policy='fcfs',
                             dt_s=.01, max_time_s=args.max_time_s)
                if not result['safety_checks_passed']:
                    raise AssertionError('Grid checks failed')
                row = {k: case[k] for k in ('scenario_id', 'missions_sha256', 'layout',
                                            'split', 'robots', 'scenario_case')}
                row.update({k: result[k] for k in ('status', 'completed', 'makespan_s',
                                                   'observed_end_s', 'mean_wait_s',
                                                   'max_wait_s', 'mean_spawn_delay_s')})
                row['episode_wall_s'] = time.perf_counter()-started
                row['trajectory_sha256'] = fingerprint(result['robots_detail'])
                state['rows'].append(row)
                atomic(checkpoint, state)
                durations = [r['episode_wall_s'] for r in state['rows']]
                eta = statistics.mean(durations)*(len(cases)-len(state['rows']))
                print(f"{len(state['rows'])}/{len(cases)} {case['scenario_id']} "
                      f"{row['status']} {row['completed']}/{case['robots']} "
                      f"wall={row['episode_wall_s']:.2f}s rough_remaining={eta/60:.1f}min", flush=True)
            except Exception as exc:
                atomic(args.output_dir/'last_error.json', dict(scenario_id=case['scenario_id'],
                       exception_type=type(exc).__name__, message=str(exc), completed_records=len(state['rows'])))
                raise
        rows = state['rows']
        summary = dict(scope='FCFS engineering audit; not policy ranking or final confirmation',
                       total_cases=len(cases), saved_records=len(rows), complete=len(rows)==len(cases),
                       statuses=dict(Counter(r['status'] for r in rows)),
                       total_measured_wall_s=sum(r['episode_wall_s'] for r in rows),
                       current_timeout_s=args.max_time_s,
                       timeout_is_provisional=True, training_started=False, final_test_started=False,
                       note='Do not choose demand counts using FCFS wait; inspect stalled cases separately from timeout.')
        atomic(args.output_dir/'summary.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
