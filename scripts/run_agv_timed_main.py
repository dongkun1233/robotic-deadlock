"""Run one explicitly named phase of the frozen physical-time experiment."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.agv_timed_main_common import ROOT, atomic_durable, read_manifest


def search_worker(task):
    manifest_path, layout, seed, method = task
    base = Path(manifest_path).parent
    label = f'{layout}_{seed}_{method}'
    log = base / 'logs' / (label + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        result = subprocess.run([sys.executable, '-m', 'scripts.train_agv_timed_main',
            '--manifest', manifest_path, '--layout', layout, '--seed', str(seed), '--method', method],
            cwd=ROOT, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
            stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{label} failed; inspect {log}')
    complete = json.loads((base / 'search' / label / 'complete.json').read_text())
    if not complete['complete'] or complete['trials'] != 72:
        raise ValueError('Job returned without complete search records')
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/agv_timed_main_v1/manifest.json'))
    parser.add_argument('--phase', choices=['calibrate', 'search', 'development'], required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('Use 1..8 workers')
    manifest = read_manifest(args.manifest)
    lock = (args.manifest.parent / 'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('Another main-phase runner is active') from error
    if args.phase in ('calibrate', 'development'):
        module = 'scripts.calibrate_agv_timed_initial' if args.phase == 'calibrate' else 'scripts.evaluate_agv_timed_main'
        subprocess.run([sys.executable, '-m', module, '--manifest', str(args.manifest),
                        '--workers', str(args.workers)], cwd=ROOT,
                       env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}, check=True)
        return
    selection = json.loads((args.manifest.parent / 'initial_weights.json').read_text())
    if not selection['safety_checks_passed']:
        raise ValueError('Initial calibration not validated')
    tasks = [(str(args.manifest), layout, seed, method)
             for seed in manifest['optimizer_seeds']
             for layout in ['single', 'double', 'multi'] for method in manifest['methods']]
    status = dict(phase='search', complete=False, expected_jobs=len(tasks), completed_jobs=[], final_test_evaluated=False)
    path = args.manifest.parent / 'search_status.json'
    atomic_durable(path, status)
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(search_worker, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                label = future.result()
                status['completed_jobs'].append(label)
                print('COMPLETE', label, len(status['completed_jobs']), '/', len(tasks), flush=True)
            except Exception as error:
                failures.append(str(error))
            status['failures'] = failures
            atomic_durable(path, status)
    status['complete'] = len(status['completed_jobs']) == len(tasks) and not failures
    atomic_durable(path, status)
    if failures:
        raise RuntimeError('; '.join(failures))


if __name__ == '__main__':
    main()
