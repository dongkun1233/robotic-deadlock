"""Audited CPU-portable continuation; frozen research sources are unchanged."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/agv_timed_main_v1'
ATOL = 8 * 2.220446049250313e-16

def checked_vector(trial, vector, source, index):
    import numpy as np
    actual = np.asarray(vector, dtype=float)
    saved = np.asarray(trial['vector'], dtype=float)
    if actual.shape != (4,) or saved.shape != (4,) or not np.isfinite(actual).all():
        raise ValueError('Invalid replay vector')
    error = float(np.max(np.abs(actual - saved)))
    if trial['evaluation'] != index + 1 or trial['source'] != source or error > ATOL:
        raise ValueError(f'Replay rejected at {index+1}: error={error}, source={source}')
    return error

def replay_adapter(original, state, label, records):
    def adapted(method, seed, budget, initial, evaluate):
        count = 0
        differences = []
        def guarded(vector, source):
            nonlocal count
            index = count
            count += 1
            if index < len(state['trials']):
                trial = state['trials'][index]
                error = checked_vector(trial, vector, source, index)
                if error:
                    differences.append(dict(evaluation=index+1, max_abs_error=error))
                # Only historical evaluator inputs are canonicalized. Optimizer
                # arithmetic and every newly generated candidate remain native.
                import numpy as np
                vector = np.array(trial['vector'], dtype=float)
            return evaluate(vector, source)
        try:
            return original(method, seed, budget, initial, guarded)
        finally:
            records.append(dict(job=label, saved_evaluations=len(state['trials']),
                                differences=differences, atol=ATOL))
    return adapted

def write(path, data):
    from scripts.agv_timed_main_common import atomic_durable
    atomic_durable(path, data)

def verify():
    import scripts.run_multicross_search as module
    original = module.search
    paths = iter(sorted((BASE/'search').glob('*/checkpoint.json')))
    records = []
    def dispatch(*args):
        path = next(paths)
        state = json.loads(path.read_text())
        return replay_adapter(original, state, path.parent.name, records)(*args)
    module.search = dispatch
    try:
        runpy.run_module('scripts.verify_agv_recovery_snapshot', run_name='__main__')
    finally:
        module.search = original
    report_path = BASE/'recovery_verification_20261104.json'
    report = json.loads(report_path.read_text())
    report['search_paths_exact'] = not any(r['differences'] for r in records)
    report['portable_replay_validated'] = True
    report['replay_atol'] = ATOL
    report['comparison'] = 'absolute tolerance only; evaluation index and source exact'
    write(report_path, report)
    write(BASE/'portable_replay_audit.json', dict(records=records, report=report,
          note='Historical evaluator inputs use saved vectors; native optimizer state may differ in final bits. Not bitwise reproducibility.'))
    return report

def train(layout, seed, method):
    import scripts.train_agv_timed_main as module
    path = BASE/'search'/f'{layout}_{seed}_{method}'/'checkpoint.json'
    state = json.loads(path.read_text()) if path.exists() else {'trials':[]}
    before = json.loads(json.dumps(state.get('trials', [])))
    records = []
    module.search = replay_adapter(module.search, state, path.parent.name, records)
    sys.argv = ['train', '--manifest', str(BASE/'manifest.json'), '--layout',layout,
                '--seed',str(seed),'--method',method]
    try:
        module.main()
    finally:
        after = json.loads(path.read_text())['trials'] if path.exists() else []
        if after[:len(before)] != before:
            raise ValueError('Historical checkpoint prefix changed')
        write(BASE/'portable_audits'/f'{layout}_{seed}_{method}.json',
              dict(records=records, historical_prefix_preserved=True,
                   original_prefix_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()))

def run_job(task):
    layout, seed, method = task
    label = f'{layout}_{seed}_{method}'
    with (BASE/'logs'/f'portable_{label}.log').open('a') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--train',layout,str(seed),method],
                       stdout=log, stderr=subprocess.STDOUT,check=True)
    complete = json.loads((BASE/'search'/label/'complete.json').read_text())
    if not complete['complete'] or complete['trials'] != 72:
        raise ValueError(f'Incomplete job: {label}')
    return label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--train',nargs=3)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if args.train:
        train(args.train[0],int(args.train[1]),args.train[2]); return
    if not 1 <= args.workers <= 8:
        raise ValueError('workers must be 1..8')
    locks = []
    for name in ['recovery_pipeline.lock','runner.lock']:
        handle = (BASE/name).open('a')
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks.append(handle)
    from scripts.agv_timed_main_common import read_manifest
    manifest = read_manifest(BASE/'manifest.json')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = BASE/'portable_backups'/stamp
    backup.mkdir(parents=True)
    for path in (BASE/'search').glob('*/checkpoint.json'):
        shutil.copy2(path,backup/(path.parent.name+'.json'))
    write(backup/'provenance.json',dict(python=sys.version,atol=ATOL,
        platform=sys.platform,note='CPU-portable continuation, not bitwise identical to original machine'))
    report = verify()
    if args.verify_only:
        print('PORTABLE_VERIFY_OK',report['observations'],'raw gaps',report['raw_gaps'],flush=True)
        return
    def phase(name, **extra):
        write(BASE/'portable_pipeline_status.json',dict(phase=name,complete=False,
             final_test_evaluated=False,**extra))
    phase('repair_raw')
    subprocess.run([sys.executable,'-m','scripts.repair_agv_recovery_raw'],check=True)
    report = verify()
    if report['raw_gaps']:
        raise ValueError('Raw evidence gaps remain')
    phase('search')
    tasks = [(l,s,m) for s in manifest['optimizer_seeds']
             for l in ['single','double','multi'] for m in manifest['methods']]
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
            phase('search',completed_jobs=completed,expected_jobs=45)
            print('COMPLETE',completed[-1],len(completed),'/ 45',flush=True)
    report = verify()
    if report['observations'] != 3240 or report['raw_gaps']:
        raise ValueError('Final search verification failed')
    phase('development')
    subprocess.run([sys.executable,'-m','scripts.evaluate_agv_timed_main',
                   '--manifest',str(BASE/'manifest.json'),'--workers',str(args.workers)],check=True)
    write(BASE/'portable_pipeline_status.json',dict(phase='development_complete',complete=True,
          final_test_evaluated=False,search_observations=3240))

if __name__ == '__main__':
    main()
