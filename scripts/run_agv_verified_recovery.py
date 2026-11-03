"""Gate raw repair, search continuation and development; never run final test."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.agv_timed_main_common import atomic_durable

BASE = Path('outputs/agv_timed_main_v1')

def main():
    lock = (BASE / 'recovery_pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'}
    phases = [('repair_raw', ['-m','scripts.repair_agv_recovery_raw']),
              ('verify_raw', ['-m','scripts.verify_agv_recovery_snapshot']),
              ('search', ['-m','scripts.run_agv_timed_main','--phase','search','--workers','8']),
              ('development', ['-m','scripts.run_agv_timed_main','--phase','development','--workers','8'])]
    for name, args in phases:
        status = dict(pid=os.getpid(), phase=name, complete=False, final_test_evaluated=False)
        atomic_durable(BASE / 'recovery_pipeline_status.json', status)
        with (BASE / f'recovery_{name}.log').open('a') as log:
            result = subprocess.run([sys.executable,*args], env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(failed=True, exit_code=result.returncode, log=str(BASE / f'recovery_{name}.log'))
            atomic_durable(BASE / 'recovery_pipeline_status.json', status)
            raise SystemExit(result.returncode)
        if name == 'verify_raw':
            report = json.loads((BASE / 'recovery_verification_20261104.json').read_text())
            if report['raw_gaps']:
                raise ValueError('Raw evidence gaps remain; search blocked')
    atomic_durable(BASE / 'recovery_pipeline_status.json', dict(complete=True, phase='development_complete', final_test_evaluated=False))

if __name__ == '__main__':
    main()
