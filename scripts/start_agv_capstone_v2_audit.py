"""Start/resume the newest v2 draft's FCFS audit in tmux; never train."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    if not shutil.which('tmux'):
        raise SystemExit('tmux is not installed')
    candidates = list((root/'outputs').glob('agv_capstone_v2_draft_*/scenario_manifest.json'))
    if not candidates:
        raise SystemExit('No v2 draft manifest found; prepare scenarios first')
    manifest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    data = json.loads(manifest.read_text())
    if data.get('protocol') != 'agv_capstone_v2_scenario_draft' or data.get('final_test_generated'):
        raise SystemExit('Not a permitted train/development draft')
    session = 'agv-v2-fcfs-audit'
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        print('Audit session already exists; no duplicate started.')
        print('Inspect: tmux attach -t '+session)
        return
    output = manifest.parent/'fcfs_audit'
    output.mkdir(exist_ok=True)
    summary = output/'summary.json'
    if summary.exists() and json.loads(summary.read_text()).get('complete'):
        print('Audit already complete:', summary)
        return
    command = shlex.join([sys.executable, '-u', '-m', 'scripts.audit_agv_capstone_v2',
                          '--manifest', str(manifest), '--output-dir', str(output)])
    # Append on resume to preserve earlier failure/measurement logs.
    command += ' >> '+shlex.quote(str(output/'run.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root),command],check=True)
    print('FCFS ENGINEERING AUDIT ONLY; no training/final test.')
    print('Selected manifest:', manifest)
    print('Started tmux session:', session)
    print('Log:', output/'run.log')
    print('Summary on successful exit:', summary)
    print('Restart this same command after interruption; completed episodes are reused.')


if __name__=='__main__':
    main()
