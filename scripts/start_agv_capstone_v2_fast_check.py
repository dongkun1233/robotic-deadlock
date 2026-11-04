"""Codespaces tmux: 432 paired equivalence checks, then isolated fast smoke only."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=args.manifest.resolve()
    data=json.loads(manifest.read_text())
    if not data.get('direction_marginals_balanced') or data.get('final_test_generated'):
        raise SystemExit('Only balanced train/development manifest permitted')
    folder=manifest.parent/'fast_verification_v1'
    report=folder/'equivalence_report.json'
    if args.worker:
        if not report.exists():
            subprocess.run([sys.executable,'-u','-m','scripts.verify_agv_capstone_v2_fast',
                            '--manifest',str(manifest),'--output-dir',str(folder),'--profile','mixed'],
                           cwd=root,check=True)
        subprocess.run([sys.executable,'-u','-m','scripts.train_agv_capstone_v2_fast',
                        '--manifest',str(manifest),'--equivalence-report',str(report),'--mode','smoke'],
                       cwd=root,check=True)
        print('FAST CHECK AND SMOKE FINISHED; no main fit, development ranking, or final test.',flush=True)
        return
    if not shutil.which('tmux'):
        raise SystemExit('tmux is not installed')
    session='agv-v2-fast-check'
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:
        print('Existing session; duplicate not started:',session)
        return
    if folder.exists() and not report.exists():
        raise SystemExit('Prior verification incomplete: inspect fast_check.log/failure.json before retrying')
    log=manifest.parent/'fast_check.log'
    command=shlex.join([sys.executable,'-u','-m','scripts.start_agv_capstone_v2_fast_check',
                       '--manifest',str(manifest),'--worker'])+' >> '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root),command],check=True)
    print('Started:',session)
    print('Log:',log)
    print('Equivalence report:',report)
    print('Runs 432 paired checks then 9 fast smoke jobs. Main training is NOT started.')


if __name__=='__main__':
    main()
