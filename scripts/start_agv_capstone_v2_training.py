"""Explicit-manifest tmux launcher; default is smoke, never automatic main fitting."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--mode',choices=['smoke','main'],default='smoke')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=args.manifest.resolve()
    data=json.loads(manifest.read_text())
    if not data.get('direction_marginals_balanced') or data.get('final_test_generated'):
        raise SystemExit('Balanced train/development manifest required')
    if not shutil.which('tmux'):
        raise SystemExit('tmux is not installed')
    session='agv-v2-training-'+args.mode
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:
        print('Existing session; no duplicate started:',session)
        return
    base=manifest.parent/('training_smoke_12' if args.mode=='smoke' else 'training_main')
    base.mkdir(exist_ok=True)
    summary=base/'status.json'
    if summary.exists():
        state=json.loads(summary.read_text())
        expected=[f'{l}_{s}_{m}' for l in ('single','double','multi')
                  for s in ([101] if args.mode=='smoke' else data['optimizer_seeds'])
                  for m in data['methods']]
        if state.get('complete') and state.get('requested_jobs')==expected:
            print('Requested training already complete:',summary)
            return
    command=shlex.join([sys.executable,'-u','-m','scripts.train_agv_capstone_v2',
                        '--manifest',str(manifest),'--mode',args.mode])
    command+=' >> '+shlex.quote(str(base/'run.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root),command],check=True)
    print('Started:',session)
    print('Mode:',args.mode, '(smoke uses 12 training scenarios/layout, 8 evaluations, seed 101)' if args.mode=='smoke' else '')
    print('Log:',base/'run.log')
    print('Status after clean return:',summary)
    print('No development evaluation or final test is launched.')


if __name__=='__main__':
    main()
