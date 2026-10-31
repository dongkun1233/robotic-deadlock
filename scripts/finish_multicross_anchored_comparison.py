"""Wait for the five fixed seed jobs, then audit, validate and run frozen final test."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
workers=[ROOT/f'outputs/anchored_seed_{s}' for s in [101,202,303,404,505]]
started=time.monotonic()
while not all((p/'run_complete.json').exists() for p in workers):
    if time.monotonic()-started>1800:raise TimeoutError('Seed jobs did not finish; inspect their logs')
    time.sleep(2)
def run(script,*args):
    print('Starting',script,flush=True);subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,check=True)
manifest=ROOT/'outputs/fixed_multicross_suite/scenario_manifest.json';training=ROOT/'outputs/multicross_anchored_train_v1';validation=ROOT/'outputs/multicross_anchored_validation_v1';test=ROOT/'outputs/multicross_anchored_test_v1'
run('merge_multicross_anchored_seeds.py','--workers',*workers,'--output-dir',training)
run('audit_multicross_training.py',training)
run('run_multicross_anchored_validation.py','--manifest',manifest,'--training-dir',training,'--output-dir',validation,'--simulator','v2')
va=json.loads((validation/'audit.json').read_text());assert va['episodes']==640 and va['duplicate_keys']==0 and va['replay_verified']
run('evaluate_multicross_anchored_test.py','--manifest',manifest,'--training-dir',training,'--validation-dir',validation,'--output-dir',test)
print('FROZEN COMPARISON FINISHED',flush=True)
