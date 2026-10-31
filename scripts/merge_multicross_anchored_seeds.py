"""Consolidate independently checkpointed seed jobs without rerunning episodes."""
import argparse,csv,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',nargs='+',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/'raw').mkdir()
    states=[json.loads((w/'checkpoint.json').read_text()) for w in a.workers];base=states[0];config={k:v for k,v in base['config'].items() if k!='seeds'};trials=[];seeds=[];best=[]
    for folder,state in zip(a.workers,states):
        assert json.loads((folder/'run_complete.json').read_text())['completed'];assert {k:v for k,v in state['config'].items() if k!='seeds'}==config;assert state['baseline']==base['baseline'];assert len(state['config']['seeds'])==1
        seeds.extend(state['config']['seeds']);trials.extend(state['trials'])
        with (folder/'best_training_weights.csv').open() as f:best.extend(csv.DictReader(f))
        for t in state['trials']:shutil.copyfile(folder/t['raw_file'],a.output_dir/t['raw_file'])
    assert len(seeds)==len(set(seeds));config['seeds']=seeds;state={'config':config,'trials':trials,'baseline':base['baseline']}
    (a.output_dir/'checkpoint.json').write_text(json.dumps(state,indent=2));shutil.copyfile(a.workers[0]/'fcfs_training_raw.csv',a.output_dir/'fcfs_training_raw.csv')
    def save(name,rows):
        with (a.output_dir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save('best_training_weights.csv',best);save('train_trials.csv',[{**{k:v for k,v in t.items() if k!='weights'},'weights_json':json.dumps(t['weights'],sort_keys=True)} for t in trials])
    for seed in seeds:
        for alg in ('basic','bo','pso'):
            ts=[t for t in trials if t['seed']==seed and t['algorithm']==alg];assert len(ts)==72;assert ts[0]['weights']=={'waiting':1.,'maneuver':0.,'exit_competition':0.,'path_conflict':0.};assert ts[0]['objective']==base['baseline']['mean_total_steps'];assert min(t['objective'] for t in ts)<=ts[0]['objective']
    (a.output_dir/'run_complete.json').write_text(json.dumps({'completed':True,'trials':len(trials),'simulation_runs':len(trials)*80,'baseline':base['baseline'],'split':'train','validation_evaluated':False,'test_evaluated':False},indent=2))
    (a.output_dir/'worker_execution.json').write_text(json.dumps({'workers':[str(w) for w in a.workers],'optimizer_seeds':seeds,'actual_baseline_episodes':len(a.workers)*80,'baseline_rows_retained':80,'objective_episodes':len(trials)*80,'anchor_training_bound_verified':True,'timing_note':'Concurrent seed jobs; wall times are descriptive, not a controlled speed comparison.'},indent=2));print('Merged',len(trials),'evaluations')
if __name__=='__main__':main()
