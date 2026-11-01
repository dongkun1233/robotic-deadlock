"""Build a training-only shared warm-start proposal from existing random-search trials."""
import argparse,csv,hashlib,json,math
from pathlib import Path
FEATURES=('waiting','maneuver','exit_competition','path_conflict')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);cp=a.training_dir/'checkpoint.json';state=json.loads(cp.read_text());assert state['config']['protocol']=='fcfs_anchor_72_v1' and state['config']['split']=='train'
    source=sorted([t for t in state['trials'] if t['algorithm']=='basic'],key=lambda t:(t['objective'],t['seed'],t['evaluation']));assert len(source)==360
    selected=[{'source':'FCFS_anchor','weights':dict(zip(FEATURES,[1.,0.,0.,0.])),'objective':state['baseline']['mean_total_steps'],'seed':None,'evaluation':None}]
    for t in source:
        v=[t['weights'][f] for f in FEATURES];assert abs(sum(x*x for x in v)-1)<1e-10
        if all(sum(v[i]*old['weights'][f] for i,f in enumerate(FEATURES))<.98 for old in selected):
            selected.append({'source':'previous_training_random_search','weights':t['weights'],'objective':t['objective'],'seed':t['seed'],'evaluation':t['evaluation']})
        if len(selected)==6:break
    assert len(selected)==6
    bank={'status':'proposal_for_future_training_not_tested_as_an_initializer','features':FEATURES,'initial_candidates':selected,'selection_rule':'FCFS plus training-objective-ranked random-search vectors, cosine similarity below 0.98 to all already selected vectors','parent_checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'manifest_sha256':state['config']['manifest_sha256'],'simulator_source_sha256':state['config']['simulator_source_sha256'],'prior_random_objective_evaluations':360,'prior_random_episode_evaluations':28800,'additional_simulations':0,'validation_or_test_used_for_selection':False,'fair_comparison_requirement':'All compared optimizers, including random continuation, receive the same bank. Charge the common 360-evaluation construction stage, or clearly report continuation performance conditional on this already available bank. Do not compare standalone random72 vs warm360+BO72 as equal-total-budget.'}
    (a.output_dir/'warm_start_bank.json').write_text(json.dumps(bank,indent=2));rows=[]
    for i,t in enumerate(selected):rows.append({'initial_index':i,'source':t['source'],'training_objective':t['objective'],'source_seed':t['seed'],'source_evaluation':t['evaluation'],**t['weights']})
    with (a.output_dir/'warm_start_bank.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(bank,indent=2))
if __name__=='__main__':main()
