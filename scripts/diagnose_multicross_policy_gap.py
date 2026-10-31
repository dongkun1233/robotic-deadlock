"""Training-only eligibility and optimizer diagnostics; no final-test execution."""
import argparse,csv,hashlib,json,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware import agv_multicross_policy_v2 as sim
from rware.agv_multicross_experiments import digest

def save(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--training-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False);m=json.loads(a.manifest.read_text());scenarios=[s for s in m['scenarios'] if s['split']=='train']
    with (a.training_dir/'best_training_weights.csv').open() as f:best=list(csv.DictReader(f))
    policies=[('fcfs','baseline',None)]+[('bo',r['seed'],json.loads(r['weights_json'])) for r in best if r['algorithm']=='bo']
    rows=[];equivalence=0
    for rule in ('strict_first','feasible_first'):
        for alg,seed,w in policies:
            print(rule,alg,seed,flush=True)
            for s in scenarios:
                kwargs=dict(robots=s['robots'],seed=s['seed'],spawn_window=s['spawn_window'],capacity=m['capacity'],max_steps=m['max_steps'],admission_rule=rule)
                r=sim.run(**kwargs,policy='fcfs' if w is None else 'heuristic',weights=w)
                assert digest([{k:x[k] for k in ('id','start','goal','planned','path')} for x in r['robot_results']])==s['missions_sha256']
                counts=Counter()
                for d in r['decision_log']:
                    counts['attempts']+=1;q=len(d['candidates']);e=len(d['eligible']);counts['queued_competition']+=q>=2;counts['eligible_competition']+=e>=2
                    counts['single_eligible']+=e==1;counts['none_eligible']+=e==0
                    counts['blocked_with_feasible_alternative']+=bool(d['blocked'] and e)
                    if e>=2:
                        pool=[x for x in d['candidates'] if x['robot'] in d['eligible']]
                        counts['eligible_same_score']+=len({round(x['score'],12) for x in pool})==1
                        for feature in sim.FEATURES:counts['eligible_varies_'+feature]+=len({x['raw'][feature] for x in pool})>1
                rows.append({'rule':rule,'algorithm':alg,'optimizer_seed':seed,'scenario_id':s['scenario_id'],'case':s['case'],'total_steps':r['total_steps'],'timed_out':r['timed_out'],**{k:counts[k] for k in ['attempts','queued_competition','eligible_competition','single_eligible','none_eligible','blocked_with_feasible_alternative','eligible_same_score']+['eligible_varies_'+f for f in sim.FEATURES]}})
                if rule=='feasible_first' and alg=='fcfs':
                    pure=sim.run(**kwargs,policy='heuristic',weights=dict(zip(sim.FEATURES,[1,0,0,0])))
                    assert r['robot_results']==pure['robot_results'];equivalence+=1
    save(a.output_dir/'eligibility_training_raw.csv',rows)
    summary=[]
    for rule in ('strict_first','feasible_first'):
        for alg,seed,_ in policies:
            rr=[r for r in rows if (r['rule'],r['algorithm'],r['optimizer_seed'])==(rule,alg,seed)]
            summary.append({'rule':rule,'algorithm':alg,'optimizer_seed':seed,'scenarios':len(rr),'mean_steps':statistics.mean(r['total_steps'] for r in rr),'timeouts':sum(r['timed_out'] for r in rr),**{k:sum(r[k] for r in rr) for k in rr[0] if k.startswith(('eligible_','queued_','blocked_','single_','none_'))}})
    save(a.output_dir/'eligibility_summary.csv',summary)
    state=json.loads((a.training_dir/'checkpoint.json').read_text());search=[]
    for alg in ('basic','bo','pso'):
        for seed in state['config']['seeds']:
            t=sorted([r for r in state['trials'] if r['algorithm']==alg and r['seed']==seed],key=lambda r:r['evaluation'])
            search.append({'algorithm':alg,'seed':seed,'evaluations':len(t),'initial_best':min(r['objective'] for r in t[:6]),'final_best':min(r['objective'] for r in t),'best_evaluation':min(t,key=lambda r:r['objective'])['evaluation'],'distinct_objective_values':len({r['objective'] for r in t}),'distinct_weight_vectors':len({tuple(round(r['weights'][f],10) for f in sim.FEATURES) for r in t})})
    save(a.output_dir/'search_diagnostics.csv',search)
    audit={'training_only':True,'test_evaluated':False,'episodes':len(rows),'pure_waiting_fcfs_identical_episodes':equivalence,'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'note':'Repeated decision attempts are descriptive counts, not independent samples. BO weights selected under v2 are held fixed across rules.'}
    (a.output_dir/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(summary,indent=2));print(audit)
if __name__=='__main__':main()
