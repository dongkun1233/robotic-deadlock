"""Frozen final test for all anchored full-model methods, with all seeds retained."""
import argparse,csv,hashlib,json,statistics,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware import agv_multicross_policy_v2 as sim
from rware.agv_multicross_experiments import digest
from rware import agv_multicross_layout, agv_multicross_policy_simulation

def save(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(rows,scenarios,seeds,draws=10000,permutations=100000):
    algorithms=['basic','bo','pso'];ids=[s['scenario_id'] for s in scenarios];case=np.array([s['case'] for s in scenarios]);lookup={(r['algorithm'],str(r['optimizer_seed']),r['scenario_id']):r for r in rows};base=np.array([lookup[('fcfs','baseline',sid)]['total_steps'] for sid in ids]);arrays={alg:np.array([[lookup[(alg,str(seed),sid)]['total_steps'] for sid in ids] for seed in seeds]) for alg in algorithms};rng=np.random.default_rng(20261031);scenario_draw=np.concatenate([rng.choice(np.where(case==c)[0],size=(draws,sum(case==c)),replace=True) for c in sorted(set(case))],axis=1);seed_draw=rng.integers(0,len(seeds),size=(draws,len(seeds)));signs=rng.choice([-1.,1.],size=(permutations,len(ids)));contrasts=[]
    for alg,other in [('basic','fcfs'),('bo','fcfs'),('pso','fcfs'),('bo','basic'),('pso','basic'),('bo','pso')]:
        matrix=arrays[alg]-(base[None,:] if other=='fcfs' else arrays[other]);d=matrix.mean(axis=0);conditional=d[scenario_draw].mean(axis=1);joint=matrix[seed_draw[:,:,None],scenario_draw[:,None,:]].mean(axis=(1,2));lo,hi=np.percentile(conditional,[2.5,97.5]);jl,jh=np.percentile(joint,[2.5,97.5]);p=(1+np.sum(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-12))/(permutations+1)
        contrasts.append({'comparison':alg+'_minus_'+other,'primary':other=='fcfs','mean_step_difference':float(d.mean()),'scenario95_lower':float(lo),'scenario95_upper':float(hi),'two_way95_lower':float(jl),'two_way95_upper':float(jh),'permutation_p_conditional':float(p),'holm_primary_p':'','unique_scenarios':len(ids),'optimizer_seeds':len(seeds),'caveat':'Approximate intervals; conditional permutation assumes sign exchangeability. No equivalence or universal algorithm claim.'})
    order=sorted(range(3),key=lambda i:contrasts[i]['permutation_p_conditional']);running=0
    for rank,i in enumerate(order):running=max(running,min(1.,(3-rank)*contrasts[i]['permutation_p_conditional']));contrasts[i]['holm_primary_p']=running
    return contrasts

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','training-dir','validation-dir','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);m=json.loads(a.manifest.read_text());state=json.loads((a.training_dir/'checkpoint.json').read_text());config=state['config'];audit=json.loads((a.training_dir/'audit.json').read_text());va=json.loads((a.validation_dir/'audit.json').read_text());assert audit['passed'] and va['replay_verified'] and va['duplicate_keys']==0
    assert config['protocol']=='fcfs_anchor_72_v1' and config['budget']==72 and config['seeds']==[101,202,303,404,505] and config['models']==['full'];assert config['manifest_sha256']==hashlib.sha256(a.manifest.read_bytes()).hexdigest();assert config['simulator_source_sha256']==hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest()
    for mod in (agv_multicross_layout,agv_multicross_policy_simulation):assert hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()==config['source_sha256'][Path(mod.__file__).name]
    with (a.training_dir/'best_training_weights.csv').open() as f:best=list(csv.DictReader(f))
    assert len(best)==15;policies=[('fcfs','baseline',None)]
    for b in best:
        t=min([t for t in state['trials'] if t['algorithm']==b['algorithm'] and str(t['seed'])==b['seed']],key=lambda t:(t['objective'],t['evaluation']));assert int(b['evaluation'])==t['evaluation'] and json.loads(b['weights_json'])==t['weights'];policies.append((b['algorithm'],b['seed'],t['weights']))
    lock={'protocol_commit':'30dce7e60b0360bbbb59388b1b1597f52f8943f6','manifest_sha256':config['manifest_sha256'],'simulator_source_sha256':config['simulator_source_sha256'],'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'policies':policies,'test_started':False,'selection':'earliest training minimum; validation scores do not select methods or weights'};lock['selection_sha256']=digest(policies);(a.output_dir/'frozen_selection_before_test.json').write_text(json.dumps(lock,indent=2))
    scenarios=[s for s in m['scenarios'] if s['split']=='test'];assert len(scenarios)==80;started=time.perf_counter();rows=[];fields=['algorithm','optimizer_seed','scenario_id','case','robots','completed','timed_out','total_steps','mean_wait','max_wait','mean_spawn_delay']
    with (a.output_dir/'test_raw.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for alg,seed,w in policies:
            print(alg,seed,flush=True)
            for s in scenarios:
                r=sim.run(robots=s['robots'],seed=s['seed'],spawn_window=s['spawn_window'],capacity=m['capacity'],max_steps=m['max_steps'],policy='fcfs' if w is None else 'heuristic',weights=w);assert digest([{k:x[k] for k in ('id','start','goal','planned','path')} for x in r['robot_results']])==s['missions_sha256'];row=dict(algorithm=alg,optimizer_seed=seed,scenario_id=s['scenario_id'],case=s['case'],robots=s['robots'],completed=r['completed'],timed_out=r['timed_out'],total_steps=r['total_steps'],mean_wait=r['mean_wait_steps_all_agents'],max_wait=r['max_wait_steps_all_agents'],mean_spawn_delay=r['mean_spawn_delay']);rows.append(row);writer.writerow(row)
            f.flush()
    assert len(rows)==1280 and len({(r['algorithm'],r['optimizer_seed'],r['scenario_id']) for r in rows})==1280
    summary=[]
    for alg in ('fcfs','basic','bo','pso'):
        rr=[r for r in rows if r['algorithm']==alg];summary.append(dict(algorithm=alg,episodes=len(rr),unique_scenarios=80,mean_steps=statistics.mean(r['total_steps'] for r in rr),incomplete=sum(r['timed_out'] for r in rr),mean_wait=statistics.mean(r['mean_wait'] for r in rr),worst_agent_wait=max(r['max_wait'] for r in rr)))
    save(a.output_dir/'summary.csv',summary)
    if not any(r['timed_out'] for r in rows):save(a.output_dir/'paired_comparisons.csv',analyze(rows,scenarios,config['seeds']))
    out={'episodes':1280,'duplicate_keys':0,'incomplete':sum(r['timed_out'] for r in rows),'test_evaluated':True,'weights_frozen_before_test':True,'no_validation_selection':True,'elapsed_seconds':time.perf_counter()-started,'scope':'80 independent final scenarios for this fixed manifest and simulator; no retuning after viewing.'};(a.output_dir/'audit.json').write_text(json.dumps(out,indent=2));print(out)
    for r in summary:print(r)
if __name__=='__main__':main()
