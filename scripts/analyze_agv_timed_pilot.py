"""Pilot-only descriptive comparisons and transparent approximate sample-size planning."""
import argparse
from statistics import NormalDist
from scripts.agv_timed_pilot_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();base=a.output_dir;m=json.loads((base/'manifest.json').read_text());assert m['source_sha256']==source_hashes();summaries=[];plans=[];trials=[];unique=[]
 z=NormalDist().inv_cdf(1-.05/(2*15))+NormalDist().inv_cdf(.8)
 for layout in ['single','double','multi']:
  jobs=[base/'jobs'/f'{layout}_{seed}' for seed in m['optimizer_seeds']]
  for job in jobs:
   assert json.loads((job/'complete.json').read_text())['trials']==54
   ts=json.loads((job/'checkpoint.json').read_text())['trials'];trials.extend(ts)
   initial_vectors=[[t['vector'] for t in ts if t['algorithm']==alg and t['evaluation']<=6] for alg in m['methods']];assert initial_vectors[0]==initial_vectors[1]==initial_vectors[2]
   for f in (job/'cache').glob('*.json'):
    data=json.loads(f.read_text())
    if f.name.endswith('.partial.json'):
     completed=json.loads(f.with_name(f.name.replace('.partial.json','.json')).read_text());assert completed['rows'][:len(data['rows'])]==data['rows'];continue
    assert len(data['rows'])==12 and len({r['scenario_id'] for r in data['rows']})==12
    if data['origin']=='this_job':unique.extend(data['rows'])
  records={f.stem[len(layout)+1:]:json.loads(f.read_text())['rows'] for f in (base/'development').glob(layout+'_*.json')};assert len(records)==12
  maps={k:{r['scenario_id']:r for r in v} for k,v in records.items()};ids=list(maps['fcfs']);assert len(ids)==24 and all(set(v)==set(ids) for v in maps.values())
  means={alg:[statistics.mean(maps[f'{alg}_{seed}'][i]['makespan_s'] for seed in m['optimizer_seeds']) for i in ids] for alg in m['methods']};means['fcfs']=[maps['fcfs'][i]['makespan_s'] for i in ids]
  for label,rows in records.items():summaries.append({'layout':layout,'policy':label,'n_scenarios':len(rows),'completed':sum(r['status']=='completed' for r in rows),'mean_makespan_s':statistics.mean(r['makespan_s'] for r in rows),'mean_wait_s':statistics.mean(r['mean_stationary_wait_s'] for r in rows)})
  for alg,reference in [(v,'fcfs') for v in m['methods']]+[('bo','basic'),('pso','basic')]:
   differences=[100*(x-y)/statistics.mean(means['fcfs']) for x,y in zip(means[alg],means[reference])];sd=statistics.stdev(differences);bycell={}
   for i,d in zip(ids,differences):bycell.setdefault(maps['fcfs'][i]['cell'],[]).append(d)
   pooled=math.sqrt(sum((len(v)-1)*statistics.variance(v) for v in bycell.values())/sum(len(v)-1 for v in bycell.values()));bootstrap_sd90=float(np.quantile(np.std(np.random.default_rng(67890).choice(differences,size=(10000,len(differences)),replace=True),axis=1,ddof=1),.9));conservative=max(sd,pooled,bootstrap_sd90)
   for target in [.5,1.,2.]:plans.append(dict(layout=layout,comparison=alg+'_vs_'+reference,pilot_scenarios=24,mean_difference_percentage_points=statistics.mean(differences),paired_sd_percentage_points=sd,pooled_within_cell_sd_percentage_points=pooled,bootstrap_sd90_percentage_points=bootstrap_sd90,planning_sd_percentage_points=conservative,target_difference_percentage_points=target,approx_required_scenarios=max(96,12*math.ceil((z*conservative/target)**2/12))))
  unique.extend(r for rows in records.values() for r in rows)
  for f in (base/'baselines').glob(layout+'_*.json'):
   rows=json.loads(f.read_text())['rows'];assert len(rows)==12;unique.extend(rows)
 save_csv(base/'development_summary.csv',summaries);save_csv(base/'sample_size_sensitivity.csv',plans)
 flat=[{**{k:v for k,v in t.items() if k not in ['vector','weights']},**t['weights']} for t in trials];save_csv(base/'train_trials.csv',flat)
 recommended={layout:max(x['approx_required_scenarios'] for x in plans if x['layout']==layout and x['target_difference_percentage_points']==1) for layout in ['single','double','multi']}
 audit={'charged_objective_calls':len(trials),'charged_training_episodes':len(trials)*12,'retained_complete_episode_records':len(unique),'storage_recovery_replayed_episodes':len(json.loads((base/'storage_recovery.json').read_text())['replayed_episodes']) if (base/'storage_recovery.json').exists() else 0,'execution_count_note':'See storage_recovery.json for any replays of missing saved rows; recorded CPU totals exclude lost original-attempt timings.','all_recorded_completed':sum(r['status']=='completed' for r in unique),'cpu_seconds_recorded':sum(r['episode_cpu_s'] for r in unique),'mean_cpu_s_per_episode':statistics.mean(r['episode_cpu_s'] for r in unique),'provisional_final_scenarios_by_layout':recommended,'provenance':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/analyze_agv_timed_pilot.py','scripts/evaluate_agv_timed_pilot.py','scripts/prepare_agv_timed_pilot.py','scripts/run_agv_timed_pilot.py']},'final_test_evaluated':False,'limitations':['Differences are normalized by layout development FCFS mean (constant denominator), matching mean-makespan comparisons.','Planning SD uses the maximum of paired SD, pooled within-cell SD, and bootstrap 90th-percentile SD (10000 resamples, seed 67890).','Normal approximation with only 24 development scenarios/layout; not a guarantee of power.','Optimizer-seed uncertainty and training-set changes require reassessment before final lock.','No final significance claims; scenario repetitions across optimizer seeds are paired.','Runtime timings under concurrent workload are not optimizer speed comparisons.']};atomic(base/'audit.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
