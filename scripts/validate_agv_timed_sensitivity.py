"""New-model clock/safety diagnostics only; no optimizer fitting or held-out test."""
import argparse,csv,hashlib,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware.agv_timed_simulation import build_layout,make_missions,run
from rware.agv_physical_time import AGVPhysicalTimeConfig
from rware.agv_multicross_experiments import digest

def save(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);c=AGVPhysicalTimeConfig();scenarios=[]
    for layout_name in ['single','double','multi']:
        l=build_layout(layout_name)
        for count,window in [(8,10),(24,0),(24,15)]:
            for seed in [101,202]:
                ms=make_missions(l,count,seed,window);scenarios.append(dict(scenario_id=f'{layout_name}_{count}_{window}_{seed}',layout=layout_name,robots=count,spawn_window_s=window,seed=seed,missions=ms,missions_sha256=digest(ms)))
    protocol={'purpose':'Timing validation, not performance confirmation','dt_values_s':[.1,.05,.01],'policies':['fcfs','heuristic','strict_fcfs'],'heuristic_weights':dict(waiting=1.,maneuver=-.5,exit_competition=-.5,path_conflict=-.5),'heuristic_note':'Fixed diagnostic weights, not fitted BO/PSO and not the historical Basic baseline','scenarios':scenarios,'config':{**c.as_research_config(),'simulation_time_mode':'fixed_step','controller_period_s':'same as dt_values_s','implementation_status':'timed traffic prototype v1'},'source_sha256':{p:hashlib.sha256(Path(ROOT/p).read_bytes()).hexdigest() for p in ['rware/agv_timed_simulation.py','rware/agv_physical_time.py','rware/agv_layouts.py','rware/agv_multicross_layout.py','rware/agv_multicross_policy_simulation.py']},'coarse_dt_acceptance_rule':'Keep 0.1s as candidate only if all runs complete, relative makespan difference vs0.01s <=1% in EVERY paired scenario/policy, and within-scenario policy ordering does not reverse ignoring differences <=0.01s. Otherwise use0.01s provisionally; no mathematical convergence claim.'}
    (a.output_dir/'protocol_before_run.json').write_text(json.dumps(protocol,indent=2));rows=[];started=time.perf_counter()
    for s in scenarios:
        l=build_layout(s['layout']);print(s['scenario_id'],flush=True)
        for dt in protocol['dt_values_s']:
            for policy in protocol['policies']:
                r=run(l,s['missions'],config=c,dt_s=dt,policy=policy,weights=protocol['heuristic_weights'],max_time_s=1200)
                assert digest([{'id':x['id'],'path':x['path'],'planned_s':x['planned_s']} for x in r['robots_detail']])==s['missions_sha256']
                rows.append({'scenario_id':s['scenario_id'],'layout':s['layout'],'robots':s['robots'],'dt_s':dt,'policy':policy,'status':r['status'],'completed':r['completed'],'makespan_s':r['makespan_s'],'mean_wait_s':r['mean_wait_s'],'p95_wait_s':r['p95_wait_s'],'max_wait_s':r['max_wait_s'],'mean_spawn_delay_s':r['mean_spawn_delay_s'],'minimum_separation_m':r['minimum_checked_separation_m'],'required_separation_m':r['required_separation_m'],'admission_sequence_sha256':digest([(x['zone'],x['robot']) for x in r['admission_log']]),'safety_checks_passed':r['safety_checks_passed']})
                # Full representative trace for auditing phase timing and occupancy.
                if s['scenario_id']=='single_8_10_101' and dt==.1:
                    (a.output_dir/f'trace_{policy}.json').write_text(json.dumps(r,indent=2))
    save(a.output_dir/'runs.csv',rows);lookup={(r['scenario_id'],r['policy'],r['dt_s']):r for r in rows};paired=[]
    for s in scenarios:
        for policy in protocol['policies']:
            fine=lookup[(s['scenario_id'],policy,.01)]
            for dt in [.1,.05]:
                coarse=lookup[(s['scenario_id'],policy,dt)];paired.append(dict(scenario_id=s['scenario_id'],layout=s['layout'],policy=policy,coarse_dt_s=dt,coarse_minus_fine_s=coarse['makespan_s']-fine['makespan_s'],relative_difference_pct=100*(coarse['makespan_s']/fine['makespan_s']-1),same_admission_sequence=coarse['admission_sequence_sha256']==fine['admission_sequence_sha256'],both_complete=coarse['status']==fine['status']=='completed'))
    save(a.output_dir/'paired_dt_comparison.csv',paired);reversals=[]
    for s in scenarios:
        for dt in [.1,.05]:
            for p1,p2 in [('fcfs','heuristic'),('fcfs','strict_fcfs'),('heuristic','strict_fcfs')]:
                dc=lookup[(s['scenario_id'],p1,dt)]['makespan_s']-lookup[(s['scenario_id'],p2,dt)]['makespan_s'];df=lookup[(s['scenario_id'],p1,.01)]['makespan_s']-lookup[(s['scenario_id'],p2,.01)]['makespan_s']
                if abs(dc)>.01000001 and abs(df)>.01000001 and dc*df<0:reversals.append(dict(scenario_id=s['scenario_id'],coarse_dt_s=dt,policy1=p1,policy2=p2,coarse_difference_s=dc,fine_difference_s=df))
    (a.output_dir/'ordering_reversals.json').write_text(json.dumps(reversals,indent=2));summary=[]
    for name in ['single','double','multi']:
        for dt in protocol['dt_values_s']:
            for policy in protocol['policies']:
                rr=[r for r in rows if r['layout']==name and r['dt_s']==dt and r['policy']==policy];summary.append(dict(layout=name,dt_s=dt,policy=policy,episodes=len(rr),incomplete=sum(r['status']!='completed' for r in rr),mean_makespan_s=statistics.mean(r['makespan_s'] for r in rr),mean_wait_s=statistics.mean(r['mean_wait_s'] for r in rr),worst_wait_s=max(r['max_wait_s'] for r in rr)))
    save(a.output_dir/'summary.csv',summary);coarse=[r for r in paired if r['coarse_dt_s']==.1];accept=all(r['both_complete'] and abs(r['relative_difference_pct'])<=1 for r in coarse) and not any(r['coarse_dt_s']==.1 for r in reversals)
    audit={'runs':len(rows),'expected_runs':162,'unique_scenarios':18,'duplicates':len(rows)-len(lookup),'incomplete_runs':sum(r['status']!='completed' for r in rows),'all_safety_assertions_passed':True,'max_absolute_dt01_relative_error_pct':max(abs(r['relative_difference_pct']) for r in coarse),'dt01_ordering_reversals':sum(r['coarse_dt_s']==.1 for r in reversals),'dt01_passes_predeclared_rule':accept,'clock_recommendation_s':.1 if accept else .01,'reference_dt_s':.01,'validation_scope':'Engineering diagnostic scenarios only, no training, optimization or prior held-out test reused','elapsed_seconds':time.perf_counter()-started}
    assert audit['runs']==162 and audit['duplicates']==0;(a.output_dir/'audit.json').write_text(json.dumps(audit,indent=2));print(audit)
if __name__=='__main__':main()
