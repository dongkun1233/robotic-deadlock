"""Observe downstream occupancy and wait causes on training only, without changing v2."""
import argparse,csv,hashlib,inspect,json,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware import agv_multicross_policy_v2 as sim
from rware.agv_multicross_experiments import digest

def downstream_nodes(robot,area,all_waiting):
    local=sim.local_segment(robot,area)
    # local ends at the first cell OUTSIDE the current core.
    start=robot.index+len(local);nodes=[]
    for node in robot.path[start:]:
        nodes.append(node)
        if node in all_waiting:break
    return nodes

def occupancy(nodes,occupied,reserved):
    assert nodes
    return len(set(nodes)&(set(occupied)|set(reserved)))/len(nodes)

def trace_points():
    source,start=inspect.getsourcelines(sim.run);choice=[start+i for i,l in enumerate(source) if l.strip()=='if not pool:'];waits=[start+i for i,l in enumerate(source) if l.strip()=='r.wait += 1'];assert len(choice)==1 and len(waits)==2
    return choice[0],waits

def observe(kwargs,scenario_id,policy_id):
    choice,waits=trace_points();records=[];causes=Counter();all_waiting=None;attempt=0
    def trace(frame,event,arg):
        nonlocal all_waiting,attempt
        if frame.f_code is not sim.run.__code__:return None
        if event!='line':return trace
        v=frame.f_locals;line=frame.f_lineno
        if line==choice:
            if len(v['queue'])<2:return trace
            if all_waiting is None:all_waiting=set().union(*(a.waiting_points for a in v['areas'].values()))
            attempt+=1
            vals={rid:occupancy(downstream_nodes(v['active'][rid],v['areas'][v['z']],all_waiting),v['occupied'],v['used']) for rid in v['queue']}
            elig=v['eligible'];fcfs=elig[0] if elig else None;pick=min(elig,key=lambda rid:(vals[rid],v['active'][rid].zone_wait_since,rid)) if elig else None
            for rid in v['queue']:
                nodes=downstream_nodes(v['active'][rid],v['areas'][v['z']],all_waiting)
                records.append(dict(scenario_id=scenario_id,policy_id=policy_id,decision=attempt,step=v['step'],zone=v['z'],robot=rid,queued=len(v['queue']),eligible=rid in elig,eligible_count=len(elig),fcfs_pick=rid==fcfs,downstream_pick=rid==pick,occupancy=vals[rid],corridor_cells=len(nodes),downstream_nodes='|'.join(nodes),**v['raw'][rid]))
        elif line in waits:
            rid=v['rid'];r=v['r'];nxt=r.path[r.index+1]
            if line==waits[0]:
                z=v['z'];holders=v['reservations'][z];used={n for h in v['reservations'].values() for ns in h.values() for n in ns};segment=sim.local_segment(r,v['areas'][z])
                if len(holders)>=v['capacity']:cause='entry_capacity_full'
                elif any(n in v['occupied'] for n in segment):cause='entry_segment_occupied'
                elif any(n in used for n in segment):cause='entry_segment_reserved'
                else:cause='entry_other_ordering'
            else:
                cause='movement_occupied' if nxt in v['occupied'] else 'movement_reserved' if nxt in v['reserved'] and v['reserved'][nxt]!=rid else 'movement_same_target'
            causes[cause]+=1
        return trace
    old=sys.gettrace();sys.settrace(trace)
    try:r=sim.run(**kwargs)
    finally:sys.settrace(old)
    assert sum(causes.values())==sum(x['wait_steps'] for x in r['robot_results'])
    return r,records,causes

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--training-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);m=json.loads(a.manifest.read_text());state=json.loads((a.training_dir/'checkpoint.json').read_text());assert state['config']['manifest_sha256']==hashlib.sha256(a.manifest.read_bytes()).hexdigest();assert state['config']['simulator_source_sha256']==hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest()
    bank_candidates=sorted([t for t in state['trials'] if t['algorithm']=='basic'],key=lambda t:(t['objective'],t['seed'],t['evaluation']));best=bank_candidates[0];policies=[('fcfs',None),('basic_training_best',best['weights'])];summary=[];decision_summary=[];all_rows=[];wait_rows=[]
    for policy_id,weights in policies:
        print(policy_id,flush=True)
        for s in [s for s in m['scenarios'] if s['split']=='train']:
            kwargs=dict(robots=s['robots'],seed=s['seed'],spawn_window=s['spawn_window'],max_steps=m['max_steps'],capacity=m['capacity'],policy='fcfs' if weights is None else 'heuristic',weights=weights)
            r,records,causes=observe(kwargs,s['scenario_id'],policy_id);control=sim.run(**kwargs);assert r['robot_results']==control['robot_results'] and r['admission_log']==control['admission_log'];assert digest([{k:x[k] for k in ('id','start','goal','planned','path')} for x in r['robot_results']])==s['missions_sha256']
            import itertools
            for _,group in itertools.groupby(records,key=lambda x:x['decision']):
                group=list(group);row=group[0];assert len(group)==row['queued'];e=[x for x in group if x['eligible']]
                decision_summary.append(dict(scenario_id=s['scenario_id'],policy_id=policy_id,step=row['step'],zone=row['zone'],queued=row['queued'],eligible_count=len(e),downstream_varies=len({x['occupancy'] for x in e})>1,old_exit_varies=len({x['exit_competition'] for x in e})>1,old_path_varies=len({x['path_conflict'] for x in e})>1,changes_fcfs=any(x['fcfs_pick']!=x['downstream_pick'] for x in e),max_occupancy=max((x['occupancy'] for x in e),default=0),min_occupancy=min((x['occupancy'] for x in e),default=0)))
            all_rows.extend(records);wait_rows.append(dict(scenario_id=s['scenario_id'],policy_id=policy_id,**{k:causes[k] for k in ['entry_capacity_full','entry_segment_occupied','entry_segment_reserved','entry_other_ordering','movement_occupied','movement_reserved','movement_same_target']}));summary.append(dict(scenario_id=s['scenario_id'],case=s['case'],policy_id=policy_id,completed=r['completed'],robots=s['robots'],total_steps=r['total_steps'],total_wait=sum(causes.values()),max_wait=r['max_wait_steps_all_agents'],mean_spawn_delay=r['mean_spawn_delay']))
    def save(name,rows):
        with (a.output_dir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save('scenario_summary.csv',summary);save('wait_causes.csv',wait_rows);save('decision_opportunities.csv',decision_summary)
    import gzip,io
    with gzip.open(a.output_dir/'candidate_observations.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    out={'split':'train','observed_runs':len(summary),'control_replays':len(summary),'instrumentation_preserves_trajectories':True,'all_wait_steps_accounted':True,'validation_or_test_simulated':False,'selected_basic_seed':best['seed'],'selected_basic_evaluation':best['evaluation'],'selected_basic_weights':best['weights'],'scope':'Decision attempts repeat within scenarios; feature variation and correlation do not prove a performance benefit.'};(a.output_dir/'audit.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
