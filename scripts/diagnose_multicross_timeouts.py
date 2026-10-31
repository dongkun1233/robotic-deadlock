"""Inspect training timeouts without altering simulator traffic rules."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rware import agv_multicross_policy_simulation as sim
from rware.agv_multicross_layout import build_multicross_layout


def cyclic_components(edges):
    def reach(start):
        seen=set();todo=[start]
        while todo:
            node=todo.pop()
            if node in seen:continue
            seen.add(node);todo.extend(edges.get(node,[]))
        return seen
    reachability={n:reach(n) for n in edges}
    groups=[];used=set()
    for node in edges:
        if node in used:continue
        component={other for other in edges if other in reachability[node] and node in reachability[other]}
        used.update(component)
        if len(component)>1:groups.append(sorted(component))
    return groups


def snapshot_run(scenario,weights):
    captured={}
    def profiler(frame,event,arg):
        if event=='return' and frame.f_code is sim.run.__code__:
            locals_=frame.f_locals
            captured['agents']=[{'id':r.robot_id,'index':r.index,'current':r.path[r.index],
                'next':r.path[r.index+1] if r.index+1<len(r.path) else None,
                'zone':r.zone,'wait_steps':r.wait} for r in locals_['active'].values()]
            captured['queues']={z:list(q) for z,q in locals_['queues'].items()}
            captured['reservations']={z:{str(rid):sorted(nodes) for rid,nodes in holders.items()} for z,holders in locals_['reservations'].items()}
    previous=sys.getprofile()
    try:
        sys.setprofile(profiler)
        result=sim.run(robots=scenario['robots'],seed=scenario['seed'],spawn_window=scenario['spawn_window'],
            policy='heuristic',weights=weights,max_steps=2000,capacity=2)
    finally:
        sys.setprofile(previous)
    occupied={r['current']:r['id'] for r in captured['agents']}
    reserved={node:int(rid) for holders in captured['reservations'].values() for rid,nodes in holders.items() for node in nodes}
    edges={r['id']:[] for r in captured['agents']}
    for r in captured['agents']:
        # Only immediate occupied/reserved-next-cell dependencies are included.
        # Admission ranking can introduce additional dependencies not captured here.
        for owner in (occupied.get(r['next']),reserved.get(r['next'])):
            if owner is not None and owner!=r['id'] and owner not in edges[r['id']]:edges[r['id']].append(owner)
    captured['next_cell_wait_for_edges']=edges
    captured['next_cell_cycles']=cyclic_components(edges)
    missions={m['id']:m for m in scenario['missions']}
    agents={r['id']:r for r in captured['agents']}
    areas={a.area_id:a for a in build_multicross_layout().interaction_areas}
    eligibility=[]
    for z,queue in captured['queues'].items():
        if not queue:continue
        last=next((d for d in reversed(result['decision_log']) if d['zone']==z),None)
        for rid in queue:
            mission=missions[rid]
            robot=sim.Robot(rid,mission['path'],mission['planned'],index=agents[rid]['index'])
            segment=sim.local_segment(robot,areas[z])
            blockers=sorted({owner for node in segment for owner in (occupied.get(node),reserved.get(node)) if owner is not None and owner!=rid})
            eligibility.append({'zone':z,'robot':rid,'ranked_first_in_last_decision':bool(last and last['ranked_first']==rid),
                'segment':segment,'blocking_robots':blockers,'segment_clear':not blockers,
                'capacity_available':len(captured['reservations'][z])<2})
    captured['queue_candidate_feasibility']=eligibility
    captured['summary']={k:result[k] for k in ('completed','robots','total_steps','timed_out','max_no_move_steps')}
    captured['recent_decisions']=result['decision_log'][-12:]
    captured['note']='Diagnostic snapshot of fixed-route simulated blocking; not a formal proof of deadlock freedom or a real-world safety claim.'
    return captured


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--training-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    scenarios={s['scenario_id']:s for s in manifest['scenarios'] if s['split']=='train'}
    state=json.loads((args.training_dir/'checkpoint.json').read_text())
    failures=[]
    for trial in state['trials']:
        with (args.training_dir/trial['raw_file']).open() as file:
            for row in csv.DictReader(file):
                if row['timed_out']=='True':
                    failures.append({'algorithm':trial['algorithm'],'optimizer_seed':trial['seed'],
                        'evaluation':trial['evaluation'],'scenario_id':row['scenario_id'],
                        'completed':int(row['completed']),'robots':int(row['robots']),
                        'weights_json':json.dumps(trial['weights'],sort_keys=True)})
    if failures:
        with (args.output_dir/'failed_training_episodes.csv').open('x',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(failures[0]));writer.writeheader();writer.writerows(failures)
    examples=[]
    for scenario_id in sorted({r['scenario_id'] for r in failures}):
        failure=next(r for r in failures if r['scenario_id']==scenario_id)
        snapshot=snapshot_run(scenarios[scenario_id],json.loads(failure['weights_json']))
        assert snapshot['summary']['timed_out']
        assert snapshot['summary']['completed']==failure['completed']
        examples.append({'failure':failure,**snapshot})
        print(scenario_id,snapshot['summary'],'next-cell cycles',snapshot['next_cell_cycles'])
    (args.output_dir/'timeout_snapshots.json').write_text(json.dumps(examples,indent=2))
    print('Recorded failures:',len(failures),'Unique scenarios:',len(examples))


if __name__=='__main__':main()
