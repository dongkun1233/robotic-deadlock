"""Fair-priority fork of agv_timed_simulation (baseline SHA256 1700161951f9bb812bdf39cf2cdd99c76d76659d87cca2886e32287567f13779).

Physical dynamics unchanged; adds the teammate-style overdue-wait priority.
Fixed-tick physical-time AGV traffic prototype with common admission eligibility.

Seconds are the external unit. Rotation and translation end at ceil(duration/dt)
phase boundaries. Source and destination remain locked through both phases.
The circle envelope bounds the rectangular body for center spin turns. This is
not a real-robot safety certification or a global liveness proof.
"""
from __future__ import annotations
from dataclasses import dataclass
import math,random
from rware.agv_layouts import build_fcfs_cross_shared_area_layout,build_fcfs_double_cross_shared_area_layout
from rware.agv_multicross_layout import build_multicross_layout
from rware.agv_multicross_policy_simulation import FEATURES,shortest_path,local_segment,candidate_features
from rware.agv_physical_time import AGVPhysicalTimeConfig


def build_layout(name):
    if name=='single':layout=build_fcfs_cross_shared_area_layout(corridor_length=8)
    elif name=='double':layout=build_fcfs_double_cross_shared_area_layout(corridor_length=8)
    elif name=='multi':return build_multicross_layout(rows=2,cols=3)
    else:raise ValueError('Unknown layout')
    # Legacy layouts rely on waypoint-constrained routing on a broadly connected
    # graph. For generic route generation restrict edges to the right-hand lanes.
    origins=[(min(layout.graph.nodes[n].position[0] for n in a.conflict_zone_nodes),min(layout.graph.nodes[n].position[1] for n in a.conflict_zone_nodes)) for a in layout.interaction_areas]
    keep={}
    for eid,e in layout.graph.edges.items():
        p=layout.graph.nodes[e.from_node].position;q=layout.graph.nodes[e.to_node].position;delta=(q[0]-p[0],q[1]-p[1]);allowed=set()
        for y,x in origins:
            if p[1]==x:allowed.add((1,0))
            if p[1]==x+1:allowed.add((-1,0))
            if p[0]==y:allowed.add((0,-1))
            if p[0]==y+1:allowed.add((0,1))
        if delta in allowed:keep[eid]=e
    layout.graph.edges=keep
    for node in layout.graph.outgoing:layout.graph.outgoing[node]=[e for e in layout.graph.outgoing[node] if e in keep]
    layout.routes={};layout.name+='_right_lane_adapter_v1'
    return layout


def make_missions(layout,robots,seed,spawn_window_s):
    if robots<1 or spawn_window_s<0:raise ValueError('Invalid mission settings')
    rng=random.Random(seed);nodes=layout.stations
    starts=sorted(n for n in nodes if n.endswith(('_START','_IN')))
    goals=sorted(n for n in nodes if n.endswith(('_EXIT','_OUT')))
    def gate(n):return n.rsplit('_',1)[0]
    missions=[]
    for i in range(robots):
        start=rng.choice(starts);goal=rng.choice([g for g in goals if gate(g)!=gate(start)])
        missions.append({'id':i,'path':shortest_path(layout.graph,start,goal),'planned_s':float(rng.randint(0,int(spawn_window_s)))})
    return missions


@dataclass
class Agent:
    robot_id:int
    path:list
    planned_s:float
    index:int=0
    spawned_s:float|None=None
    finished_s:float|None=None
    heading:tuple|None=None
    zone_wait_since:float|None=None
    zone:str|None=None
    stationary_wait_s:float=0.
    entry_wait_s:float=0.
    rotation_s:float=0.
    translation_s:float=0.
    quantization_s:float=0.
    trips:int=0


def ticks(duration,dt):return max(0,math.ceil(duration/dt-1e-10))

def _position(agent,trip,t,graph,config):
    origin=graph.nodes[agent.path[agent.index]].position
    if trip is None:return tuple(x*config.cell_size_m for x in origin)
    dest=graph.nodes[trip['to']].position
    fraction=max(0.,min(1.,(t-trip['move_start_s'])/trip['move_duration_s']))
    return tuple((origin[i]+fraction*(dest[i]-origin[i]))*config.cell_size_m for i in range(2))

def _minimum_separation(a,ta,b,tb,now,graph,config):
    end=max(ta['finish_s'] if ta else now,tb['finish_s'] if tb else now)
    cuts={now,end}
    for tr in (ta,tb):
        if tr:
            for t in (tr['move_start_s'],tr['move_start_s']+tr['move_duration_s'],tr['finish_s']):
                if now<t<end:cuts.add(t)
    cuts=sorted(cuts);minimum=math.inf
    for start,stop in zip(cuts,cuts[1:]):
        pa=_position(a,ta,start,graph,config);pb=_position(b,tb,start,graph,config);qa=_position(a,ta,stop,graph,config);qb=_position(b,tb,stop,graph,config)
        d=[pa[i]-pb[i] for i in range(2)];v=[qa[i]-qb[i]-d[i] for i in range(2)];vv=sum(x*x for x in v)
        f=max(0.,min(1.,-sum(d[i]*v[i] for i in range(2))/vv)) if vv else 0.
        minimum=min(minimum,math.hypot(*(d[i]+f*v[i] for i in range(2))))
    return minimum


def run(layout,missions,*,config=None,dt_s=.01,policy='fcfs',weights=None,max_time_s=1200.,record_decisions=False,wait_threshold_s=None):
    c=config or AGVPhysicalTimeConfig()
    if wait_threshold_s is not None and (not math.isfinite(wait_threshold_s) or wait_threshold_s<0):raise ValueError('Invalid wait threshold')
    if not math.isfinite(dt_s) or dt_s<=0 or not math.isfinite(max_time_s) or max_time_s<=0:raise ValueError('Invalid clock')
    if policy not in ('fcfs','heuristic','strict_fcfs'):raise ValueError('Unknown policy')
    if c.shared_zone_capacity not in (1,2):raise ValueError('Prototype supports capacity one or two')
    if 2*c.reserved_radius_m>min(c.cell_size_m,c.single_lane_width_m):raise ValueError('Envelope does not fit a cell/lane')
    if not math.isclose(c.single_lane_width_m,c.cell_size_m) or not math.isclose(c.two_lane_width_m,2*c.cell_size_m):raise ValueError('Grid spacing and lane widths differ')
    w=dict(zip(FEATURES,[1.,-.5,-.5,-.5])) if weights is None else dict(weights)
    if set(w)!=set(FEATURES) or not all(math.isfinite(float(v)) for v in w.values()):raise ValueError('Four finite weights required')
    graph=layout.graph;areas={a.area_id:a for a in layout.interaction_areas};zone_by_node={n:z for z,a in areas.items() for n in a.conflict_zone_nodes};agents=[];ids=set()
    for m in missions:
        path=list(m['path']);rid=m['id'];planned=float(m['planned_s'])
        if rid in ids or len(path)<2 or not math.isfinite(planned) or planned<0:raise ValueError('Invalid mission')
        ids.add(rid)
        if path[0] in zone_by_node:raise ValueError('Spawn inside core not supported')
        for n,nxt in zip(path,path[1:]):
            a=graph.nodes[n].position;b=graph.nodes[nxt].position
            if sum(abs(a[i]-b[i]) for i in range(2))!=1 or not any(graph.edges[e].to_node==nxt for e in graph.outgoing[n]):raise ValueError('Require adjacent cardinal directed edges')
            if nxt in zone_by_node and zone_by_node.get(n)!=zone_by_node[nxt] and n not in areas[zone_by_node[nxt]].waiting_points:raise ValueError('Unprotected entry')
        first=graph.nodes[path[0]].position;second=graph.nodes[path[1]].position;heading=tuple(m.get('initial_heading',(second[0]-first[0],second[1]-first[1])))
        if heading not in [(-1,0),(0,1),(1,0),(0,-1)]:raise ValueError('Invalid heading')
        agents.append(Agent(rid,path,planned,heading=heading))
    if not agents:raise ValueError('Empty mission set')
    active={};trips={};queues={z:[] for z in areas};holds={z:{} for z in areas};decisions=[];admissions=[];events=[];min_separation=math.inf;finish_tick=ticks(max_time_s,dt_s);status='timeout';time_s=0.;loop_ticks=0;fairness_log=[]
    def reservations():
        reserved={}
        for hs in holds.values():
            for rid,nodes in hs.items():
                for n in nodes:
                    assert n not in reserved or reserved[n]==rid,'Reservation ownership collision'
                    reserved[n]=rid
        for rid,tr in trips.items():
            n=tr['to'];assert n not in reserved or reserved[n]==rid,'Destination reservation collision';reserved[n]=rid
        return reserved
    for tick in range(finish_tick+1):
        time_s=tick*dt_s;loop_ticks+=1
        if time_s>max_time_s+1e-9:break
        # Complete ALL arrivals before any release, spawn or policy selection.
        arrived=[]
        for rid,tr in sorted(list(trips.items())):
            if tr['rotation_ticks'] and tr['move_start_tick']==tick:
                active[rid].heading=tr['heading'];events.append({'t':time_s,'event':'rotation_finished','robot':rid});events.append({'t':time_s,'event':'translation_started','robot':rid})
            if tr['finish_tick']==tick:arrived.append((rid,tr))
        for rid,tr in arrived:
            r=active[rid];r.index+=1;r.heading=tr['heading'];del trips[rid];events.append({'t':time_s,'event':'move_finished','robot':rid,'node':r.path[r.index]})
        for rid,tr in arrived:
            r=active[rid]
            if r.zone and r.path[r.index] not in areas[r.zone].conflict_zone_nodes:
                z=r.zone;assert tr['from'] in areas[z].conflict_zone_nodes;holds[z].pop(rid);r.zone=None;r.zone_wait_since=None;events.append({'t':time_s,'event':'zone_released','robot':rid,'zone':z})
        for rid,tr in arrived:
            r=active[rid]
            if r.index==len(r.path)-1:r.finished_s=time_s;del active[rid]
        occupied={r.path[r.index]:rid for rid,r in active.items()};assert len(occupied)==len(active),'Cell overlap'
        reserved=reservations()
        for r in sorted(agents,key=lambda r:(r.planned_s,r.robot_id)):
            if r.spawned_s is None and r.planned_s<=time_s+1e-9 and r.path[0] not in occupied and r.path[0] not in reserved:
                r.spawned_s=time_s;active[r.robot_id]=r;occupied[r.path[0]]=r.robot_id;events.append({'t':time_s,'event':'spawn','robot':r.robot_id})
        if all(r.finished_s is not None for r in agents):status='completed';break
        for rid,r in sorted(active.items()):
            if rid in trips or r.zone is not None:continue
            z=zone_by_node.get(r.path[r.index+1])
            if z and zone_by_node.get(r.path[r.index])!=z and rid not in queues[z]:
                assert r.path[r.index] in areas[z].waiting_points;r.zone_wait_since=time_s;queues[z].append(rid)
        new_admissions=0
        for z,queue in queues.items():
            queue.sort(key=lambda rid:(active[rid].zone_wait_since,rid))
            while queue and len(holds[z])<c.shared_zone_capacity:
                reserved=reservations();eligible=[rid for rid in queue if not any(n in occupied or n in reserved for n in local_segment(active[rid],areas[z]))]
                raw,scaled=candidate_features(queue,active,areas[z],graph,time_s);scores={rid:sum(w[f]*scaled[rid][f] for f in FEATURES) for rid in queue}
                chosen=queue[0] if policy=='strict_fcfs' else eligible[0] if eligible and policy=='fcfs' else min(eligible,key=lambda rid:(-scores[rid],active[rid].zone_wait_since,rid)) if eligible else None
                original_chosen=chosen
                overdue=[]
                if wait_threshold_s is not None and policy=='heuristic':
                    chosen,overdue=fair_choice(queue,eligible,active,scores,time_s,dt_s,wait_threshold_s)
                if chosen is not None and chosen in overdue:
                    fairness_log.append({'t':time_s,'zone':z,'robot':chosen,'queue_wait_s':queue_wait_s(active[chosen],time_s,dt_s),'eligible_count':len(eligible),'baseline_selected':original_chosen,'override':chosen!=original_chosen})
                if record_decisions:decisions.append({'t':time_s,'zone':z,'candidates':list(queue),'eligible':list(eligible),'selected':chosen,'admitted':chosen in eligible if chosen is not None else False,'raw':raw,'scaled':scaled})
                if chosen is None or chosen not in eligible:break
                r=active[chosen];holds[z][chosen]=set(local_segment(r,areas[z]));r.zone=z;r.entry_wait_s+=time_s-r.zone_wait_since;queue.remove(chosen);new_admissions+=1;admissions.append({'t':time_s,'zone':z,'robot':chosen,'eligible':list(eligible)});events.append({'t':time_s,'event':'admitted','robot':chosen,'zone':z})
        started=0
        for rid,r in sorted(active.items()):
            if rid in trips:continue
            origin=r.path[r.index];target=r.path[r.index+1];z=zone_by_node.get(target)
            if z and zone_by_node.get(origin)!=z and rid not in holds[z]:continue
            reserved=reservations()
            if target in occupied or (target in reserved and reserved[target]!=rid):continue
            p=graph.nodes[origin].position;q=graph.nodes[target].position;heading=(q[0]-p[0],q[1]-p[1]);cross=r.heading[0]*heading[1]-r.heading[1]*heading[0];dot=sum(r.heading[i]*heading[i] for i in range(2));angle=abs(math.atan2(cross,dot));rot=c.rotation_time_s(angle);move=c.translation_time_s(c.cell_size_m);rt=ticks(rot,dt_s);mt=ticks(move,dt_s)
            tr={'from':origin,'to':target,'heading':heading,'rotation_ticks':rt,'move_start_tick':tick+rt,'finish_tick':tick+rt+mt,'move_start_s':(tick+rt)*dt_s,'move_duration_s':move,'finish_s':(tick+rt+mt)*dt_s}
            for other_id,other in active.items():
                if other_id==rid:continue
                separation=_minimum_separation(r,tr,other,trips.get(other_id),time_s,graph,c);min_separation=min(min_separation,separation);assert separation+1e-9>=2*c.reserved_radius_m,'Continuous swept-envelope overlap'
            trips[rid]=tr;r.trips+=1;r.rotation_s+=rot;r.translation_s+=move;r.quantization_s+=(rt+mt)*dt_s-rot-move;started+=1;events.append({'t':time_s,'event':'trip_started','robot':rid,'from':origin,'to':target,'angle_rad':angle,'rotation_s':rot,'translation_s':move,'move_start_s':tr['move_start_s'],'finish_s':tr['finish_s']})
        reserved=reservations()
        for n,rid in occupied.items():assert n not in reserved or reserved[n]==rid,'Reservation intersects another occupied cell'
        for hs in holds.values():assert len(hs)<=c.shared_zone_capacity
        # With all releases due, no trip and no possible action, this fixed-route
        # finite batch cannot change state in future ticks (no replanning).
        if not trips and not started and not new_admissions and time_s>=max(r.planned_s for r in agents)-1e-9:status='stalled';break
        if tick<finish_tick:
            for rid,r in active.items():
                if rid not in trips:r.stationary_wait_s+=dt_s
    robot_rows=[]
    for r in agents:
        delay=(r.spawned_s if r.spawned_s is not None else min(time_s,max_time_s))-r.planned_s
        robot_rows.append({'id':r.robot_id,'planned_s':r.planned_s,'spawned_s':r.spawned_s,'finished_s':r.finished_s,'spawn_delay_s':max(0.,delay),'stationary_wait_s':r.stationary_wait_s,'entry_wait_s':r.entry_wait_s,'rotation_s':r.rotation_s,'translation_s':r.translation_s,'quantization_s':r.quantization_s,'path':r.path})
        if r.finished_s is not None:assert abs((r.finished_s-r.spawned_s)-(r.stationary_wait_s+r.rotation_s+r.translation_s+r.quantization_s))<1e-6,'Time accounting mismatch'
    waits=sorted(r.stationary_wait_s for r in agents);pos=.95*(len(waits)-1);lo=int(pos);hi=math.ceil(pos);p95=waits[lo]+(waits[hi]-waits[lo])*(pos-lo)
    return {'wait_threshold_s':wait_threshold_s,'fairness_log':fairness_log,'fairness_protected_admissions':len(fairness_log),'fairness_override_count':sum(x['override'] for x in fairness_log),'fairness_protected_competitions':sum(x['eligible_count']>=2 for x in fairness_log),'layout':layout.name,'policy':policy,'weights':w if policy=='heuristic' else None,'dt_s':dt_s,'status':status,'robots':len(agents),'completed':sum(r.finished_s is not None for r in agents),'makespan_s':max((r.finished_s or 0) for r in agents) if status=='completed' else max_time_s,'observed_end_s':min(time_s,max_time_s),'mean_wait_s':sum(waits)/len(waits),'p95_wait_s':p95,'max_wait_s':max(waits),'mean_spawn_delay_s':sum(r['spawn_delay_s'] for r in robot_rows)/len(agents),'minimum_checked_separation_m':min_separation if math.isfinite(min_separation) else None,'required_separation_m':2*c.reserved_radius_m,'simulation_ticks':loop_ticks,'waits_censored':status!='completed','safety_checks_passed':True,'robots_detail':robot_rows,'admission_log':admissions,'decision_log':decisions,'events':events,'limitations':'Conservative cell/path reservation, fixed routes, center-spin circle envelope, no linear acceleration or service; no global liveness guarantee.'}


def queue_wait_s(agent, now, dt):
    # Original teammate uses integer tick difference * dt, not float subtraction.
    return round((now-agent.zone_wait_since)/dt)*dt


def fair_choice(queue, eligible, active, scores, now, dt, threshold):
    overdue=[rid for rid in eligible if queue_wait_s(active[rid],now,dt)>=threshold]
    if overdue:
        return min(overdue,key=lambda rid:(-queue_wait_s(active[rid],now,dt),active[rid].zone_wait_since,rid)),overdue
    return (min(eligible,key=lambda rid:(-scores[rid],active[rid].zone_wait_since,rid)) if eligible else None),overdue
