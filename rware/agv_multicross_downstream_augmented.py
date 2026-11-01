"""Experimental training-only downstream policy; v2 safety/admission preserved.

The v1 traffic rules, routes, reservations, feature values and normalization are
retained. The only default change is who is eligible for admission. Features are
computed on ALL queued candidates; FCFS is earliest arrival among feasible ones.
No claim of global deadlock freedom is made.
"""
from __future__ import annotations
from rware.agv_multicross_downstream_features import downstream_nodes, occupancy
import math
import random
from rware.agv_multicross_layout import build_multicross_layout
from rware.agv_multicross_policy_simulation import Robot, shortest_path, local_segment, candidate_features, FEATURES

def run(robots=12, seed=101, capacity=2, max_steps=2000, spawn_window=20,
        policy='fcfs', weights=None, grid_rows=2, grid_cols=3,
        admission_rule='feasible_first', downstream_penalty=0.0):
    if not math.isfinite(downstream_penalty) or downstream_penalty<0:
        raise ValueError('Invalid downstream penalty')
    if robots < 1 or capacity not in (1, 2) or max_steps < 1 or spawn_window < 0:
        raise ValueError('Invalid simulation settings')
    if admission_rule not in ('strict_first', 'feasible_first'):
        raise ValueError('Invalid admission rule')
    if policy not in ('fcfs', 'heuristic', 'downstream', 'waiting_downstream', 'full_downstream'):
        raise ValueError('policy must be fcfs, heuristic or downstream')
    weights = dict(weights) if weights is not None else dict(zip(FEATURES, (1.0,-0.5,-0.5,-0.5)))
    if set(weights) != set(FEATURES) or not all(math.isfinite(float(v)) for v in weights.values()):
        raise ValueError('Provide exactly four finite feature weights')
    layout = build_multicross_layout(rows=grid_rows, cols=grid_cols)
    rng = random.Random(seed)
    starts = sorted(n for n in layout.stations if n.endswith('_IN'))
    goals = sorted(n for n in layout.stations if n.endswith('_OUT'))
    all_waiting=set().union(*(a.waiting_points for a in layout.interaction_areas))
    agents = []
    for robot_id in range(robots):
        start = rng.choice(starts)
        goal = rng.choice([g for g in goals if g.rsplit('_', 1)[0] != start.rsplit('_', 1)[0]])
        agents.append(Robot(robot_id, shortest_path(layout.graph, start, goal), rng.randint(0, spawn_window)))
    zone_by_node = {n: a.area_id for a in layout.interaction_areas for n in a.conflict_zone_nodes}
    areas = {a.area_id: a for a in layout.interaction_areas}
    reservations = {z: {} for z in areas}
    queues = {z: [] for z in areas}
    admissions = {z: 0 for z in areas}
    admission_log = []
    decision_log = []
    competitive_decisions = 0
    diversity = {f: 0 for f in FEATURES}
    active = {}
    steps_without_progress = 0
    max_stall = 0
    step = -1

    for step in range(max_steps):
        occupied = {r.path[r.index]: rid for rid, r in active.items()}
        reserved = {n: rid for holders in reservations.values() for rid, nodes in holders.items() for n in nodes}
        for r in sorted(agents, key=lambda r: (r.planned, r.robot_id)):
            if r.spawned is None and r.planned <= step:
                if r.path[0] not in occupied and r.path[0] not in reserved:
                    r.spawned = step
                    active[r.robot_id] = r
                    occupied[r.path[0]] = r.robot_id
        # Every outside->inside transition must be from the area's waiting point.
        for rid, r in sorted(active.items()):
            nxt = r.path[r.index+1]
            z = zone_by_node.get(nxt)
            current_z = zone_by_node.get(r.path[r.index])
            if z and z != current_z:
                assert r.path[r.index] in areas[z].waiting_points
                if rid not in reservations[z] and rid not in queues[z]:
                    r.zone_wait_since = step
                    queues[z].append(rid)
        # FCFS has head-of-line blocking. A second compatible path may be admitted.
        for z, queue in queues.items():
            while queue and len(reservations[z]) < capacity:
                raw, scaled = candidate_features(queue, active, areas[z], layout.graph, step)
                scores = {rid: sum(float(weights[f])*scaled[rid][f] for f in FEATURES) for rid in queue}
                used = {n for holders in reservations.values() for nodes in holders.values() for n in nodes}
                eligible = [candidate for candidate in queue
                    if not any(n in occupied or n in used for n in local_segment(active[candidate], areas[z]))]
                # Feature values and scaling still use the entire waiting set.
                # Only admission eligibility is changed, identically for both policies.
                if policy in ('downstream','waiting_downstream','full_downstream'):
                    down={candidate:occupancy(downstream_nodes(active[candidate],areas[z],all_waiting),occupied,used) for candidate in queue}
                    if policy=='downstream':scores={r:-down[r] for r in queue}
                    elif policy=='waiting_downstream':scores={r:scaled[r]['waiting']-downstream_penalty*down[r] for r in queue}
                    else:scores={r:scores[r]-downstream_penalty*down[r] for r in queue}
                pool = queue if admission_rule == 'strict_first' else eligible
                if not pool:
                    decision_log.append({'step':step, 'zone':z, 'ranked_first':None,
                        'blocked':True,'eligible':[], 'candidates':[
                            {'robot':candidate,'raw':raw[candidate],'scaled':scaled[candidate],
                             'score':scores[candidate]} for candidate in queue]})
                    break
                if policy == 'fcfs':
                    rid = pool[0]
                else:
                    rid = min(pool, key=lambda rid: (-scores[rid], active[rid].zone_wait_since, rid))
                # In feasible_first mode FCFS means earliest arrival among feasible candidates.
                if len(queue) >= 2:
                    competitive_decisions += 1
                    for f in FEATURES:
                        diversity[f] += len({raw[r][f] for r in queue}) > 1
                r = active[rid]
                segment = local_segment(r, areas[z])
                used = {n for holders in reservations.values() for nodes in holders.values() for n in nodes}
                blocked = any(n in occupied or n in used for n in segment)
                decision_log.append({'step': step, 'zone': z, 'ranked_first': rid,
                    'blocked': blocked, 'eligible':eligible, 'candidates': [{'robot': candidate, 'raw': raw[candidate],
                    'scaled': scaled[candidate], 'score': scores[candidate]} for candidate in queue]})
                if blocked:
                    break
                reservations[z][rid] = set(segment)
                r.zone = z
                admissions[z] += 1
                admission_log.append({'step': step, 'robot': rid, 'zone': z,
                    'waiting': step-r.zone_wait_since})
                queue.remove(rid)
        reserved = {n: rid for holders in reservations.values() for rid, nodes in holders.items() for n in nodes}
        proposals = {}
        for rid, r in sorted(active.items()):
            current, nxt = r.path[r.index], r.path[r.index+1]
            z = zone_by_node.get(nxt)
            needs_permission = z is not None and zone_by_node.get(current) != z
            if needs_permission and rid not in reservations[z]:
                r.wait += 1
                continue
            if nxt in occupied or (nxt in reserved and reserved[nxt] != rid) or nxt in proposals.values():
                r.wait += 1
                continue
            proposals[rid] = nxt
        # Conservative synchronous movement: occupied cells cannot be entered this step.
        assert len(set(proposals.values())) == len(proposals)
        for rid, nxt in proposals.items():
            r = active[rid]
            assert nxt not in occupied
            r.index += 1
            if r.zone and nxt not in areas[r.zone].conflict_zone_nodes:
                reservations[r.zone].pop(rid)
                r.zone = None
                r.zone_wait_since = None
        positions = [r.path[r.index] for r in active.values()]
        assert len(positions) == len(set(positions)), 'Node collision'
        for rid, r in list(active.items()):
            if r.index == len(r.path)-1:
                r.finished = step+1
                del active[rid]
        for z, holders in reservations.items():
            assert len(holders) <= capacity
        if all(r.finished is not None for r in agents):
            break
        if active and not proposals:
            steps_without_progress += 1
        else:
            steps_without_progress = 0
        max_stall = max(max_stall, steps_without_progress)

    result = {'admission_rule':admission_rule, 'policy': policy, 'weights': weights if policy == 'heuristic' else None,
        'grid_rows': grid_rows, 'grid_cols': grid_cols,
        'robots': robots, 'seed': seed, 'capacity': capacity,
        'completed': sum(r.finished is not None for r in agents), 'total_steps': step+1,
        'max_steps': max_steps, 'timed_out': any(r.finished is None for r in agents),
        'max_no_move_steps': max_stall, 'zone_admissions': admissions,
        'mean_wait_steps_all_agents': sum(r.wait for r in agents)/robots,
        'max_wait_steps_all_agents': max(r.wait for r in agents),
        'mean_spawn_delay': sum((r.spawned if r.spawned is not None else step+1)-min(r.planned,step+1) for r in agents)/robots,
        'competitive_decisions': competitive_decisions, 'feature_diversity': diversity,
        'safety_checks': 'Node overlap, occupied-cell moves, entry permission and zone capacity asserted each step',
        'settings_note': 'Node-move prototype; no rotation/service duration; exit-cell reservation; fixed routes',
        'robot_results': [{'id': r.robot_id, 'start': r.path[0], 'goal': r.path[-1],
            'planned': r.planned, 'spawned': r.spawned, 'finished': r.finished,
            'wait_steps': r.wait, 'path_nodes': len(r.path), 'path': r.path} for r in agents],
        'admission_log': admission_log, 'decision_log': decision_log}
    return result


