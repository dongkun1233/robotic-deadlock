"""Discrete node-move FCFS prototype for the multicross layout.

One move takes one step; rotation/service times are not modeled.
Core cells and the first downstream cell are reserved on admission.
This prototype can report incomplete/stalled runs; it does not prove liveness.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from rware.agv_multicross_layout import build_multicross_layout


@dataclass
class Robot:
    robot_id: int
    path: list[str]
    planned: int
    index: int = 0
    spawned: int | None = None
    finished: int | None = None
    wait: int = 0
    zone_wait_since: int | None = None
    zone: str | None = None


def shortest_path(graph, start, goal):
    parents, queue = {start: None}, deque([start])
    while queue and goal not in parents:
        node = queue.popleft()
        for edge_id in graph.outgoing[node]:
            nxt = graph.edges[edge_id].to_node
            if nxt not in parents:
                parents[nxt] = node
                queue.append(nxt)
    if goal not in parents:
        raise ValueError(f'No route: {start} -> {goal}')
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    return list(reversed(path))


def run(robots=12, seed=101, capacity=2, max_steps=2000, spawn_window=20):
    if robots < 1 or capacity not in (1, 2) or max_steps < 1 or spawn_window < 0:
        raise ValueError('Invalid simulation settings')
    layout = build_multicross_layout()
    rng = random.Random(seed)
    starts = sorted(n for n in layout.stations if n.endswith('_IN'))
    goals = sorted(n for n in layout.stations if n.endswith('_OUT'))
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
                rid = queue[0]
                r = active[rid]
                segment = []
                for n in r.path[r.index+1:]:
                    segment.append(n)
                    if n not in areas[z].conflict_zone_nodes:
                        break
                used = {n for holders in reservations.values() for nodes in holders.values() for n in nodes}
                if any(n in occupied or n in used for n in segment):
                    break
                reservations[z][rid] = set(segment)
                r.zone = z
                admissions[z] += 1
                admission_log.append({'step': step, 'robot': rid, 'zone': z,
                    'waiting': step-r.zone_wait_since})
                queue.pop(0)
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

    result = {'policy': 'fcfs', 'robots': robots, 'seed': seed, 'capacity': capacity,
        'completed': sum(r.finished is not None for r in agents), 'total_steps': step+1,
        'max_steps': max_steps, 'timed_out': any(r.finished is None for r in agents),
        'max_no_move_steps': max_stall, 'zone_admissions': admissions,
        'safety_checks': 'Node overlap, occupied-cell moves, entry permission and zone capacity asserted each step',
        'settings_note': 'Node-move prototype; no rotation/service duration; exit-cell reservation; fixed routes',
        'robot_results': [{'id': r.robot_id, 'start': r.path[0], 'goal': r.path[-1],
            'planned': r.planned, 'spawned': r.spawned, 'finished': r.finished,
            'wait_steps': r.wait, 'path_nodes': len(r.path), 'path': r.path} for r in agents],
        'admission_log': admission_log}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--robots', type=int, default=12)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--capacity', type=int, default=2)
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = run(args.robots, args.seed, args.capacity, args.max_steps)
    print(json.dumps({k:v for k,v in result.items() if k not in ('robot_results','admission_log')}, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print('Saved:', args.output)
