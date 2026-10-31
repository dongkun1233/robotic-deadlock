"""Warehouse-style repetition of the existing 2x2 intersection concept."""
from collections import deque
from rware.agv_layouts import AGVLayout
from rware.agv_topology import LaneGraph
from rware.agv_types import Node, InteractionArea


def build_multicross_layout(rows=2, cols=3, margin=6, gap=8):
    if min(rows, cols) < 1 or margin < 3 or gap < 4:
        raise ValueError('Invalid layout dimensions')
    ys = [margin + i * (gap + 2) for i in range(rows)]
    xs = [margin + i * (gap + 2) for i in range(cols)]
    height, width = ys[-1] + 2 + margin, xs[-1] + 2 + margin
    roads = {(r, c) for r in range(height) for x in xs for c in (x, x+1)}
    roads |= {(r, c) for y in ys for r in (y, y+1) for c in range(width)}
    special, areas, stations = {}, [], {}
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            z = f'Z{i*cols+j+1}'
            core = {}
            for label, pos in [('NW',(y,x)),('SW',(y+1,x)),('SE',(y+1,x+1)),('NE',(y,x+1))]:
                node = f'{z}_CP_{label}'
                special[pos] = (node, 'conflict', z)
                core[node] = pos
            waiting = set()
            for label, pos in [('N',(y-1,x)),('S',(y+2,x+1)),('W',(y+1,x-1)),('E',(y,x+2))]:
                node = f'{z}_WAIT_{label}'
                special[pos] = (node, 'waiting', z)
                waiting.add(node)
            areas.append(InteractionArea(area_id=z, area_type='intersection',
                communication_zone_nodes=waiting.copy(), conflict_zone_nodes=set(core),
                waiting_points=waiting, conflict_points=set(core), allowed_routes={}))
    gates = []
    for j, x in enumerate(xs, 1):
        gates += [(f'N{j}', (0,x), (0,x+1)), (f'S{j}', (height-1,x+1), (height-1,x))]
    for i, y in enumerate(ys, 1):
        gates += [(f'W{i}', (y+1,0), (y,0)), (f'E{i}', (y,width-1), (y+1,width-1))]
    for gate, start, end in gates:
        for suffix, pos in [('IN', start), ('OUT', end)]:
            node = f'{gate}_{suffix}'
            special[pos] = (node, 'station', None)
            stations[node] = node
    graph = LaneGraph()
    for pos in sorted(roads):
        r,c = pos
        name,kind,area = special.get(pos, (f'M_{r:02d}_{c:02d}', 'normal_lane', None))
        graph.add_node(Node(name, pos, kind, ['forward','turn_left','turn_right','wait'], interaction_area_id=area))
    for pos, name in graph.position_to_node.items():
        r,c = pos
        deltas = set()
        for x in xs:
            if c == x: deltas.add((1,0))
            if c == x+1: deltas.add((-1,0))
        for y in ys:
            if r == y: deltas.add((0,-1))
            if r == y+1: deltas.add((0,1))
        for dr,dc in sorted(deltas):
            target = graph.position_to_node.get((r+dr,c+dc))
            if target is not None:
                graph.add_adjacent_edge(name, target, edge_type='intersection' if graph.nodes[target].node_type == 'conflict' else 'lane')
    cells = [['x'] * width for _ in range(height)]
    for r,c in roads: cells[r][c] = '.'
    for _,_,end in gates: cells[end[0]][end[1]] = 'g'
    return AGVLayout(name=f'multicross_{rows}x{cols}_v1', grid_size=(height,width), graph=graph,
        interaction_areas=areas, rware_layout='\n'.join(''.join(row) for row in cells),
        routes={}, initial_nodes={}, initial_headings={}, stations=stations,
        shelf_storage_cells={(r,c) for r in range(height) for c in range(width) if (r,c) not in roads})


def check_layout(layout):
    graph = layout.graph
    failures = []
    incoming = 0
    for area in layout.interaction_areas:
        for edge in graph.edges.values():
            if edge.to_node in area.conflict_zone_nodes and edge.from_node not in area.conflict_zone_nodes:
                incoming += 1
                if edge.from_node not in area.waiting_points:
                    failures.append((area.area_id, edge.from_node, edge.to_node))
    starts = sorted(n for n in layout.stations if n.endswith('_IN'))
    goals = sorted(n for n in layout.stations if n.endswith('_OUT'))
    unreachable = []
    for start in starts:
        seen, queue = {start}, deque([start])
        while queue:
            node = queue.popleft()
            for edge_id in graph.outgoing[node]:
                nxt = graph.edges[edge_id].to_node
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        unreachable.extend((start,goal) for goal in goals if goal not in seen)
    print('Layout:', layout.name, 'Grid:', layout.grid_size)
    print('Shared zones:', len(layout.interaction_areas))
    print('Nodes:', len(graph.nodes), 'Edges:', len(graph.edges))
    print('Start gates:', len(starts), 'Exit gates:', len(goals))
    print('Zone entry edges:', incoming, 'Unprotected entries:', len(failures))
    print('Reachable start-exit pairs:', len(starts)*len(goals)-len(unreachable), '/', len(starts)*len(goals))
    assert not failures, failures
    assert not unreachable, unreachable
    print('Topology checks passed; multi-AGV movement is not yet validated.')


if __name__ == '__main__':
    check_layout(build_multicross_layout())
