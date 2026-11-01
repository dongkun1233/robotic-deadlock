"""Observable route-corridor occupancy, without future arrivals or simulation rollout."""
from rware.agv_multicross_policy_simulation import local_segment

def downstream_nodes(robot,area,all_waiting):
    start=robot.index+len(local_segment(robot,area));nodes=[]
    for node in robot.path[start:]:
        nodes.append(node)
        if node in all_waiting:break
    if not nodes:raise ValueError('Missing downstream corridor')
    return nodes

def occupancy(nodes,occupied,reserved):
    if not nodes:raise ValueError('Empty downstream corridor')
    return len(set(nodes)&(set(occupied)|set(reserved)))/len(nodes)
