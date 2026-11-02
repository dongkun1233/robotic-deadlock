"""Utilities for the predeclared physical-time baseline pilot."""
import csv,hashlib,json,math,random,statistics,sys,time
from dataclasses import fields
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from rware.agv_timed_simulation import build_layout,run
from rware.agv_multicross_policy_simulation import FEATURES,shortest_path
from rware.agv_physical_time import AGVPhysicalTimeConfig
from rware.agv_multicross_experiments import digest
from scripts.run_multicross_search import search,unit
SOURCE_FILES=['rware/agv_timed_simulation.py','rware/agv_physical_time.py','rware/agv_layouts.py','rware/agv_multicross_layout.py','rware/agv_multicross_policy_simulation.py','scripts/run_multicross_search.py','scripts/agv_timed_pilot_common.py','scripts/train_agv_timed_pilot.py']
FIXED4=unit(np.array([1.,.8,1.2,1.5]))

def atomic(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def read_csv(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def save_csv(path,rows):
 if not rows:return
 temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 temp.replace(path)
def source_hashes():return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES}
def config_from_manifest(m):return AGVPhysicalTimeConfig(**{f.name:m['physical_config'][f.name] for f in fields(AGVPhysicalTimeConfig)})
def initial(seed):
 rng=np.random.default_rng(seed)
 return np.array([np.array([1.,0.,0.,0.]),FIXED4.copy(),*[unit(v) for v in rng.uniform(-1,1,(4,4))]])
def key(vector):return digest([float(v).hex() for v in vector])
def weights(vector):return {f:float(v) for f,v in zip(FEATURES,vector)}
def objective(rows,max_time):return sum(r['status']!='completed' for r in rows)*(max_time+1)+statistics.mean(r['makespan_s'] for r in rows)

def generate_missions(layout,robots,seed,window,profile):
 rng=random.Random(seed);starts=sorted(n for n in layout.stations if n.endswith(('_IN','_START')));goals=sorted(n for n in layout.stations if n.endswith(('_OUT','_EXIT')));dominant_start=rng.choice(starts);dominant_goal=rng.choice(goals);missions=[]
 for i in range(robots):
  start=dominant_start if profile=='approach_skew' and rng.random()<.6 else rng.choice([s for s in starts if s!=dominant_start]) if profile=='approach_skew' else rng.choice(starts)
  available=[g for g in goals if g.rsplit('_',1)[0]!=start.rsplit('_',1)[0]]
  goal=dominant_goal if profile=='exit_skew' and dominant_goal in available and rng.random()<.6 else rng.choice([g for g in available if g!=dominant_goal] or available) if profile=='exit_skew' and dominant_goal in available else rng.choice(available)
  missions.append(dict(id=i,path=shortest_path(layout.graph,start,goal),planned_s=float(rng.randint(0,window))))
 return missions,dominant_start,dominant_goal

def simulate(layout,scenario,config,dt,policy,vector,max_time):
 start=time.perf_counter();cpu=time.process_time();r=run(layout,scenario['missions'],config=config,dt_s=dt,policy=policy,weights=None if vector is None else weights(vector),max_time_s=max_time)
 assert digest([dict(id=x['id'],path=x['path'],planned_s=x['planned_s']) for x in r['robots_detail']])==scenario['missions_sha256']
 row=dict(scenario_id=scenario['scenario_id'],layout=scenario['layout'],cell=scenario['cell'],robots=scenario['robots'],status=r['status'],completed=r['completed'],makespan_s=r['makespan_s'],mean_stationary_wait_s=r['mean_wait_s'],p95_stationary_wait_s=r['p95_wait_s'],max_stationary_wait_s=r['max_wait_s'],mean_entry_wait_s=statistics.mean(x['entry_wait_s'] for x in r['robots_detail']),mean_spawn_delay_s=r['mean_spawn_delay_s'],mean_quantization_s=statistics.mean(x['quantization_s'] for x in r['robots_detail']),episode_cpu_s=time.process_time()-cpu,episode_wall_s=time.perf_counter()-start)
 return row,r
