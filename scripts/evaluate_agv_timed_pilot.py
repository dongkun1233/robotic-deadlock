"""Resumable shared-baseline preparation or development evaluation."""
import argparse
from scripts.agv_timed_pilot_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--layout',required=True);p.add_argument('--phase',choices=['baseline','development'],required=True);p.add_argument('--jobs-dir',type=Path);a=p.parse_args();m=json.loads(a.manifest.read_text());assert source_hashes()==m['source_sha256'];cfg=config_from_manifest(m);layout=build_layout(a.layout);a.output_dir.mkdir(parents=True,exist_ok=True)
 split='pilot_train' if a.phase=='baseline' else 'pilot_development';cases=[s for s in m['scenarios'] if s['layout']==a.layout and s['split']==split]
 policies=[('fcfs','fcfs',None),('fixed4_reference','heuristic',FIXED4),('strict_fcfs','strict_fcfs',None)]
 if a.phase=='baseline':policies.append(('waiting_equivalence','heuristic',np.array([1.,0,0,0])))
 else:
  for seed in m['optimizer_seeds']:
   best=json.loads((a.jobs_dir/f'{a.layout}_{seed}'/'best.json').read_text())
   policies.extend((f"{t['algorithm']}_{seed}",'heuristic',np.array(t['vector'])) for t in best)
 for label,policy,v in policies:
  dest=a.output_dir/f'{a.layout}_{label}.json';record=json.loads(dest.read_text()) if dest.exists() else {'label':label,'split':split,'rows':[]}
  for i,row in enumerate(record['rows']):assert row['scenario_id']==cases[i]['scenario_id']
  for s in cases[len(record['rows']):]:
   row,result=simulate(layout,s,cfg,m['dt_s'],policy,v,m['max_time_s']);row['trajectory_sha256']=digest({'robots':result['robots_detail'],'admission_log':result['admission_log']});record['rows'].append(row);atomic(dest,record)
  print(a.layout,label,len(record['rows']),flush=True)
 if a.phase=='baseline':
  f=json.loads((a.output_dir/f'{a.layout}_fcfs.json').read_text())['rows'];w=json.loads((a.output_dir/f'{a.layout}_waiting_equivalence.json').read_text())['rows'];assert all(x['trajectory_sha256']==y['trajectory_sha256'] for x,y in zip(f,w)),'FCFS anchor is not equivalent'
if __name__=='__main__':main()
