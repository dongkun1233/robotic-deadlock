"""One independently resumable layout/seed pilot job, training split only."""
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.agv_timed_pilot_common import *
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--baseline-dir',type=Path,required=True);p.add_argument('--layout',choices=['single','double','multi'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();m=json.loads(a.manifest.read_text());assert m['source_sha256']==source_hashes();assert a.seed in m['optimizer_seeds'];scenarios=[s for s in m['scenarios'] if s['split']=='pilot_train' and s['layout']==a.layout];assert len(scenarios)==12;cfg=config_from_manifest(m);layout=build_layout(a.layout)
 a.output_dir.mkdir(parents=True,exist_ok=True);cache_dir=a.output_dir/'cache';cache_dir.mkdir(exist_ok=True);spec={'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'layout':a.layout,'seed':a.seed,'source_sha256':m['source_sha256']};checkpoint=a.output_dir/'checkpoint.json'
 state=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'config':spec,'trials':[]}
 assert state['config']==spec
 for baseline,vector in [('fcfs',initial(a.seed)[0]),('fixed4_reference',FIXED4)]:
  dest=cache_dir/(key(vector)+'.json')
  if not dest.exists():
   rows=json.loads((a.baseline_dir/(a.layout+'_'+baseline+'.json')).read_text())['rows'];atomic(dest,{'vector':list(map(float,vector)),'rows':rows,'objective':objective(rows,m['max_time_s']),'origin':'shared_baseline','actual_here_episodes':0})
 for alg in m['methods']:
  print(a.layout,a.seed,alg,flush=True);history=[t for t in state['trials'] if t['algorithm']==alg];number=0
  def evaluate(vector,source):
   nonlocal number
   number+=1;v=list(map(float,vector))
   if number<=len(history):
    t=history[number-1];assert t['evaluation']==number and all(abs(x-y)<1e-12 for x,y in zip(v,t['vector']));return t['objective']
   path=cache_dir/(key(vector)+'.json');hit=path.exists()
   if hit:cached=json.loads(path.read_text());assert cached['vector']==v
   else:
    partial=path.with_suffix('.partial.json');rows=json.loads(partial.read_text())['rows'] if partial.exists() else []
    for i in range(len(rows)):assert rows[i]['scenario_id']==scenarios[i]['scenario_id']
    for s in scenarios[len(rows):]:
     row,_=simulate(layout,s,cfg,m['dt_s'],'heuristic',vector,m['max_time_s']);rows.append(row);atomic(partial,{'rows':rows})
    cached={'vector':v,'rows':rows,'objective':objective(rows,m['max_time_s']),'origin':'this_job','actual_here_episodes':len(rows)};atomic(path,cached);partial.unlink(missing_ok=True)
   t={'algorithm':alg,'seed':a.seed,'layout':a.layout,'evaluation':number,'source':source,'vector':v,'weights':weights(vector),'objective':cached['objective'],'incomplete':sum(r['status']!='completed' for r in cached['rows']),'mean_makespan_s':statistics.mean(r['makespan_s'] for r in cached['rows']),'cache_key':key(vector),'cache_hit':hit};state['trials'].append(t);atomic(checkpoint,state);return t['objective']
  search(alg,a.seed,m['budget'],initial(a.seed),evaluate);assert number==18
 best=[]
 for alg in m['methods']:
  ts=[t for t in state['trials'] if t['algorithm']==alg];chosen=min(ts,key=lambda t:(t['objective'],t['evaluation']));assert chosen['objective']<=ts[0]['objective']+1e-9;best.append(chosen)
 atomic(a.output_dir/'best.json',best);atomic(a.output_dir/'complete.json',{'complete':True,'trials':len(state['trials']),'split':'pilot_train','final_test_evaluated':False});print('DONE',a.layout,a.seed,flush=True)
if __name__=='__main__':main()
