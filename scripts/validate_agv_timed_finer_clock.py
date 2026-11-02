"""Extra engineering cross-check at 0.005s on six loaded diagnostic scenarios."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware.agv_timed_simulation import run,build_layout

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);protocol=json.loads((a.base_dir/'protocol_before_run.json').read_text());assert hashlib.sha256((ROOT/'rware/agv_timed_simulation.py').read_bytes()).hexdigest()==protocol['source_sha256']['rware/agv_timed_simulation.py'];ss=[s for s in protocol['scenarios'] if s['robots']==24 and s['seed']==101];assert len(ss)==6
 with (a.base_dir/'runs.csv').open() as f:base=list(csv.DictReader(f))
 (a.output_dir/'protocol_before_run.json').write_text(json.dumps({'scenario_selection':'All six original 24-robot, seed101 diagnostic cases across three layouts and two release windows','dt_s':.005,'comparison_dt_s':.01,'policies':protocol['policies'],'purpose':'Extra discretization check, not an optimizer performance test','source_sha256':protocol['source_sha256']},indent=2));rows=[];reversals=[]
 for s in ss:
  print(s['scenario_id'],flush=True)
  for policy in protocol['policies']:
   r=run(build_layout(s['layout']),s['missions'],dt_s=.005,policy=policy,weights=protocol['heuristic_weights'],max_time_s=1200);old=next(x for x in base if x['scenario_id']==s['scenario_id'] and x['policy']==policy and float(x['dt_s'])==.01)
   rows.append(dict(scenario_id=s['scenario_id'],layout=s['layout'],policy=policy,dt005_makespan_s=r['makespan_s'],dt001_makespan_s=float(old['makespan_s']),dt001_minus_dt005_pct=100*(float(old['makespan_s'])/r['makespan_s']-1),completed=r['completed'],robots=r['robots'],safety_checks_passed=r['safety_checks_passed']))
  rr=[r for r in rows if r['scenario_id']==s['scenario_id']]
  for i in range(3):
   for j in range(i+1,3):
    x=rr[i];y=rr[j];d1=x['dt001_makespan_s']-y['dt001_makespan_s'];d2=x['dt005_makespan_s']-y['dt005_makespan_s']
    if abs(d1)>.01000001 and abs(d2)>.01000001 and d1*d2<0:reversals.append([s['scenario_id'],x['policy'],y['policy'],d1,d2])
 with (a.output_dir/'runs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 audit={'runs':len(rows),'incomplete':sum(r['completed']!=r['robots'] for r in rows),'max_absolute_dt001_vs_dt0005_difference_pct':max(abs(r['dt001_minus_dt005_pct']) for r in rows),'ordering_reversals':reversals,'scope':'Six loaded diagnostic cases, not proof of convergence across arbitrary traffic and weights.'};(a.output_dir/'audit.json').write_text(json.dumps(audit,indent=2));print(audit)
if __name__=='__main__':main()
