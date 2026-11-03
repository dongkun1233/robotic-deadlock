from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from scripts.agv_timed_main_common import *
from scripts.run_multicross_search import search
b=Path('outputs/agv_timed_main_v1'); m=read_manifest(b/'manifest.json'); bank=json.loads((b/'initial_weights.json').read_text())
class EndReplay(Exception): pass
report=[]
for p in sorted((b/'search').glob('*/checkpoint.json')):
 s=json.loads(p.read_text()); c=s['config']; layout=c['layout']; items=bank['selected_by_layout'][layout]['candidates']; cases=training_cases(m,layout)
 assert c['manifest_sha256']==hashlib.sha256((b/'manifest.json').read_bytes()).hexdigest()
 assert c['initial_weights_sha256']==hashlib.sha256((b/'initial_weights.json').read_bytes()).hexdigest()
 count=[0]; raw=0; gaps=[]
 def ev(v,source):
  i=count[0]
  if i==len(s['trials']): raise EndReplay()
  t=s['trials'][i]; assert t['evaluation']==i+1 and t['vector']==list(map(float,v)) and t['source']==source,(p,i,'replay mismatch')
  count[0]+=1; return t['objective']
 try: search(c['method'],c['seed'],72,np.array([x['vector'] for x in items]),ev)
 except EndReplay: pass
 assert count[0]==len(s['trials'])
 for t in s['trials']:
  initial=next((x for x in items if x['vector']==t['vector']),None)
  rp=b/'calibration'/layout/(initial['candidate_id']+'.json') if initial else p.parent/'cache'/(key(t['vector'])+'.json')
  if not rp.exists(): gaps.append(dict(evaluation=t['evaluation'],available_rows=0)); continue
  r=json.loads(rp.read_text()); require_rows(r['rows'],cases,complete=False)
  assert initial is not None or r['vector']==t['vector']
  assert all(x['safety_checks_passed'] for x in r['rows'])
  if len(r['rows'])<48: gaps.append(dict(evaluation=t['evaluation'],available_rows=len(r['rows']))); continue
  assert objective(r['rows'],m['max_time_s'])==t['objective']
  raw+=1
 report.append(dict(job=p.parent.name,trials=count[0],replay_exact=True,raw_complete=raw,raw_gaps=gaps))
 print(p.parent.name,count[0],raw,len(gaps),flush=True)
atomic_durable(b/'recovery_verification_20261104.json',dict(source_hashes_match=True,search_paths_exact=True,jobs=report,observations=sum(x['trials'] for x in report),raw_gaps=sum(len(x['raw_gaps']) for x in report),final_test_evaluated=False))
