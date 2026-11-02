"""Fix new pilot train/development cases; never generate/evaluate final test here."""
import argparse,itertools,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.agv_timed_pilot_common import *
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((ROOT/'configs/agv_timed_v1.json').read_text());cases=[];seen=set();seeds=set()
 old=json.loads((ROOT/'outputs/agv_timed_validation_v1/protocol_before_run.json').read_text());old_cases={(s['layout'],s['missions_sha256']) for s in old['scenarios']}
 for split_index,(split,repeats) in enumerate([('pilot_train',1),('pilot_development',2)]):
  for layout_index,name in enumerate(['single','double','multi']):
   layout=build_layout(name)
   for cell,(count,window,profile) in enumerate(itertools.product([12,24],[0,15],['uniform','approach_skew','exit_skew'])):
    for rep in range(repeats):
     seed=15100000+split_index*100000+layout_index*10000+cell*100+rep;ms,domstart,domgoal=generate_missions(layout,count,seed,window,profile);d=digest(ms);identity=(name,d)
     assert identity not in seen and identity not in old_cases and seed not in seeds;seen.add(identity);seeds.add(seed)
     cases.append(dict(scenario_id=f'{split}_{name}_c{cell:02d}_{rep:02d}',split=split,layout=name,cell=cell,robots=count,spawn_window_s=window,demand_profile=profile,dominant_start=domstart,dominant_goal=domgoal,seed=seed,missions_sha256=d,missions=ms))
 manifest={'protocol':'timed_baseline_pilot_v1','physical_config':cfg,'dt_s':.01,'max_time_s':1200.,'optimizer_seeds':[101,202,303],'budget':18,'initial_count':6,'methods':['basic','bo','pso'],'method_labels':{'basic':'random_search','bo':'GP_LCB_BO','pso':'projected_PSO'},'fixed4_reference':weights(FIXED4),'fixed4_note':'Projection of legacy fixed weights [1,.8,1.2,1.5]; approach_queue is omitted, so not identical to legacy seven-feature Basic. Fixed, not trained.','common_initialization':'FCFS + normalized fixed4 + four seeded random unit vectors','cache_rule':'Shared observations reused within a layout/seed; every initial observation counts toward each method budget. Actual unique simulations and charged evaluation counts reported separately.','pilot_counts':{'pilot_train':36,'pilot_development':72,'final_test':0},'source_sha256':source_hashes(),'scenarios':cases}
 atomic(a.output_dir/'manifest.json',manifest);save_csv(a.output_dir/'scenario_index.csv',[{k:v for k,v in s.items() if k!='missions'} for s in cases]);print(manifest['pilot_counts'])
if __name__=='__main__':main()
