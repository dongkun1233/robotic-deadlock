"""Paired BO/PSO versus random search in saved validation runs; no simulation."""
import argparse,csv,statistics
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('validation_dirs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rng=np.random.default_rng(20261031);rows=[]
    for folder in a.validation_dirs:
        with (folder/'validation_raw.csv').open() as f:raw=list(csv.DictReader(f))
        assert not any(r['timed_out']=='True' for r in raw);assert len(raw)==len({(r['algorithm'],r['optimizer_seed'],r['scenario_id']) for r in raw})
        ids=sorted({r['scenario_id'] for r in raw});cases=np.array([int(next(r['case'] for r in raw if r['scenario_id']==sid)) for sid in ids]);sample=np.concatenate([rng.choice(np.where(cases==c)[0],size=(5000,sum(cases==c)),replace=True) for c in sorted(set(cases))],axis=1)
        for alg in ('bo','pso'):
            diff=[]
            for sid in ids:
                av={r['optimizer_seed']:float(r['total_steps']) for r in raw if r['algorithm']==alg and r['scenario_id']==sid};bv={r['optimizer_seed']:float(r['total_steps']) for r in raw if r['algorithm']=='basic' and r['scenario_id']==sid};assert set(av)==set(bv)
                diff.append(statistics.mean(av[s]-bv[s] for s in sorted(av)))
            d=np.array(diff);lo,hi=np.percentile(d[sample].mean(axis=1),[2.5,97.5]);rows.append(dict(validation_dir=folder.name,comparison=alg+'_minus_random',mean_difference=float(d.mean()),ci_lower=float(lo),ci_upper=float(hi),unique_validation_scenarios=len(ids),interpretation='Exploratory unadjusted scenario bootstrap, conditional on fitted seeds; no optimizer-population inference.'))
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(rows)
if __name__=='__main__':main()
