"""Exploratory paired, cell-stratified analysis; never accesses final-test data."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import numpy as np

SEEDS = [101, 202, 303, 404, 505]

def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def analyze(base, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((base/'manifest.json').read_text())
    sha = hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()
    paired, variability, cells, plans, convergence = [], [], [], [], []
    rng = np.random.default_rng(20261104)
    z = NormalDist().inv_cdf(1-.05/(2*15)) + NormalDist().inv_cdf(.8)
    for layout in ['single', 'double', 'multi']:
        cases = [x for x in manifest['scenarios'] if x['split']=='main_development' and x['layout']==layout]
        ids = sorted(x['scenario_id'] for x in cases)
        cell_of = {x['scenario_id']:x['cell'] for x in cases}
        groups = [np.array([i for i,s in enumerate(ids) if cell_of[s]==c]) for c in sorted(set(cell_of.values()))]
        assert len(ids)==48 and len(groups)==12 and all(len(g)==4 for g in groups)
        policies = {}
        for path in sorted((base/'development'/layout).glob('*.json')):
            d=json.loads(path.read_text()); rows={r['scenario_id']:r for r in d['rows']}
            assert len(d['rows'])==48 and set(rows)==set(ids) and d['config']['manifest_sha256']==sha
            assert d['status']=='complete' and all(r['status']=='completed' and r['safety_checks_passed'] for r in rows.values())
            policies[d['config']['label']]=np.array([rows[s]['makespan_s'] for s in ids])
        assert len(policies)==19
        aggregate={p:policies[p] for p in ['fcfs','strict_fcfs','fixed4_reference','warmstart_best']}
        for method in ['basic','bo','pso']:
            matrix=np.array([policies[f'{method}_{s}'] for s in SEEDS])
            aggregate[method]=matrix.mean(axis=0)
            means=matrix.mean(axis=1)
            variability.append(dict(layout=layout,method=method,mean_makespan_s=float(means.mean()),
                seed_mean_sd_s=float(means.std(ddof=1)),seed_mean_min_s=float(means.min()),seed_mean_max_s=float(means.max())))
            for cell,g in enumerate(groups):
                cells.append(dict(layout=layout,cell=cell,method=method,scenarios=4,
                    method_mean_s=float(aggregate[method][g].mean()),fcfs_mean_s=float(aggregate['fcfs'][g].mean()),
                    difference_s=float((aggregate[method][g]-aggregate['fcfs'][g]).mean())))
            trials=[json.loads((base/'search'/f'{layout}_{s}_{method}'/'checkpoint.json').read_text())['trials'] for s in SEEDS]
            for budget in [6,18,36,54,72]:
                values=[min(t['objective'] for t in ts[:budget]) for ts in trials]
                convergence.append(dict(layout=layout,method=method,charged_search_observations=budget,
                    mean_best_train_objective=float(np.mean(values)),sd_best_train_objective=float(np.std(values,ddof=1))))
        # Same within-cell resampling indices for every contrast. Five optimizer
        # seeds are averaged within scenario, never counted as 240 independent cases.
        resamples=np.concatenate([rng.choice(g,size=(20000,len(g)),replace=True) for g in groups],axis=1)
        contrasts=[(method,ref) for method in ['basic','bo','pso'] for ref in ['fcfs','strict_fcfs','warmstart_best']]
        contrasts += [('bo','basic'),('pso','basic'),('strict_fcfs','fcfs')]
        for target,reference in contrasts:
            diff=aggregate[target]-aggregate[reference]
            lower,upper=np.quantile(diff[resamples].mean(axis=1),[.025,.975])
            paired.append(dict(layout=layout,target=target,reference=reference,n_scenarios=48,
                difference_s=float(diff.mean()),bootstrap_lower_s=float(lower),bootstrap_upper_s=float(upper),
                relative_difference_pct=float(100*diff.mean()/aggregate[reference].mean()),
                faster=int((diff < -1e-8).sum()),tied=int((np.abs(diff)<=1e-8).sum()),slower=int((diff>1e-8).sum()),
                scope='exploratory unadjusted 95% CI; fixed fitted policies; within-cell scenario uncertainty only'))
            if (target in ['basic','bo','pso'] and reference=='fcfs') or (target in ['bo','pso'] and reference=='basic'):
                percent=100*diff/aggregate['fcfs'].mean()
                sd=float(percent.std(ddof=1))
                pooled=math.sqrt(sum((len(g)-1)*float(percent[g].var(ddof=1)) for g in groups)/36)
                sd90=float(np.quantile(percent[resamples].std(axis=1,ddof=1),.9))
                planning_sd=max(sd,pooled,sd90)
                for effect in [.5,1.,2.]:
                    plans.append(dict(layout=layout,target=target,reference=reference,target_percentage_points=effect,
                        planning_sd_percentage_points=planning_sd,approx_scenarios=max(96,12*math.ceil((z*planning_sd/effect)**2/12)),
                        family_size=15,power_target=.8,scope='normal approximation; not a power guarantee; excludes new-training uncertainty'))
    for name,rows in [('paired_comparisons.csv',paired),('seed_variability.csv',variability),
                      ('condition_comparisons.csv',cells),('sample_size_sensitivity.csv',plans),('training_convergence.csv',convergence)]:
        csv_write(out/name,rows)
    metadata=dict(manifest_sha256=sha,analysis_seed=20261104,bootstrap_replicates=20000,
        paired_scenarios_per_layout=48,condition_cells=12,replicates_per_cell=4,optimizer_seeds=SEEDS,
        final_test_accessed=False,final_sample_sizes_locked=False,
        limitations=['Development-only exploratory analysis after observing these data.',
          'Unadjusted intervals across 36 contrasts; no confirmatory significance claim.',
          'Intervals condition on five fitted policies and fixed cell mixture; do not cover new training draws.',
          'Means across optimizer seeds are paired within each scenario; effective scenario n is 48, not 240.',
          'A proposed 15-comparison final family and practical effect size require explicit protocol lock.',
          'Training step size and budget adequacy require additional analysis; small seed SD is not proof of convergence.'])
    (out/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2))
    return paired,variability,plans,convergence

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('outputs/agv_timed_main_v1'))
    p.add_argument('--output',type=Path,default=Path('outputs/agv_timed_main_v1/development_analysis'))
    args=p.parse_args(); result=analyze(args.base,args.output)
    print('Analysis complete:',len(result[0]),'paired comparisons; final test untouched')
