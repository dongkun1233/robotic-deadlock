"""Predeclared final-test analysis of fixed fits; no policy reselection."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import t as student_t
from scripts.run_agv_final_test import OUT,COUNTS,load_locked,collect,sha

def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i])
    adjusted=[0.]*len(order);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(order)-rank)*pvalues[i]))
        adjusted[i]=running
    return adjusted

def stratified(diff,groups):
    mean=float(np.mean([diff[g].mean() for g in groups]))
    terms=np.array([diff[g].var(ddof=1)/len(g)/len(groups)**2 for g in groups])
    variance=float(terms.sum())
    if variance==0:
        return mean,0.,float(len(diff)-len(groups)),1. if abs(mean)<1e-12 else 0.
    df=float(variance**2/sum(term**2/(len(g)-1) for term,g in zip(terms,groups)))
    se=float(np.sqrt(variance));p=float(2*student_t.sf(abs(mean/se),df))
    return mean,se,df,p

def save(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def analyze(out=OUT):
    m,sig=load_locked(out)
    if m['analyzer_sha256']!=sha(__file__):raise ValueError('Frozen analysis code changed')
    records=collect(out,m,sig,True)
    if len(records)!=38988:raise ValueError('Final data incomplete')
    summaries=[];primary=[];secondary=[]
    seeds=[101,202,303,404,505]
    for layout,n in COUNTS.items():
        cases=sorted([s for s in m['scenarios'] if s['layout']==layout],key=lambda s:s['scenario_id'])
        groups=[np.array([i for i,s in enumerate(cases) if s['cell']==c]) for c in range(12)]
        rows={p['label']:{} for p in m['policies'][layout]}
        for r in records:
            if r['layout']==layout:rows[r['label']][r['scenario_id']]=r
        values={}
        for label,mapping in rows.items():
            ordered=[mapping[s['scenario_id']] for s in cases]
            # Average across scenarios equals the frozen training-style objective:
            # (max_time+1)*incomplete_count + mean makespan, also for mixed seeds.
            values[label]=np.array([r['makespan_s']+(m['max_time_s']+1)*n*(r['status']!='completed') for r in ordered])
            summaries.append(dict(layout=layout,policy=label,scenarios=n,
                completed=sum(r['status']=='completed' for r in ordered),
                mean_makespan_s=float(np.mean([r['makespan_s'] for r in ordered])),
                penalized_objective=float(values[label].mean()),
                mean_stationary_wait_s=float(np.mean([r['mean_stationary_wait_s'] for r in ordered]))))
        for method in ['basic','bo','pso']:
            values[method]=np.array([values[f'{method}_{s}'] for s in seeds]).mean(axis=0)
        contrasts=[(a,'fcfs') for a in ['basic','bo','pso']]+[('bo','basic'),('pso','basic')]
        extra=[(a,r) for a in ['basic','bo','pso'] for r in ['strict_fcfs','warmstart_best','fixed4_reference']]+[('strict_fcfs','fcfs')]
        for target,reference in contrasts+extra:
            d=values[target]-values[reference];mean,se,df,p=stratified(d,groups)
            width=float(student_t.ppf(1-.05/(2*15),df))*se
            record=dict(layout=layout,target=target,reference=reference,scenarios=n,
                difference=mean,se=se,df=df,p_raw=p,bonferroni15_lower=mean-width,bonferroni15_upper=mean+width,
                faster=int((d < -1e-8).sum()),tied=int((abs(d)<=1e-8).sum()),slower=int((d>1e-8).sum()))
            if (target,reference) in contrasts:primary.append(record)
            else:
                record.pop('p_raw');record.pop('bonferroni15_lower');record.pop('bonferroni15_upper')
                secondary.append(record)
    assert len(primary)==15
    for r,p in zip(primary,holm([r['p_raw'] for r in primary])):
        r['p_holm15']=p;r['holm_reject_005']=p<.05
    destination=out/'analysis';destination.mkdir(exist_ok=True)
    save(destination/'policy_summary.csv',summaries)
    save(destination/'primary_15_comparisons.csv',primary)
    save(destination/'secondary_descriptive_comparisons.csv',secondary)
    (destination/'analysis_audit.json').write_text(json.dumps(dict(manifest_sha256=sig,
        episodes=38988,primary_comparisons=15,secondary_comparisons=len(secondary),
        incomplete_episodes=sum(r['status']!='completed' for r in records),
        safety_failures=sum(not r['safety_checks_passed'] for r in records),
        method='Condition-stratified paired Welch-Satterthwaite t approximation; Holm correction over 15 two-sided p values',
        estimand='Performance of the fixed five fitted policies, not fresh training data or optimizer seeds',
        failure_penalty='Each failed scenario contributes (max_time+1)*layout_N in addition to capped makespan; policy mean equals count penalty + mean makespan',
        limitations=['Approximate t-test assumptions; no guarantee of exact FWER under arbitrary distributions.',
                    'Fixed scenario mixture, routes and speed; no claims of broad real-world generalization.',
                    'If failures occur, the penalty can dominate; report completion counts and interpret comparisons cautiously.']),indent=2))
    print('FINAL_ANALYSIS_COMPLETE',len(primary),'primary comparisons; no policy selection',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    analyze(p.parse_args().output)
