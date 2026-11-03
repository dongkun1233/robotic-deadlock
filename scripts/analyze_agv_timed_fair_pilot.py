"""Audit and describe the preregistered teammate-style fair-priority pilot."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
from scripts.agv_timed_main_common import ROOT, atomic_durable, read_manifest, require_rows, save_csv
from scripts.run_agv_timed_fair_pilot import initialize, episode, source_hashes


def paired_ci(values,seed):
    rng=np.random.default_rng(seed);values=np.asarray(values)
    means=np.mean(rng.choice(values,size=(5000,len(values)),replace=True),axis=1)
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('outputs/agv_timed_fair_pilot_v1/protocol.json'))
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();base=args.protocol.parent;p=json.loads(args.protocol.read_text())
    if p['source_sha256']!=source_hashes():raise ValueError('Runtime source differs')
    complete=json.loads((base/'complete.json').read_text())
    if not complete['complete']:raise ValueError('Pilot incomplete')
    manifest=read_manifest(ROOT/p['main_manifest'])
    cases={split:[s for s in manifest['scenarios'] if s['layout']=='multi' and s['split']=='main_'+split]
           for split in ['train','development']}
    byid={s['scenario_id']:s for split in cases.values() for s in split}
    records={};physical=[];reused=[];count=0
    for split in ['train','development']:
        for path in sorted((base/split/'cache').glob('*.json')):
            r=json.loads(path.read_text());require_rows(r['rows'],cases[split])
            if r['status']!='complete':raise ValueError('Partial cache remains')
            count+=len(r['rows'])
            for row in r['rows']:
                if row['execution_kind']=='simulated':physical.append(row)
                else:reused.append((r['config'],row))
            records[(split,path.stem)]=r
    trials=[]
    for path in sorted((base/'train').glob('*.json')):
        state=json.loads(path.read_text());ts=state['trials']
        if len(ts)!=p['budget']:raise ValueError('Trial count mismatch')
        if [t['vector'] for t in ts[:6]]!=p['common_initial_vectors']:raise ValueError('Initial weights differ')
        trials.extend(dict(fit=path.stem,**{k:v for k,v in t.items() if k!='vector'},
                           **dict(zip(['waiting','maneuver','exit_competition','path_conflict'],t['vector']))) for t in ts)
    if len(trials)!=9*p['budget']:raise ValueError('Expected nine fitted policies')
    save_csv(base/'training_trials.csv',trials)
    registry=json.loads((base/'development_registry.json').read_text());summary=[];maps={}
    for item in registry:
        rows=records[('development',item['cache_key'])]['rows'];maps[item['label']]={r['scenario_id']:r for r in rows}
        summary.append(dict(policy=item['label'],scenarios=len(rows),completed=sum(r['status']=='completed' for r in rows),
            mean_makespan_s=statistics.mean(r['makespan_s'] for r in rows),
            mean_entry_wait_s=statistics.mean(r['entry_mean_s'] for r in rows),
            mean_scenario_max_entry_wait_s=statistics.mean(r['entry_max_s'] for r in rows),
            mean_scenario_p95_entry_wait_s=statistics.mean(r['entry_p95_s'] for r in rows),
            mean_scenario_max_stationary_wait_s=statistics.mean(r['stationary_max_s'] for r in rows),
            mean_stationary_wait_s=statistics.mean(r['stationary_mean_s'] for r in rows),
            scenarios_with_override=sum(r['overrides']>0 for r in rows),override_admissions=sum(r['overrides'] for r in rows),
            protected_admissions=sum(r['protected_admissions'] for r in rows),
            mean_jain_entry_wait=statistics.mean(r['jain_entry_wait'] for r in rows)))
    save_csv(base/'development_summary.csv',summary)
    comparisons=[]
    for method in p['methods']:
        original=f'{method}_plain_retrained'
        for rule in ['fair20','fair40']:
            overlay=f'{method}_sameweights_{rule}';refit=f'{method}_{rule}_retrained'
            comparisons.extend([(overlay,original,'rule_only'),(refit,original,'rule_and_refit'),(refit,overlay,'refit_given_rule')])
    paired=[]
    for idx,(target,reference,kind) in enumerate(comparisons):
        if set(maps[target])!=set(maps[reference]):raise ValueError('Unpaired development rows')
        for metric in ['makespan_s','entry_max_s','stationary_max_s']:
            diffs=[maps[target][i][metric]-maps[reference][i][metric] for i in maps[reference]]
            lower,upper=paired_ci(diffs,20261103+idx)
            paired.append(dict(target=target,reference=reference,comparison_kind=kind,metric=metric,n=48,
                mean_difference=statistics.mean(diffs),ci95_lower=lower,ci95_upper=upper,
                faster_or_lower=sum(d<-1e-8 for d in diffs),equal=sum(abs(d)<=1e-8 for d in diffs),
                slower_or_higher=sum(d>1e-8 for d in diffs),scope='exploratory unadjusted paired bootstrap'))
    save_csv(base/'paired_development_differences.csv',paired)
    # Independently rerun a fixed random sample of certified inactive-threshold reuses.
    validation_path=base/'reuse_validation.json'
    if validation_path.exists():
        validation=json.loads(validation_path.read_text())
        if validation['source_sha256']!=p['source_sha256']:raise ValueError('Validation source changed')
    else:
        rng=np.random.default_rng(20261103);indices=rng.choice(len(reused),size=min(p['reuse_validation_sample'],len(reused)),replace=False)
        selected=[reused[int(i)] for i in indices];validation=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
            initargs=(manifest['physical_config'],manifest['dt_s'],manifest['max_time_s'])) as pool:
            futures=[pool.submit(episode,byid[row['scenario_id']],config['vector'],config['threshold_s'],config['policy'])
                     for config,row in selected]
            for (config,row),future in zip(selected,futures):
                direct=future.result()
                for field in direct:
                    if field not in ['execution_kind','episode_cpu_s','episode_wall_s'] and direct[field]!=row[field]:
                        raise AssertionError(('Inactive-threshold reuse differs',field,row['scenario_id']))
                validation.append(dict(config=config,direct=direct,matched=True))
        validation=dict(source_sha256=p['source_sha256'],checked=len(validation),rows=validation)
        atomic_durable(validation_path,validation)
    audit=dict(complete=True,optimizer_seeds=[p['optimizer_seed']],objective_calls=len(trials),
        charged_training_episodes=len(trials)*48,distinct_training_scenarios=48,distinct_development_scenarios=48,
        retained_cache_episode_records=count,physical_episode_records=len(physical),certified_reused_records=len(reused),
        completed_cache_records=sum(r['status']=='completed' for r in physical)+sum(r['status']=='completed' for _,r in reused),
        safety_checks_passed=all(r['safety_checks_passed'] for r in physical),
        reuse_validation_episodes=validation['checked'],cpu_seconds_recorded=sum(r['episode_cpu_s'] for r in physical),
        final_test_evaluated=False,teammate_commit=p['teammate_commit'],
        limits=['One optimizer seed; 30 calls/fit; pilot only, no convergence or superiority claim.',
                'Fair means overdue-entry priority, not a new BO/PSO optimizer.',
                'Entry wait sums all local-zone queue waits per robot; thresholds apply to each local queue.',
                'Bootstrap intervals are exploratory and unadjusted; development is not final test.',
                'Shared initial candidates were selected on main training data; historical preparation cost is separate.'])
    atomic_durable(base/'audit.json',audit)
    lines=['# 팀원 FairBO·FairPSO 방식의 다중 교차로 예비 학습','',
           '동일 가중치에 보호 규칙을 추가한 비교와, 보호 규칙을 켠 상태에서 재학습한 비교를 분리했다. 최종 테스트는 사용하지 않았다.','',
           '## 설정','',
           '학습 48개·개발 48개, 다중 교차로, 탐색 시드 101 하나, 방법별 30회 평가, 같은 초기 여섯 후보를 사용했다. 무작위 탐색도 같은 예산의 대조군이다. 물리 조건·진입 가능 후보·안전 검사는 기존과 같다.','',
           '팀원 원본의 FairBO/FairPSO는 BO/PSO가 찾은 기존 가중치에 20초·40초 장기 대기 우선권을 더하는 방식이다. 본 실험은 그 동일 가중치 비교를 재현하고 보호 상태 재학습을 추가했다. 팀원 별도 Improved 학습의 최대 대기 벌점은 섞지 않았다. 최적화기는 우리 기존 GP-LCB/투영 PSO를 고정하여 규칙 효과를 분리했으므로 팀원 저장소 전체 실행을 그대로 재현한 결과는 아니다.','',
           '## 개발 자료 평균','',
           '최대 대기 열은 각 시나리오에서 차량별 누적 구역 진입 대기의 최댓값을 구한 뒤 48개 시나리오에서 평균한 수치다. 일반 통로의 정지 대기는 별도로 저장했다.','',
           '| 정책 | 평균 완료시간(초) | 평균 시나리오 최대 진입대기(초) | 선택 변경 횟수 |','|---|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['policy']} | {row['mean_makespan_s']:.2f} | {row['mean_scenario_max_entry_wait_s']:.2f} | {row['override_admissions']} |")
    lines.extend(['','plain_retrained는 보호 없이 학습, sameweights는 해당 plain 가중치를 그대로 두고 규칙만 추가, fair20/fair40_retrained는 보호 규칙을 켜고 다시 학습한 모델이다. 초기 최선이 FCFS일 수 있으므로 학습 모델이 항상 다른 가중치를 택했다고 가정하지 않는다.','',
                  '## 검증과 해석 범위','',
                  f"예산상 목적함수 호출 {len(trials)}회. 완료된 캐시 기록 {count}개 중 직접 시뮬레이션 {len(physical)}개, 보호 임계값에 도달할 수 없음을 확인한 재사용 {len(reused)}개다. 재사용 {validation['checked']}개를 직접 재실행해 지표·궤적 일치를 확인했다.",'',
                  '재사용 근거: 완료된 무보호 궤적에서 모든 차량의 누적 진입 대기가 T보다 작으면, 각 지역 대기는 그 누적값 이하이므로 보호 T가 어느 시점에도 개입할 수 없다. 경계에 충분한 여유가 있을 때만 사용하며 미완료 궤적은 재사용하지 않는다. 이 비용 절감과 별개로 모든 방법의 초기 여섯 관측은 30회 예산 안에 포함한다.','',
                  '이전 초기값 보정 및 예비 학습 비용은 공통 선행 비용이다. 이번 방법별 30회가 처음부터 모든 초기값을 생성하는 전체 비용이라는 뜻은 아니다. 반복 정책 평가나 재사용을 독립 시나리오 수에 더하지 않는다.','',
                  '한 시드·제한된 탐색 횟수의 개발 실험이다. 평균 완료시간과 긴 대기의 상충을 함께 보며, 대기만 줄었다고 전체 성능이 개선됐다고 부르지 않는다. 보호 규칙은 안전상 진입 불가능한 AGV를 강제로 보내지 않고, 이 임계값 자체가 최대 대기 상한을 보장하지도 않는다.','',
                  '## 출처','',f"팀원 고정 커밋: {p['teammate_commit']}. analyze_step103_holdout.py의 FairBOModel 및 train_realistic_bo_pso.py를 검토했다. review.json과 사전 protocol.json에 출처·설정·해시를 기록했다. 사용자 저장소만 변경했다."])
    (ROOT/'docs/agv_timed_fair_pilot_results_20261103.md').write_text('\n'.join(lines)+'\n')
    archive={str(path.relative_to(base)):path.read_text() for split in ['train','development']
             for path in sorted((base/split).rglob('*.json'))}
    blob=gzip.compress(json.dumps(archive,separators=(',',':')).encode(),mtime=0)
    (base/'raw_records.json.gz').write_bytes(blob)
    atomic_durable(base/'archive_integrity.json',dict(sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),files=len(archive)))
    print(json.dumps(audit,indent=2))
    for row in summary:print(row['policy'],round(row['mean_makespan_s'],3),round(row['mean_scenario_max_entry_wait_s'],3),row['override_admissions'])


if __name__=='__main__':main()
