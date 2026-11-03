"""Archive and describe training-only initial calibration; never evaluate development."""
import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path
from scripts.agv_timed_main_common import atomic_durable, read_manifest, save_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/agv_timed_main_v1/manifest.json'))
    args = parser.parse_args()
    base = args.manifest.parent
    manifest = read_manifest(args.manifest)
    selection = json.loads((base / 'initial_weights.json').read_text())
    if selection['retained_episode_records'] != 3744 or selection['completed_episode_records'] != 3744:
        raise ValueError('This report requires all 3744 calibration episodes to complete; inspect failures first')
    lines = [
        '# 확대 학습 자료의 공통 초기 가중치 보정 (2026-11-03)', '',
        '완료 범위는 새 학습 자료 평가와 초기 여섯 후보 선정이다. 72회×5시드의 후속 검색과 새 개발 평가는 아직 실행하지 않았다. 최종 테스트는 생성·실행하지 않았다.', '',
        '## 자료와 평가 범위', '',
        '- 레이아웃별 학습 48개·개발 48개: 각 144개. 12개 조건에 네 반복씩 배정했다.',
        '- 기존 물리 모델 예비/공학 검증 자료 및 새 학습·개발 간 내용 해시와 시드 중복을 배제했다.',
        '- 초기 후보 24개×48개 학습 시나리오×3개 레이아웃 = 3,456개 평가 기록.',
        '- 엄격 FCFS 및 순수 Waiting 동등성 검증으로 288개 추가 기록. 합계 3,744개 기록이 모두 완료·안전 assertion 통과했다.',
        '- 이는 독립 시나리오 3,744개가 아니다. 서로 다른 학습 시나리오는 144개이며 정책 후보를 반복 적용했다.',
        '- 세 개 체크포인트의 마지막 저장 행을 재실행하여 복구했다. 원래 시도의 시간 기록은 없으므로 CPU 합계에서 제외하며 storage_recovery.json에 남겼다.',
        '- 기존 26개와 새 자료 분리·예산·선정 검사 8개를 합쳐 테스트 34개가 통과했다. 재개 검증에서는 추가 시뮬레이션 없이 같은 초기값이 재생성됐다.', '',
        '## 같은 학습 시나리오에서의 평균 완료 시간 (초)', '',
        '| 레이아웃 | FCFS | 기존 고정 4변수 기준 | 선정 초기 후보 중 최선 | FCFS 대비 감소율 |',
        '|---|---:|---:|---:|---:|',
    ]
    selected_rows = []
    for layout, label in [('single', '단일'), ('double', '이중'), ('multi', '다중')]:
        data = selection['selected_by_layout'][layout]
        items = data['candidates']
        reference = next(c for c in items if c['candidate_id'] == 'fcfs')
        old = next(c for c in items if c['candidate_id'] == 'fixed4_reference')
        best = min(items, key=lambda c:c['objective'])
        improvement = 100 * (reference['objective'] - best['objective']) / reference['objective']
        lines.append(f"| {label} | {reference['objective']:.2f} | {old['objective']:.2f} | {best['objective']:.2f} | {improvement:.2f}% |")
        for position, item in enumerate(items):
            selected_rows.append(dict(layout=layout, initial_position=position, candidate_id=item['candidate_id'],
                                      training_objective=item['objective'], waiting=item['vector'][0],
                                      maneuver=item['vector'][1], exit_competition=item['vector'][2],
                                      path_conflict=item['vector'][3], diversity_fallback_used=data['diversity_fallback_used']))
    lines.extend([
        '', '위 표는 학습 자료에서 후보를 선택한 뒤의 수치다. 일반화 성능이나 알고리즘 우열을 입증하지 않는다. 서로 다른 시나리오였던 이전 개발 결과와 직접 비교해서 개선 폭을 계산하면 안 된다.', '',
        '단일·이중에서는 기존 고정 가중치를 그대로 사용하는 것보다 나은 초기 후보가 있었다. 다중에서는 전체 후보 24개 중 FCFS가 가장 좋았다. FCFS를 유지하고 이후 탐색에서 개선 가능성을 확인한다. 모든 레이아웃에서 BO·PSO가 유리하도록 후보나 규칙을 변경하지 않았다.', '',
        '후보에는 모든 예비 방법/시드의 학습 최선 벡터를 포함했다. 예비 개발 점수로 일부 방법만 선택하지 않았다. 각 레이아웃에서 FCFS·고정 기준과 학습 순위/다양성 규칙으로 정한 네 후보를 사용하며, 다양성 fallback은 발생하지 않았다.', '',
        '가중치 부호·크기만으로 변수 중요도를 판정하지 않는다. 변수 기여는 동일 예산 재학습 ablation으로 검증한다. 단일의 최선 후보 이름에 PSO가 들어 있어도 이는 과거 예비 학습 후보의 출처이며, 이번 72회 PSO 검색의 결과가 아니다.', '',
        '## 고정한 후속 학습 설계', '',
        '- 레이아웃별 동일 초기 여섯 후보를 무작위 탐색·BO·PSO와 다섯 탐색 시드에 공통 제공한다.',
        '- 검색 예산 72회 안에 초기 여섯 관측을 포함한다. 6입자 PSO는 초기화 뒤 11번의 갱신을 수행한다.',
        '- 초기값 보정 24회와 검색 72회의 단계별 호출 합은 96이다. 초기 여섯은 양쪽 단계에서 재사용되므로 고유 후보 평가 수는 추가 중복이 없을 때 최대 90개다. 실제 실행 횟수는 따로 기록한다.',
        '- 과거 예비 학습 후보를 만드는 데 든 공통 역사적 비용 486회/5,832개 예산상 에피소드도 별도 명시한다.',
        '- 새 개발 평가에 warmstart_best 기준을 남겨 공통 초기값 효과와 후속 탐색 효과를 구분한다. 이 추가 대조군은 기술통계용이다.',
        '- 다중에서 개선 후보가 없다는 이유로 진입 규칙·후보 집합·물리 시간 모델을 바꾸지 않는다.',
        '- 다음 단계는 45개 검색 작업이며, 이후 개발 자료로 조건별 편차·탐색 곡선·시드 변동을 확인하고 최종 표본 수를 다시 결정한다.', '',
        '## 원자료와 재현', '',
        '사전 등록 커밋: 8eee32722f1dfd8e7dcf84b35039e24e285197fc. 물리 엔진과 변수 정의는 이전 예비 단계의 해시를 유지했다. 자세한 원자료 복원·재개 방법은 outputs/agv_timed_main_v1/README.md에 있다.', '',
    ])
    report = Path('docs/agv_timed_initial_calibration_results_20261103.md')
    report.write_text('\n'.join(lines))
    save_csv(base / 'selected_initial_weights.csv', selected_rows)
    records = {str(path.relative_to(base)): path.read_text()
               for path in sorted((base / 'calibration').glob('*/*.json'))}
    if len(records) != 78:
        raise ValueError('Expected 78 complete candidate/baseline records')
    blob = gzip.compress(json.dumps(records, separators=(',', ':')).encode(), mtime=0)
    (base / 'calibration_records.json.gz').write_bytes(blob)
    atomic_durable(base / 'calibration_archive_integrity.json', dict(
        filename='calibration_records.json.gz', sha256=hashlib.sha256(blob).hexdigest(),
        archived_files=len(records), retained_episode_records=3744, compressed_bytes=len(blob)))
    print(report)
    print('archived bytes', len(blob))


if __name__ == '__main__':
    main()
