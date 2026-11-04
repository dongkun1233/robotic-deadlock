# AGV 독립 테스트: 고정 계획 및 실행 안내

## 고정 내용

- single 804개, double 792개, multi 456개: 총 2,052개의 새로운 시나리오.
- 각 레이아웃에서 12개 조건 비중을 동일하게 유지(각 조건 67/66/38회).
- 레이아웃당 19개 정책: 두 FCFS, 고정4변수, 학습기준 초기 최선, basic/BO/PSO 각 5개 고정 시드. 총 38,988개 에피소드.
- 기존 45개 학습 결과의 가중치를 변경하거나 최종 결과로 시드를 선택하지 않는다.
- 이전 물리 모델 예비/공학 검증/본 학습/개발의 414개 고유 미션 해시 및 기존 시드와 중복을 검사했다. 충돌 발생 시 새 시나리오로 바꿔 끼우지 않고 중단한다.
- 물리 조건, 0.01초 간격, 제한시간 1200초를 유지한다.
- 최종 시나리오와 분석 코드를 첫 평가 전에 해시로 고정한다. 원래 연구 소스·학습 결과·초기 가중치·제외 목록 해시도 실행/재개 시 검사한다.

## 통계와 주장 범위

고정된 5개 학습 정책의 결과를 같은 시나리오 안에서 먼저 평균한다. 5개 시드의 반복 결과를 독립 시나리오로 취급하지 않는다. 15개 주 비교는 3레이아웃 × (basic/BO/PSO 대 feasible-first FCFS 3개, BO/PSO 대 basic 2개)다. 조건별 쌍체 차이의 평균과 분산을 사용한 Welch–Satterthwaite t 근사 및 양측 p값에 Holm 보정(FWER .05)을 적용한다. 15개에 대한 Bonferroni 동시 구간도 별도로 제공한다. 이는 분포 가정 아래 근사 추론이며 임의 분포에서 오류율을 정확히 보장하지 않는다.

Strict FCFS, 초기 최선, 고정4변수 비교는 보조 기술통계로 보고한다. 기존 종설 전체 구현의 동일성은 아직 미확인이다. 새 학습 자료나 새로운 최적화 시드 전체로 일반화하는 검정은 아니다.

실패 에피소드를 삭제하지 않는다. 레이아웃 표본 수 N에서 시나리오별 점수는 capped makespan + (1200+1)*N*실패표시다. 그 평균은 기존 목적함수인 실패 시나리오 수*1201 + 평균 완료시간과 같다. 실패가 있으면 패널티와 완료율을 함께 보고하고 큰 패널티가 지배하는 비교를 조심스럽게 해석한다. 표본 수 계획은 개발에서 완료된 자료의 분산에 근거하며 실패가 발생하면 계획한 검정력이 유지된다는 보장은 없다.

## 준비 검증

완료된 본 학습 백업과 Python 3.12 가상환경이 있는 agv-resume 폴더에서 실행한다. 원자료 압축파일을 GitHub에 업로드하지 않는다. requirements-final.txt는 원래 requirements.txt에 추가로 설치하며 scipy==1.17.0만 추가한다.

```bash
~/.local/bin/uv pip install --python .agv312/bin/python -r requirements-final.txt
.agv312/bin/python -m unittest tests.test_agv_final_protocol -v
.agv312/bin/python scripts/run_agv_final_test.py --phase prepare
```

prepare는 시나리오 생성과 계획 고정만 수행하고 시뮬레이션은 실행하지 않는다. 예상 final manifest SHA256은 아래 별도 잠금 문서에 저장했다. SkylakeX와 Haswell 설정에서 동일한 manifest 해시를 확인했다. 다르면 실행하지 말고 원인을 확인한다.

## 실제 실행 및 재개

```bash
set -o pipefail
.agv312/bin/python -u scripts/run_agv_final_test.py --phase run --workers 2 2>&1 | tee -a final_run.log
```

10개 에피소드마다 진행 상황을 출력한다. 각 에피소드를 원자적 쓰기·fsync로 저장한다. 중단 후 같은 명령을 다시 실행하면 저장된 에피소드의 실험 해시·시나리오 ID·미션 해시·정책·안전 지표를 검사하고 재사용한다. 무조건 재실행하거나 결과를 덮어쓰지 않는다. 중복 실행은 잠금으로 거부한다. 메모리/자체 정지 문제까지 방지하는 기능은 아니다.

```bash
.agv312/bin/python scripts/run_agv_final_test.py --phase status
```

완료 후 다음 명령으로 고정된 분석을 실행한다.

```bash
.agv312/bin/python scripts/analyze_agv_final_test.py
```

FINAL_EVALUATION_COMPLETE 38988 및 status.json의 complete=true를 확인해야 평가 완료다. 분석 완료는 FINAL_ANALYSIS_COMPLETE와 analysis 폴더 결과를 별도로 확인한다. 출력 폴더를 삭제하거나 이전 백업을 덮어 풀지 않는다. 새 결과를 확인한 뒤 코드·가중치·정책·표본 수를 유리하게 변경하지 않는다.

## 현재 상태

계획·코드·시나리오 구성을 검증했다. 준비 시점에는 독립 테스트 에피소드 0개이며, 실제 실행은 사용자의 Codespaces에서 수행한다. 프로토콜 변경이 필요한 코드 오류가 발견되면 기존 기록과 이유를 보존하고 조용히 수정하여 같은 실험으로 이어가지 않는다.
