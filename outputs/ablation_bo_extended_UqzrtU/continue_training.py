import csv
import json
import os
import runpy
import time
import traceback
from pathlib import Path

import numpy as np

try:
    run_dir = Path(os.environ["AGV_EXT_DIR"])
    trials_path = run_dir / "train_trials.csv"
    source_script = run_dir / "run_ablation.py"

    # 기존 실험 함수만 불러옵니다.
    # run_ablation.py의 main은 실행하지 않습니다.
    os.environ["AGV_RUN_DIR"] = str(run_dir)
    original = runpy.run_path(str(source_script))

    models = original["MODELS"]
    seeds = original["SEEDS"]
    make_weights = original["make_weights"]
    training_result = original["training_result"]
    choose_bo_candidate = original["choose_bo_candidate"]
    append_row = original["append_row"]

    from scripts.run_transfer_generalization_train import _load_scenarios

    train_path = Path(
        "final_output/basic_one_factor_sensitivity/train_scenarios_400.csv"
    )
    train_scenarios = _load_scenarios(train_path)

    target_evaluations = 24

    trial_fields = [
        "model",
        "seed",
        "evaluation",
        "source",
        "incomplete",
        "mean_steps",
        "objective",
        "weights_json",
    ]

    best_fields = [
        "model",
        "seed",
        "train_incomplete",
        "train_mean_steps",
        "weights_json",
    ]

    best_path = run_dir / "best_weights_extended.csv"
    started = time.perf_counter()

    print("\n[추가 BO 탐색 시작]", flush=True)
    print("학습 시나리오:", len(train_scenarios), flush=True)
    print("모델·시드별 목표:", target_evaluations, "회", flush=True)
    print("결과 폴더:", run_dir.resolve(), flush=True)

    for seed in seeds:
        for model_name, active_features in models.items():
            with trials_path.open(
                encoding="utf-8", newline=""
            ) as file:
                all_rows = list(csv.DictReader(file))

            matching = [
                row for row in all_rows
                if row["model"] == model_name
                and int(row["seed"]) == seed
            ]

            matching.sort(key=lambda row: int(row["evaluation"]))

            expected = list(range(1, len(matching) + 1))
            actual = [int(row["evaluation"]) for row in matching]

            if actual != expected:
                raise ValueError(
                    f"{model_name}, 시드 {seed}: "
                    "기존 평가 번호가 연속적이지 않습니다."
                )

            vectors = []
            objectives = []

            for row in matching:
                weights = json.loads(row["weights_json"])

                vectors.append(
                    np.array(
                        [weights[f] for f in active_features],
                        dtype=float,
                    )
                )
                objectives.append(float(row["objective"]))

            if len(vectors) < 4:
                raise ValueError(
                    f"{model_name}, 시드 {seed}: "
                    "기존 탐색 기록이 부족합니다."
                )

            # 이어서 탐색할 때 별도의 재현 가능한 난수 시드 사용
            rng = np.random.default_rng(seed + 100000)

            print(
                f"\n시드 {seed} / {model_name}: "
                f"{len(matching)}회 완료 상태",
                flush=True,
            )

            for evaluation in range(
                len(matching),
                target_evaluations,
            ):
                vector = choose_bo_candidate(
                    vectors,
                    objectives,
                    rng,
                    len(active_features),
                )

                weights = make_weights(active_features, vector)
                incomplete, mean_steps = training_result(
                    train_scenarios,
                    weights,
                )

                objective = mean_steps + incomplete * 1500

                vectors.append(vector)
                objectives.append(objective)

                append_row(
                    trials_path,
                    trial_fields,
                    {
                        "model": model_name,
                        "seed": seed,
                        "evaluation": evaluation + 1,
                        "source": "bo_lcb_continued",
                        "incomplete": incomplete,
                        "mean_steps": mean_steps,
                        "objective": objective,
                        "weights_json": json.dumps(
                            weights,
                            sort_keys=True,
                        ),
                    },
                )

                print(
                    f"  {evaluation + 1}/{target_evaluations}: "
                    f"미완료 {incomplete}, "
                    f"평균 {mean_steps:.2f}단계",
                    flush=True,
                )

            # 기존 12회와 추가 12회 중 학습 결과가 가장 좋은 후보 선택
            with trials_path.open(
                encoding="utf-8", newline=""
            ) as file:
                updated_rows = list(csv.DictReader(file))

            updated = [
                row for row in updated_rows
                if row["model"] == model_name
                and int(row["seed"]) == seed
            ]

            best = min(
                updated,
                key=lambda row: (
                    int(row["incomplete"]),
                    float(row["mean_steps"]),
                ),
            )

            append_row(
                best_path,
                best_fields,
                {
                    "model": model_name,
                    "seed": seed,
                    "train_incomplete": best["incomplete"],
                    "train_mean_steps": best["mean_steps"],
                    "weights_json": best["weights_json"],
                },
            )

            print(
                f"  선택된 학습 평균: "
                f'{float(best["mean_steps"]):.2f}단계',
                flush=True,
            )

    print("\n[추가 탐색 완료]")
    print("결과 폴더:", run_dir.resolve())
    print("총 실행시간:", round(time.perf_counter() - started, 1), "초")
    print("전체 탐색 기록:", trials_path)
    print("선택 가중치:", best_path)
    print("\n※ 아직 추가 탐색 모델의 테스트는 실행하지 않았습니다.")

except Exception:
    print("\n[추가 탐색 중 오류]")
    traceback.print_exc()
    raise
