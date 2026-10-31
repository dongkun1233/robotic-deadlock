import csv
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# 프로젝트 루트를 import 경로에 추가
project_dir = Path.cwd()
sys.path.insert(0, str(project_dir))

import scripts.run_total_time_bo_policy_comparison as comparison
from scripts.run_transfer_generalization_train import _load_scenarios
from scripts.run_transfer_generalization_test import (
    _load_scenarios as load_test_scenarios,
    _run_policy,
)

run_dir = Path(os.environ["AGV_RUN_DIR"])

train_path = Path(
    "final_output/basic_one_factor_sensitivity/train_scenarios_400.csv"
)

test_path = (
    run_dir
    / "test_scenarios"
    / "transfer_test_scenarios_all_layouts.csv"
)

FEATURES = (
    "waiting",
    "maneuver",
    "exit_competition",
    "path_conflict",
)

MODELS = {
    "full": FEATURES,
    "without_waiting": tuple(
        f for f in FEATURES if f != "waiting"
    ),
    "without_maneuver": tuple(
        f for f in FEATURES if f != "maneuver"
    ),
    "without_exit_competition": tuple(
        f for f in FEATURES if f != "exit_competition"
    ),
    "without_path_conflict": tuple(
        f for f in FEATURES if f != "path_conflict"
    ),
}

SEEDS = (101, 202, 303)
EVALUATIONS = 12
INITIAL_EVALUATIONS = 4
CANDIDATE_COUNT = 256

settings = SimpleNamespace(
    max_steps=1500,
    corridor_length=8,
    west_exit_extension=0,
    admission_window_steps=2,
    shared_area_capacity=2,
)


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        vector = np.ones(len(vector), dtype=float)
        norm = np.linalg.norm(vector)

    return vector / norm


def make_weights(active_features, vector):
    # 모든 변수를 먼저 0으로 설정해 제거 대상이 기본값으로
    # 되살아나는 일을 방지합니다.
    weights = {feature: 0.0 for feature in FEATURES}

    for feature, value in zip(active_features, vector):
        weights[feature] = float(value)

    weights["approach_queue"] = 0.0
    weights["remaining_path"] = 0.0
    weights["same_direction_backlog"] = 0.0

    return weights


def training_result(train_scenarios, weights):
    times = []
    incomplete = 0

    for scenario in train_scenarios:
        result = comparison._run_single_policy(
            scenario=scenario,
            policy="bo_heuristic",
            weights=weights,
            max_steps=settings.max_steps,
            corridor_length=settings.corridor_length,
            west_exit_extension=settings.west_exit_extension,
            admission_window_steps=settings.admission_window_steps,
            shared_area_capacity=settings.shared_area_capacity,
        )

        times.append(result["total_time"])

        if result["completed"] != result["robots"]:
            incomplete += 1

    return incomplete, statistics.mean(times)


def choose_bo_candidate(vectors, objectives, rng, dimension):
    # 후보를 먼저 정규화하고, 실제 평가할 후보를 기준으로
    # GP 예측을 수행합니다.
    candidates = np.array([
        normalize(rng.uniform(-1.0, 1.0, dimension))
        for _ in range(CANDIDATE_COUNT)
    ])

    train_x = (np.array(vectors) + 1.0) / 2.0
    candidate_x = (candidates + 1.0) / 2.0

    mu, sigma = comparison._gp_predict(
        train_x=train_x,
        train_y=np.array(objectives),
        test_x=candidate_x,
        length_scale=0.45,
        noise=1e-6,
    )

    lcb = mu - 1.96 * sigma
    return candidates[int(np.argmin(lcb))]


def append_row(path, fieldnames, row):
    file_exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    if not test_path.exists():
        raise FileNotFoundError(f"테스트 파일이 없습니다: {test_path}")

    train_scenarios = _load_scenarios(train_path)
    test_scenarios = load_test_scenarios(test_path)

    train_seeds = {scenario.seed for scenario in train_scenarios}
    test_seeds = {scenario.seed for scenario in test_scenarios}

    if train_seeds & test_seeds:
        raise ValueError("학습·테스트 시나리오에 중복 시드가 있습니다.")

    metadata = {
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_count": len(train_scenarios),
        "test_count": len(test_scenarios),
        "seeds": list(SEEDS),
        "evaluations_per_model_per_seed": EVALUATIONS,
        "initial_random_evaluations": INITIAL_EVALUATIONS,
        "candidate_count": CANDIDATE_COUNT,
        "settings": vars(settings),
        "note": (
            "Exploratory independent test. "
            "Keep results separate from final confirmatory evaluation."
        ),
    }

    with (run_dir / "settings.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    trial_fields = [
        "model", "seed", "evaluation", "source",
        "incomplete", "mean_steps", "objective", "weights_json",
    ]

    best_fields = [
        "model", "seed", "train_incomplete",
        "train_mean_steps", "weights_json",
    ]

    test_fields = [
        "model", "seed", "layout", "case_id", "run_index",
        "scenario_seed", "total_agvs", "completed", "total_time",
    ]

    trials_path = run_dir / "train_trials.csv"
    best_path = run_dir / "best_weights.csv"
    test_results_path = run_dir / "test_raw_runs.csv"

    started = time.perf_counter()
    selected_weights = {}

    print("\n[실험 시작]", flush=True)
    print("학습 시나리오:", len(train_scenarios), flush=True)
    print("테스트 시나리오:", len(test_scenarios), flush=True)
    print("결과 폴더:", run_dir.resolve(), flush=True)

    for seed in SEEDS:
        for model_name, active_features in MODELS.items():
            print(
                f"\n시드 {seed} / {model_name} 탐색 시작",
                flush=True,
            )

            rng = np.random.default_rng(seed)
            vectors = []
            objectives = []
            best_key = None
            best_weights = None

            for evaluation in range(EVALUATIONS):
                if evaluation < INITIAL_EVALUATIONS:
                    vector = normalize(
                        rng.uniform(-1.0, 1.0, len(active_features))
                    )
                    source = "random_initial"
                else:
                    vector = choose_bo_candidate(
                        vectors,
                        objectives,
                        rng,
                        len(active_features),
                    )
                    source = "bo_lcb"

                weights = make_weights(active_features, vector)
                incomplete, mean_steps = training_result(
                    train_scenarios,
                    weights,
                )

                objective = mean_steps + incomplete * settings.max_steps

                vectors.append(vector)
                objectives.append(objective)

                key = (incomplete, mean_steps)

                if best_key is None or key < best_key:
                    best_key = key
                    best_weights = weights.copy()

                append_row(trials_path, trial_fields, {
                    "model": model_name,
                    "seed": seed,
                    "evaluation": evaluation + 1,
                    "source": source,
                    "incomplete": incomplete,
                    "mean_steps": mean_steps,
                    "objective": objective,
                    "weights_json": json.dumps(weights, sort_keys=True),
                })

                print(
                    f"  {evaluation + 1}/{EVALUATIONS}: "
                    f"미완료 {incomplete}, "
                    f"평균 {mean_steps:.2f}단계",
                    flush=True,
                )

            selected_weights[(seed, model_name)] = best_weights

            append_row(best_path, best_fields, {
                "model": model_name,
                "seed": seed,
                "train_incomplete": best_key[0],
                "train_mean_steps": best_key[1],
                "weights_json": json.dumps(best_weights, sort_keys=True),
            })

    print("\n[별도 테스트 시작]", flush=True)

    # FCFS는 학습 시드와 무관하므로 시나리오별 1회 실행
    for scenario in test_scenarios:
        result = _run_policy(
            scenario=scenario,
            policy="fcfs",
            weights=None,
            args=settings,
        )

        append_row(test_results_path, test_fields, {
            "model": "fcfs",
            "seed": "baseline",
            "layout": scenario.layout_key,
            "case_id": scenario.case_id,
            "run_index": scenario.run_index,
            "scenario_seed": scenario.seed,
            "total_agvs": scenario.total_agvs,
            "completed": result["completed"],
            "total_time": result["total_time"],
        })

    for (seed, model_name), weights in selected_weights.items():
        print(
            f"  시드 {seed} / {model_name} 테스트 중",
            flush=True,
        )

        for scenario in test_scenarios:
            result = _run_policy(
                scenario=scenario,
                policy="heuristic",
                weights=weights,
                args=settings,
            )

            append_row(test_results_path, test_fields, {
                "model": model_name,
                "seed": seed,
                "layout": scenario.layout_key,
                "case_id": scenario.case_id,
                "run_index": scenario.run_index,
                "scenario_seed": scenario.seed,
                "total_agvs": scenario.total_agvs,
                "completed": result["completed"],
                "total_time": result["total_time"],
            })

    print("\n[실험 완료]")
    print("결과 폴더:", run_dir.resolve())
    print("총 실행시간:", round(time.perf_counter() - started, 1), "초")
    print("탐색 기록:", trials_path)
    print("선택 가중치:", best_path)
    print("테스트 원본:", test_results_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[실험 중 오류 발생]", flush=True)
        traceback.print_exc()
        raise
