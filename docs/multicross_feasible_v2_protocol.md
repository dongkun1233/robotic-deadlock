# Feasible admission v2: protocol and results

The v1 module and historical output folders are preserved. The v2 module adds one rule: rank only candidates whose entire local reservation segment is currently free of occupancy and reservations. The reservation segment is the existing intersection core path plus its first downstream cell. Raw feature values and scaling are computed over ALL queued candidates before filtering. FCFS chooses the first feasible arrival; weighted ranking chooses the highest score among feasible candidates. Both have the same feasibility mask. Global zone iteration order is unchanged, so this experiment does not remove any zone-order effects.

`admission_rule='strict_first'` in v2 reproduces v1 movement and decision behavior and exists for regression checks. The default is `feasible_first`. There is no global deadlock-freedom or starvation-freedom theorem. Finite successful simulations provide empirical evidence only. The layout is still a node-move repeated-intersection prototype: no rotation duration, acceleration, physical vehicle footprint, shelf service, continuous replenishment, or real warehouse calibration. Steps are not seconds.

The exact 200-scenario manifest from the original sync branch is used. Local path `outputs/fixed_multicross_suite/scenario_manifest.json` is a reconstruction with Git blob SHA `6df52672bd65993dcb29f80eca7525e3dddcc969`; the remote source is `outputs/multicross_scenarios_f4Z7B0/scenario_manifest.json` on the original sync branch. All simulated missions are checked against their saved digest. Training: 80; validation: 40; final test: 80. The final test has NOT been simulated.

The same validation set was already used to diagnose v1, so subsequent model development treats it as a development/validation set, not a fresh independent final test. No claim that repeated use preserves independence is made.

## Controlled checks

- 10 strict-mode regression runs across FCFS and weighted policies preserve v1 per-robot trajectories, feature values, scaled scores and decisions (excluding v2's eligibility metadata).
- The previously timed-out 23 training evaluation episodes all complete under v2 with the SAME weights and missions. These are 23 recorded evaluations, not 23 unique scenarios; they involve three traffic scenarios.
- The nine v1 training-selected weight sets plus FCFS were rerun on all 80 training cases (800 episodes): zero timeouts. The changed rule is isolated from reoptimization in this comparison.
- Both policies also passed 40 extra 48-vehicle/simultaneous-release seed checks with the simulator's collision/reservation assertions active. No general guarantee is inferred.
- Six unit/regression tests pass. The known HOL failure test is self-contained (train_c4_001, seed 703001, 48 vehicles, release window zero).

## Search and validation

Random search (label `basic`), the existing GP-LCB BO implementation, and the existing projected PSO each use 24 evaluations per optimizer seed, three seeds (101, 202, 303), and the same six initial vectors within each seed. Full four-variable model only. The numerical definitions, objective, budgets and replay/resume behavior are unchanged from the v1 protocol. Reoptimization uses training only.

216 objective evaluations × 80 scenarios = 17,280 episode rows, plus FCFS's 80 training episodes. Audit recomputes every objective and confirms shared initialization, equal budgets, selected-weight consistency and unique keys. All 17,280 v2 candidate episodes completed; none were removed. All raw objective episodes are in the gzip CSV archive and can be restored by the existing audit command. Results should not be interpreted as converged global optima.

| Policy | Mean best training steps (three optimizer seeds where applicable) | Validation mean steps | Paired validation difference vs v2 FCFS, 95% bootstrap CI |
|---|---:|---:|---|
| FCFS, first feasible arrival | 84.1750 | 84.7000 | baseline |
| Random search | 83.6458 | 84.4917 | -0.2083 [-0.9000, +0.4500] |
| BO | 83.8292 | 85.1583 | +0.4583 [-0.0833, +0.9917] |
| PSO | 83.9583 | 85.7417 | +1.0417 [-0.0252, +2.0417] |

All 400 validation episodes complete, with zero duplicate policy/seed/scenario keys. The same 40 unique validation scenarios are used for all policies. Bootstrap averages the three optimizer seeds within each scenario, then resamples scenario pairs within each of the four traffic conditions (5,000 draws). Intervals are conditional on those fitted seeds and do not fully estimate optimizer uncertainty; no multiple-comparison correction is applied at this exploratory stage. All intervals include zero. This does not establish optimization superiority, inferiority or statistical equivalence. PSO seed 202 is visibly worse than its other two seeds, so reporting only a favorable seed would be misleading.

v1 FCFS on these same training/validation cases averaged 87.0375/88.7000 steps, versus v2's 84.1750/84.7000. This observed paired rule change should not be presented as BO's gain or extrapolated to real warehouses.

## Reproduction

From a complete repository checkout with a usable Python environment and numpy:

```bash
python -m unittest discover -s tests -v
python scripts/compare_multicross_admission.py --manifest outputs/multicross_scenarios_f4Z7B0/scenario_manifest.json --old-training-dir outputs/multicross_search_train_v1 --failures outputs/multicross_timeout_diagnostics_v1_complete/failed_training_episodes.csv --output-dir outputs/new_admission_comparison
python scripts/run_multicross_search.py --manifest outputs/multicross_scenarios_f4Z7B0/scenario_manifest.json --output-dir outputs/new_search_v2 --simulator v2
python scripts/audit_multicross_training.py outputs/new_search_v2
python scripts/run_multicross_validation.py --manifest outputs/multicross_scenarios_f4Z7B0/scenario_manifest.json --training-dir outputs/new_search_v2 --output-dir outputs/new_validation_v2 --simulator v2
```

For resuming saved runs, source, manifest bytes, Python/numpy versions and configuration must match the checkpoint exactly. A fresh rerun in a different environment must use a new output directory. Recorded timings describe this execution environment only. The local partial checkout uses an empty package initializer; the repository's original `rware/__init__.py` is not changed. A full checkout may therefore also require the package's original dependencies when importing `rware`.

## Next experiments

Keep v2 fixed while testing the four leave-one-feature-out models, with reoptimization and equal objective budgets. This tests marginal utility given the remaining features. Separately, a small rule × feature-model comparison can measure whether variable utility changes between v1 and v2. New downstream-occupancy, approach-pressure, aging and compatible-pair ideas remain proposals in `multicross_literature_review_20261031.md`; none was silently inserted in this run. Final test stays closed until definitions, budgets and analysis are frozen.
