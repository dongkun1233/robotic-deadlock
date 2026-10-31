# Training comparison: Basic / BO / PSO

This is a development-stage **training** result, not a validation or test result.

Base commit: `4f309b1358d5797bad5e09a788743aa6d109db8f`.
Scenario manifest: `outputs/multicross_scenarios_f4Z7B0/scenario_manifest.json`.
Its regenerated Git blob SHA matched `6df52672bd65993dcb29f80eca7525e3dddcc969`
exactly. See `docs/multicross_search_protocol.md` for the full protocol.

## Setup and result

80 fixed training scenarios; 24/48 vehicles; release windows 20/0;
optimizer seeds 101, 202, 303; 24 objective calls per method and seed;
six identical initial candidates across methods. Each call executes all 80
training scenarios. The full four-feature score is optimized here; this is
not the warehouse-layout ablation result yet.

| Method | Best training steps, seed 101 | Seed 202 | Seed 303 | Mean across optimizer seeds |
|---|---:|---:|---:|---:|
| FCFS | 87.0375 | 87.0375 | 87.0375 | 87.0375 |
| Basic random search | 86.3625 | 86.7750 | 86.6500 | 86.5958 |
| BO | 86.5375 | 85.8500 | 86.6500 | 86.3458 |
| PSO | 86.6750 | 88.0125 | 86.5000 | 87.0625 |

The FCFS baseline is executed once, not independently per optimizer seed.
No algorithm wins every seed. The differences are small and do not establish
a reliable optimizer ranking or generalization advantage.

All selected candidates completed all 80 training scenarios. During search,
21 of 216 evaluated candidates had at least one incomplete scenario, totaling
23 incomplete episodes. These outcomes are retained, not discarded. The
objective prioritizes fewer incomplete scenarios before mean horizon/steps.

Total objective evaluations: 216. Training episodes: 17,280, plus 80 FCFS
baseline episodes. Repeated evaluations of the same scenarios are not
independent new scenarios. Candidate simulation time totaled about 148.5 s
in this execution environment; this is not a Codespaces runtime estimate.

## Files

- `train_trials.csv`: all 216 objective evaluations, weights and elapsed times.
- `best_training_weights.csv`: nine candidates selected on training only.
- `fcfs_training_raw.csv`: the 80 baseline episodes.
- `training_episode_results.csv.gz`: all 17,280 candidate episode rows,
  identified by model, optimizer seed, algorithm, evaluation and scenario.
- `checkpoint.json`: deterministic search continuation state and configuration.
- `run_complete.json`: expected-run completion marker.
- `training_summary.csv`: summary of the selected training candidates.
- `audit.json`: record-level checks.

Run `python scripts/audit_multicross_training.py outputs/multicross_search_train_v1`
to verify records. When the per-evaluation `raw/` directory is absent, this
command first reconstructs its 216 CSVs from the compressed archive.

Checks completed: shared initial vectors/objectives; 216 distinct evaluation
keys; 80 distinct training scenarios per evaluation; recomputed episode
objectives; chosen weights are the training minima; unit-test optimizer
interruption/replay; actual completed-checkpoint replay without recomputation;
archive-only checkout restoration and audit.

Validation (40 scenarios) and test (80 scenarios) have **not** been simulated
by this comparison. Next work is convergence/validation and frozen-protocol
ablation, not a claim of completed research.
