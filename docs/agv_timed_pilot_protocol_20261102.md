# Physical-time pilot protocol (fixed before fitting)

Purpose: feasibility, cost and paired-variance estimation, not final algorithm ranking.

All policies use the same physical model, dt=0.01 s, eligibility rules and scenario missions. Strict FCFS is a separate admission-rule baseline. Fixed4 reference projects legacy weights onto four features and is not the original seven-feature Basic. Internal algorithm `basic` means random search.

Three layouts; twelve demand cells per layout: 12/24 robots × release window 0/15 s × uniform/approach-skew/exit-skew demand. One training replicate per cell (36 total) and two development replicates (72 total), new seeds and mission-content deduplication against the preceding engineering checks. No final test is generated or evaluated.

Fit each layout separately. Each method receives 18 objective calls for each of optimizer seeds 101,202,303. Shared six initial observations: FCFS, fixed4 reference, four seeded random unit vectors. Every initial observation counts in each method's budget. Cached observations save computation; report charged calls separately from physically executed episodes. PSO has only two updates after initialization: no convergence claim is justified.

Objective: incomplete-scenario count × 1201 + mean capped makespan (horizon 1200 s). Retain failures. Before caching the FCFS anchor, verify pure waiting reproduces FCFS trajectories on every training case. Preserve raw timings, scenario results, candidates, checkpoints, configuration and source hashes. Development is for planning, never described as independent final confirmation.

Main experiment planning: provisional 48 training cases/layout, 72 calls/method/optimizer seed, at least five optimizer seeds. Determine final scenario count from paired development variance for a predeclared meaningful 1% makespan difference (also show 0.5% and 2% sensitivity), power target 80%, conservative planning alpha 0.05/15 for nine method-vs-FCFS and six BO/PSO-vs-random comparisons across three layouts. Average optimizer seeds within scenario; repeated fits are not independent traffic samples. Round scenario counts up to multiples of twelve with floor 96/layout. Pilot variance is uncertain; report assumptions and cost, reassess on development data before final-test lock, and never silently cap required sample size while claiming adequate power. Main comparisons use Holm correction and paired scenario analysis; inspect optimizer-seed variability separately.

Ablation follows after baseline protocol validation, with equal retraining budgets for each removed-feature model. Do not tune admission rules or initializations against final-test outcomes. Freeze final-test protocol before generation/evaluation. If gains approach numerical-clock sensitivity, run finer-clock checks on the relevant fitted-policy cases.
