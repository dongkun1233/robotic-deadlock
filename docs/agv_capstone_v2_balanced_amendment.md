# Balanced input and optimizer initialization amendment

Preparation amendment before v2 optimizer fitting. Existing 720-case audit stays
as preliminary engineering evidence; it is not an audit of the new manifest.
No old run is relabeled as a balanced run. No final-test data are generated.

## Direction balance and scope

Round the planned repetitions upward to whole inlet-count multiples separately
for each layout/count/type/split. Active dominant inlet and outlet marginals are
then exactly equal. The inherited one-position offset pairing is retained; this
does NOT exhaust all dominant inlet/outlet pairs and no such generalization is
claimed. Non-dominant remainders are balanced and exact OD quotas are matched.

| Layout | Inlets | Train reps/cell | Train total | Dev reps/cell | Dev total |
|---|---:|---:|---:|---:|---:|
| Single | 4 | 12 | 144 | 8 | 192 |
| Double | 6 | 12 | 144 | 6 | 144 |
| Multi | 10 | 10 | 120 | 10 | 240 |

Total 984 unique train/development scenarios. Counts stay single 20/24/28,
double 24/30/36, multi 20/30/40. Multi is an input-defined traffic-demand range,
not a field-calibrated fleet size or FCFS-defined congestion category.

## Frozen fitting inputs

All layouts/methods/seeds get the same six unit vectors: pure Waiting,
normalized [1,.8,1.2,1.5], four cube-uniform vectors normalized to L2=1 from Python
Random seed 20261105. No outcome-based calibration, previous learned weights, or
development-ranked candidates. Initial observations count inside 72 calls;
identical simulations may be cached but are not independent new observations.
Five optimizer seeds: 101/202/303/404/505. Use the frozen source implementation
in scripts/run_multicross_search.py: Random, GP-LCB (256 candidates, kappa 1.96),
and synchronous six-particle PSO. No claim that 72 calls guarantees convergence.

Fixed dt .01s and provisional engineering-audited timeout 1200s. Objective is
incomplete-scenario count*1201 + mean capped scenario makespan (<=1200); one extra
failed scenario therefore dominates any change in the bounded time term.
Failures, stalls, and timeouts remain in the records. Common FCFS is evaluated
as a separate baseline; pure Waiting equivalence must be checked, not assumed.

Before main fitting, complete the FCFS audit of the exact new manifest and
validate the source hashes and resumable trainer. Estimate full fitting compute
cost from actual Codespaces timings; the one-policy audit duration is not the
duration of 45 optimization jobs. Final sample size and balanced directional
blocks will be frozen separately before any final-test generation.
