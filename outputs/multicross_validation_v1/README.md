# Validation of the full four-feature policy

These are validation results, not final test results. All nine candidates were
selected on the 80 training scenarios before these 40 validation scenarios
were simulated. The 80 final-test scenarios remain unevaluated.

| Algorithm | Seed 101 | Seed 202 | Seed 303 | Mean steps |
|---|---:|---:|---:|---:|
| FCFS (one baseline run) | 88.700 | 88.700 | 88.700 | 88.700 |
| Basic | 88.875 | 89.200 | 89.025 | 89.033 |
| BO | 88.525 | 88.300 | 89.025 | 88.617 |
| PSO | 89.300 | 90.225 | 89.100 | 89.542 |

400 episode records passed mission-replay, count, and duplicate-key checks.
All 400 completed. The three fitted seeds share the same 40 scenarios, so
they are not 120 independent scenarios.

Mean paired differences versus FCFS (negative is better): Basic +0.333,
BO -0.083, PSO +0.842 steps. Stratified scenario-pair bootstrap 95% intervals
were respectively [-0.533,1.142], [-0.783,0.625], and [-0.250,1.925].
These intervals condition on these three fitted models and do not fully
estimate optimizer-seed uncertainty. All include zero; the results do not
establish an improvement over FCFS. They also do not prove equivalence.

The recorded training curves continue improving through later evaluations
for some seeds. Twenty-four evaluations should not be described as proven
convergence. These observations are development feedback, not a justification
to cherry-pick another final test set.

## Prototype limitation found during timeout diagnosis

The v1 simulator deliberately stops admissions when its first-ranked vehicle
cannot reserve its core path plus downstream cell. In three reproduced
training-timeout snapshots all core reservations were empty, but the highest
ranked candidate was blocked and other queued candidates had clear segments.
Thus this head-of-line admission rule can cause persistent avoidable holds.
This is a property of this prototype and policy, not a real-warehouse deadlock
claim. All 23 timeout episodes were preserved.

Next: preserve v1, explicitly compare a version that masks infeasible
candidates before selection for BOTH FCFS and weighted ranking. Keep the
feature definitions unchanged initially, document any candidate-set convention,
and apply identical safety/reservation rules. Do not silently mix v1/v2 results.
Then revisit convergence and reoptimized ablation on the corrected protocol.

Sources and detailed records are in this folder and
`outputs/multicross_timeout_diagnostics_v1_complete`.
