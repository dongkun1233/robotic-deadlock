# AGV capstone v2: implementation protocol

Status: accepted research design, preparation in progress. No v2 training or final
test has started. Prior v1 outputs remain separate and are not v2 final evidence.

## Scope and common engine

One predefined trip per AGV; fixed directed routes and planned release times.
Optimize batch makespan from common t=0. No routing, dispatch, loading, battery,
body envelope, or continuous-body safety claim. Grid occupancy, directed-edge
reservation, incompatible shared-zone paths, and capacity <=2 remain common to
all policies. Rotation retains its cell; movement reserves its segment and
destination until completion. Release zone permission only after exit completes.
Implement separately from the existing body-envelope v1 engine.

Cell length 1.5m; speed 0.8m/s; spin omega 1rad/s, angular acceleration 2rad/s^2;
linear acceleration omitted; base tick 0.01s. Validate with a smaller tick.

## Demand and scenarios

Single: 20/24/28; double: 24/30/36 (original generator defaults, not a claim about
the published table's provenance). Multi candidates: 2/3/4 vehicles per inlet.
Finalize from topology and exact quota feasibility, never learned-policy ranking.

S1 balanced; S2 burst; S3 approach skew; S4 exit skew; S5 all three; S6 burst+
approach skew; S7 burst+exit skew; S8 approach+exit skew. Basic release interval
7.5s per inlet. Burst: half randint(0,2)*1.875s, rest randint(3,8)*1.875s.
This is a new scaling assumption, not a reconstruction of original step seconds.
Rotate dominant inlets/outlets across replicates. Preserve exact start and exit
counts through feasible OD matching; reject impossible quota specifications
explicitly rather than silently altering them. Fix shortest-path tie breaking.
Save routes, releases, quotas, seeds, hashes, and last planned release time.

## Policies and fitting

Common-feasible FCFS; frozen fixed4; Random (legacy Basic label); BO; PSO.
Strict FCFS is a separate rule control. All use identical admissibility and ties.
Four features: waiting/maneuver/exit competition/path conflict. Document exact
feature formulas and normalization. Use L2 unit weights including negative values.
One fitted vector per layout across demand cells; five optimizer seeds; initial
budget 72 calls including common initial observations. Freeze initialization
without selecting from final/development outcomes; account for its full cost.
Record noncompletion and minimize makespan second; do not discard failed runs.
Failure penalty must dominate the declared bounded time term, or use a verified
lexicographic implementation. Timeout and budget extension rules must be fixed
before main fits/final tests respectively.

Fair OFF/20s/40s applies to the SAME Random/BO/PSO fitted vectors. Longest local
wait among eligible protected candidates takes priority, otherwise use score.
Record interventions; thresholds do not guarantee a waiting bound. Protected
retraining is optional and is not part of the required main comparison.

Multi-only BO ablation: Full and each removed feature, OFF protection, equal
budgets and five seeds, retrain projected/deduplicated initial candidates. Keep
safety rules. Interpret as BO-specific marginal contribution, not proof that all
four features or specific coefficient values are universally necessary.

## Splits, analysis, release gates

Per layout: train 4 types*3 counts*10=120; development 8*3*5=120; planned final
8*3*30=720. Final size is provisional pending development-based precision review;
do not generate or inspect final scenarios yet. Keep seeds and content disjoint.
Primary evaluation: multi S5-S8, equal demand/type weights; other cells secondary.
Paired scenario comparisons, optimizer-seed variability, uncertainty and multiple
comparison control for confirmatory claims. Never select the best final seed.
Report makespan, completion, post-release/pending delay, mean/p95/max stationary
wait, zone wait, interventions, queue/blocking and compute cost. Respect shared
scenario/seed dependence. Different release patterns are not pure congestion
interventions; compare policies within the same scenario.

Before training: validate isolated v2 engine, route and quota feasibility, exact
multi counts/skew table, timeout, objective, initializer, resumption and duplicate
runner lock. Before final test: freeze all policies/weights, sample size, analysis,
and any budget changes. Publish only actual measurements, including null/adverse
findings. Do not overwrite old results or present v1 runs as v2 confirmation.
