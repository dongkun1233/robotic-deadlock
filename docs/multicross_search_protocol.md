# Multicross search: reproducible training stage

Base source commit: `4f309b1358d5797bad5e09a788743aa6d109db8f`.

The 2x3 warehouse-style layout uses six repeated 2x2 conflict cores. This is a
discrete traffic-control simulation, not a calibrated model of warehouse
picking, service, acceleration, or rotation duration. Shelf cells are obstacles;
missions travel from perimeter input gates to perimeter output gates.

## Fixed scenarios and splits

The existing scenario manifest is
`outputs/multicross_scenarios_f4Z7B0/scenario_manifest.json`.
It contains 80 training, 40 validation, and 80 test scenarios, with equal counts
per condition: 24/48 vehicles crossed with release windows of 20/0 steps.
The explicit missions, routes, and planned releases are recorded in the
manifest. The replayed manifest is byte-identical to the uploaded manifest.

The search runner executes **training scenarios only**. Reading definitions
of the other splits is used solely to check duplicate hashes; it does not
simulate those splits. Do not select weights on test outcomes.

## Common traffic and feature definitions

All policies share the same deterministic mission generator, fixed shortest
paths, node-move timing, occupied-cell movement restriction, and reservation
of the core path plus first downstream cell. Capacity is two compatible
reservations per zone. Both FCFS and the heuristic hold when their first-ranked
candidate is blocked. Vehicle IDs resolve ties deterministically.

Features are computed per local intersection, not per final destination:

- waiting: steps since reaching the current zone's waiting point;
- maneuver: left=0, straight=0.5, right=1;
- exit competition: other queued candidates sharing the first downstream cell;
- path conflict: other queued candidates sharing any local core cell.

Waiting, exit competition, and path conflict are divided by the maximum among
current candidates, with denominator at least one. This is max scaling, not
general min-max scaling. The weighted score is maximized.

For exactly two candidates, both exit-competition counts are equal and both
path-conflict counts are equal. These two variables therefore cannot change
their pairwise ordering under the current definitions. Feature diversity
is not evidence of performance benefit; ablation is still required.

## Equal-budget search

The first training comparison uses the full four-feature score:

- algorithms: Basic random search, GP-LCB BO, projected PSO;
- optimizer seeds: 101, 202, 303;
- objective calls: 24 per algorithm and seed;
- first six unit-norm weight vectors: identical across the three algorithms
  for each optimizer seed;
- each objective call: all 80 training scenarios;
- BO: RBF length scale 0.45, LCB kappa 1.96, 256 normalized candidates with
  65% global and 35% local proposals;
- PSO: swarm six, inertia decreases from 0.7 toward 0.4, cognitive/social
  coefficients 1.4, velocity clamp +/-0.35, position projection to the unit
  sphere after clipping to [-1,1].

BO predicts at the normalized candidates actually evaluated. These are
documented search implementations; they are not a claim to reproduce the
historical presentation's optimization settings.

Objective = number of incomplete scenarios * 2001 + mean simulation steps.
With a 2000-step horizon this prioritizes fewer incomplete scenarios over
mean steps. Completed and incomplete outcomes are saved separately. Do not
interpret a timeout's horizon as an observed completion time.

Wall-clock time and repeated candidates are recorded. Equal calls are not
equal wall-clock budgets. The three optimizer seeds share the same scenario
set and must not be treated as additional independent scenarios.

## Run and resume

```bash
python -m unittest discover -s tests -p test_multicross_search.py -v

python -u scripts/run_multicross_search.py \
  --manifest outputs/multicross_scenarios_f4Z7B0/scenario_manifest.json \
  --output-dir outputs/multicross_search_train_v1 \
  --budget 24 --initial-count 6 --seeds 101,202,303
```

Re-running the identical command replays optimizer state and reuses completed
evaluations from the atomic checkpoint. Changed source/configuration is
rejected. Keep an output directory tied to one protocol.

Each evaluation writes its 80 episode records before checkpointing. The
consolidated compressed CSV preserves algorithm/model/seed/evaluation IDs.
Inspect incomplete runs and distributions alongside average steps and waits.
The current wait statistic counts stationary steps after spawning; planned
release-to-spawn delay is reported separately. At a timeout, unfinished
vehicles' waits are observed only up to the horizon.

## Next stages

1. Inspect training convergence and unfinished runs without opening test.
2. Use validation to decide search settings/budget if needed.
3. Apply removal masks, force removed weights to zero, and reoptimize.
4. Compare paired scenarios and waiting/fairness outcomes, including uncertainty.
5. Fix the protocol before final test, and report unsuccessful outcomes too.

The existing 24-evaluation comparison is a development-stage training result,
not evidence of test generalization or a universal optimizer ranking.
