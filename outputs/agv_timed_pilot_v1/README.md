# Physical-time baseline pilot v1

Pre-fit protocol and mission manifest committed as `614ae43116d21f55fe7b6443954931ffa7d8dee8` on `research/physical-time-v1-20261102` before any fitting.

This is a pilot: 36 training scenarios and 72 development scenarios across three layouts. No final-test scenarios are present. All policies share physical timing and feasible-entry rules; strict FCFS is a separate rule comparison. `basic` in trial records means random search; `fixed4_reference` is the projected legacy fixed heuristic.

Resume from repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_agv_timed_pilot
```

The runner retains per-job checkpoints and per-scenario evaluation records. Do not start two runners for the same output directory. Existing completed observations are reused and counted in all corresponding method budgets.

Files: `manifest.json` contains full missions and source hashes; `scenario_index.csv` indexes splits and demand strata. `train_trials.csv` records charged calls. `development_summary.csv` records each fitted policy and baseline separately; do not count repeated optimizer-seed evaluations as new traffic scenarios. `sample_size_sensitivity.csv` is a rough normal-approximation planning calculation, not a final power guarantee. `audit.json` reports executed episodes, timings and limitations.

Raw records are archived in `raw_records.json.gz` when the pilot completes. Restore before resuming in a fresh checkout:

```sh
python - <<'PY'
import gzip,json
from pathlib import Path
root=Path('outputs/agv_timed_pilot_v1')
for name,content in json.loads(gzip.decompress((root/'raw_records.json.gz').read_bytes())).items():
    path=root/name
    if path.is_absolute() or '..' in Path(name).parts:
        raise ValueError('Invalid archive path')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=content:
        raise ValueError(f'Refusing to overwrite different data: {path}')
    path.write_text(content)
PY
```

The archive includes baseline equivalence checks, all training caches/checkpoints/best weights, development runs and logs. Use archive SHA256 from `archive_integrity.json` to verify integrity. An interrupted pilot should be resumed, not described as complete. Engineering safety assertions apply to simulated trajectories under the stated body-only assumptions, not certification of real robots.
