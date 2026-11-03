# Fair priority pilot: completed, one optimizer seed

Read `docs/agv_timed_fair_pilot_results_20261103.md` for interpretation. This pilot uses 48 training and 48 development scenarios, nine fits of 30 evaluations, seed 101. No final test was generated or evaluated. The final retained cache has 12,144 completed records. These are repeated policy evaluations, not 12,144 independent scenarios.

The results do not demonstrate a makespan advantage over FCFS. Same-weight 20-second protection slightly reduces maximum cumulative entry waiting while increasing mean makespan for BO/PSO; 40-second protection does not change the development trajectories of their plain-trained best weights.

`training_trials.csv`, `best_fits.json`, `development_summary.csv`, and `paired_development_differences.csv` contain the numeric results. `raw_records.json.gz` preserves all train and development JSON files, including checkpoints and per-episode cache records. `archive_integrity.json` authenticates the compressed bytes. Restore from repository root:

```sh
python - <<'PY'
import gzip, hashlib, json
from pathlib import Path
base = Path('outputs/agv_timed_fair_pilot_v1')
blob = (base / 'raw_records.json.gz').read_bytes()
assert hashlib.sha256(blob).hexdigest() == json.loads((base / 'archive_integrity.json').read_text())['sha256']
for name, content in json.loads(gzip.decompress(blob)).items():
    rel = Path(name)
    assert not rel.is_absolute() and '..' not in rel.parts
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == content, str(path)
    else:
        path.write_text(content)
print('Verified and restored raw records')
PY
```

The main manifest is restored and verified automatically by `read_manifest`. Initial-calibration raw records have their own archive and restoration instructions under `outputs/agv_timed_main_v1/README.md`.

After restoration, rerun analysis with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.analyze_agv_timed_fair_pilot --workers 8
```

The analysis regenerates its audit and result document; retain the separate recovery records when interpreting CPU costs. `cache_recovery.json` records 38 incomplete cache files and 137 restored rows, with all 270 stored training objectives verified unchanged. `environment_recovery_20261103.json` records the interruption. Original compute cost for lost, unpersisted attempts is unavailable. Retained-record CPU totals exclude that cost.
