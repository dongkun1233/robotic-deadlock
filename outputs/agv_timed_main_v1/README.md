# Expanded physical-time training preparation

Pre-fit protocol/data commit: `8eee32722f1dfd8e7dcf84b35039e24e285197fc`.

New scenarios: 48 train and 48 development per layout; three layouts; twelve strata with four replicates each. No final test. Candidate-bank selection reads only training observations. The physical engine, feature definitions and admission rules are unchanged from the pilot.

`manifest.json.gz` contains the full frozen mission set and candidate banks. `manifest_integrity.json` authenticates the uncompressed and compressed bytes by SHA256. Scripts automatically restore `manifest.json` if absent and verify its checksum on restoration.

Commands from repository root (run one phase at a time):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_agv_timed_main --phase calibrate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_agv_timed_main --phase search
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_agv_timed_main --phase development
```

Calibration evaluates 24 shared candidates per layout on training cases, plus strict-FCFS and pure-waiting equivalence checks. Initial weights comprise FCFS, the fixed four-feature reference, and four training-ranked diverse candidates. No development result is used to select these weights. The bank includes every method/seed's pilot training-best fit, not a selection based on pilot development outcomes.

Search means 45 jobs: three layouts × three methods × five optimizer seeds, 72 charged calls per job. For total fitting cost, also charge the shared 24-call calibration to every method/seed (96 calls in all). The six initial search observations count inside 72 and reuse already computed calibration records. Distinguish physical simulation executions from charged observations. `basic` means random search.

Development evaluation is blocked until all 45 searches finish. It includes FCFS, fixed4 reference, strict FCFS and the training-best warmstart baseline, plus fifteen fitted policies per layout. Warmstart-best is descriptive, separating shared initialization gains from search gains; the primary comparison family remains the predeclared fifteen comparisons across layouts. Development is not final confirmation.

Calibration raw records are saved as `calibration_records.json.gz` when this preparation stage is archived. To restore:

```sh
python - <<'PY'
import gzip,json
from pathlib import Path
root=Path('outputs/agv_timed_main_v1')
for name,content in json.loads(gzip.decompress((root/'calibration_records.json.gz').read_bytes())).items():
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid archive path')
    path=root/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=content:
        raise ValueError(f'Refusing to overwrite different record: {path}')
    path.write_text(content)
PY
```

Verify archive SHA256 in `calibration_archive_integrity.json`. Every row is matched to scenario ID and mission-content hash before use. Jobs retain incomplete runs, checkpoint each episode, and reject incompatible source/config/checkpoint inputs. Do not run multiple writers for the same phase/output directory.

Read `stage_status.json` for what has actually completed. The existence of scripts or initial weights does not mean the 72-call search or development evaluation has finished.

Cost scope: 96 is incremental fitting budget. Historical pilot training that produced source vectors cost 486 charged objectives / 5832 charged episodes across all layouts. Report that common pretraining cost separately; do not claim end-to-end training cost is only 96 calls.

The sum 24+72=96 is a phase-call ledger including the six initial cache hits twice (once in calibration and once in search). A fit uses at most 24+66=90 unique candidate observations before other duplicate hits. Neither 96 nor repeated optimizer fits represent independent traffic samples.
