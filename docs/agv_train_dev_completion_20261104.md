# AGV physical-time training and development completion — 2026-11-04

Repository: dongkun1233/robotic-deadlock
Branch: research/physical-time-v1-20261102

## Verified completion
- 45 search jobs, 72 evaluations each: 3,240 recorded objective evaluations.
- All search trajectories passed audited portable replay; no missing raw evidence in the supplied completed backup.
- Development: 57 completed policy/layout records, 2,736 episode rows, 144 unique scenarios.
- No duplicate/missing development records, incomplete episodes, or failed safety flags were found in the supplied backup.
- Final independent test has NOT been executed.
- Counts and completion do not establish superiority or statistical significance.

## Published files
- agv_portable_resume.py: additive CPU-portable runner; frozen research source files are unchanged.
- outputs/agv_timed_main_v1/search/<job>/{checkpoint,best,complete}.json: all 45 final jobs.
- outputs/agv_timed_main_v1/training_summary.csv
- outputs/agv_timed_main_v1/development_summary.csv: 57 policy/layout means.
- outputs/agv_timed_main_v1/completed_validation_summary.json
- outputs/agv_timed_main_v1/portable_pipeline_status.json
- outputs/agv_timed_main_v1/development_complete.json

The completed_validation_summary.json and portable_pipeline_status.json describe the latest completion. Older stage/recovery status artifacts describe earlier checkpoints and must not be interpreted as current execution status.

## CPU portability and provenance
Original OpenBLAS CPU dispatch was SkylakeX; the first Codespaces dispatch was Haswell.
A 2,805-evaluation cross-dispatch audit observed a maximum vector difference of 4.440892098500626e-16.
The portable runner allows only absolute vector error <= 8 * float64 epsilon (1.7763568394002505e-15), with exact evaluation index and source checks.
Historical vectors and objectives are retained; historical evaluator inputs are canonicalized to saved vectors.
Native optimizer arithmetic is not bitwise identical across CPUs. Differences and checkpoint-prefix integrity are audited.
This is tolerance-validated continuation, not a claim of bitwise reproducibility.
The frozen manifest/source hashes remain checked.

## Baselines and interpretation
fcfs is feasible-first FCFS; strict_fcfs retains head-of-line blocking.
Both were included in development evaluation alongside fixed4_reference, warmstart_best, and 15 trained policies.
Equivalence of current strict_fcfs with the original capstone implementation still requires a separate code audit.
Summary values are simulation seconds, not optimizer runtime or step counts.
Development data are not the final independent test.

## Deliberately excluded from this publication
No new per-scenario training/development raw records, cache files, trajectories, binary files or compressed raw archives were transmitted.
The complete user-supplied backup agv_completed_train_dev.tar.gz preserves these materials separately.
Historical raw archives already published on this branch were left unchanged.
The previously blocked file was research_archives/recovery_raw_20261104/single_101_pso.tar.gz. Its transfer was not retried.
The earlier automatic-review reason concerned external transfer of compressed raw data; no claim is made that every raw file is individually prohibited.
This published checkpoint set is not a complete raw-data backup: do not delete the full user backup.
The portable runner's raw-evidence gate requires the full backup (or independently verified reconstruction) to run against these completed checkpoints.

## Next steps
Paired scenario comparisons, optimizer-seed variability analysis, legacy Strict-FCFS audit, and a frozen final-test protocol precede independent testing.
