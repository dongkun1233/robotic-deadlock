# V2 resumable training runner

Training commands use only the training split. All candidates see the same
scenario order within a layout. Smoke and main have separate output roots and
cannot reuse one another's checkpoints/caches.

Smoke: 12 scenarios/layout, one representative per count/type cell, seed 101,
8 observations including the six common initial vectors. This verifies execution
and estimates cost; it cannot establish policy ranking or convergence. Local
2-scenario smoke mode is for engineering checks only and uses another directory.

Main: 72 observations, five seeds, three methods, three layouts =45 fits.
Six initial vectors are shared across fits, so absent other duplicate vectors the
unique vector count/layout is 6+66*15=996. With 408 total training scenarios the
planned maximum is 406,368 scenario executions, before additional duplicates.
Do not interpret a roughly 18-minute 984-case FCFS audit as full training time.

Each vector cache is compressed and updated atomically after every scenario;
each completed observation is checkpointed. Cached episodes carry scenario ID,
mission hash, per-robot time evidence, admission log and wall time. Runtime source,
manifest, initialization and Python/NumPy versions are checked on resume.
Optimizer randomness is reconstructed and saved observations replayed exactly.
A process lock prevents concurrent runners writing to the same mode directory.
An interruption may rerun the one episode not yet saved, not completed episodes.

Main fitting requires the complete, matching FCFS audit and no incomplete audit
cases. The source is immutable during a run; stop safely before upgrading runtime.
The default tmux launcher starts SMOKE only. Main requires explicit --mode main.
No development evaluation, Fair evaluation, ablation or final test is automatic.

Cache sharing affects physical compute but every observation is charged inside
each optimizer's identical budget. Six anchor observations are not new samples
every time another fit reads them. Report realized cost as well as nominal cost.
Mean capped makespan plus 1201 per failed scenario preserves failure-first order
because the bounded mean is in [0,1200]. Stalls/timeouts are retained.

Use report_agv_capstone_v2_smoke to estimate serial compute/storage on Codespaces.
No automatic push is configured: Codespaces artifacts are not on GitHub until a
separate verified commit/push or artifact transfer succeeds. Do not use git add .
to upload recovery archives or unrelated files.
