# Idle-skip execution optimization

The reference v2 engine and its frozen manifests remain unchanged. The fast engine
uses the same integer tick resolution, action durations, candidate features,
eligibility, tie order, reservations, and protection thresholds.

After each processed tick all currently eligible admissions and possible moves
have been attempted. Until the next trip/rotation boundary or planned release,
occupancy/reservations and candidate eligibility cannot change. Blocked candidates
cannot enter just because their scores or protection ages change. Skip only this
idle interval and accumulate stationary wait as skipped_ticks*dt. Rotation events,
spawn timing tolerance and timeout tick semantics remain intact. When complete
per-tick decision traces are requested, execute the reference engine instead.

Engineering comparison covers every development layout/count/type cell (72), one
scenario each, six configurations: FCFS, strict FCFS, random signed heuristic,
and the same heuristic with thresholds 0/20/40 seconds. Total 432 paired runs.
Compare the entire result recursively except the executed-loop counter. Floating
fields allow absolute error <=1e-7; integer choices, event order and admissions
must match. Exercise non-grid planned releases, non-grid timeout cutoffs and
.005/.01 clocks separately. This is finite evidence, not a proof for all inputs.

Fast fitting requires a matching manifest/source-hashed 432-pair mixed report.
Its own runner/source/version identity and caches are isolated from reference
smoke/main outputs. It retains 72 observations, five seeds, the same initial bank
and train scenarios. The launcher runs equivalence followed by eight-observation
smoke ONLY; it does not launch main fitting or final tests.

Reported runtime speedup is hardware/case dependent. Re-estimate total budget from
Codespaces fast smoke, including compression, optimizer and checkpoint costs;
do not promise the full remaining research fits a quota from simulator-only times.

## Local engineering evidence

On the balanced preparation manifest, 432/432 paired runs matched. There were
810 protection overrides across these cases, so protection was actually exercised.
Measured total reference/fast simulator wall-time ratio: 30.08x (single 38.39x,
double 31.40x, multi 26.11x). These are local measurements, not Codespaces estimates.
The two-scenario engineering smoke completed all nine layout/method jobs at eight
observations; every candidate vector, objective and failure count exactly matched
the reference trainer's corresponding trajectory. Non-grid releases/cutoffs,
two clock resolutions and same-inlet blocking matched separately. Full per-tick
decision tracing fallback was also checked. Codespaces repeats the 432 paired
checks before its isolated 12-scenario fast smoke; main remains an explicit step.
