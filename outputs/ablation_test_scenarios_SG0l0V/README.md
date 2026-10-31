# Transfer Generalization Test Scenarios

Train layout: single-cross only, using the existing 400 training scenarios.
Test layouts: single-cross and double-cross.

Scenario cases:
- 5: Scenario 5: 2+3+4 arrival burst + direction skew + exit concentration
- 6: Scenario 6: 2+3 arrival burst + direction skew
- 7: Scenario 7: 2+4 arrival burst + exit concentration
- 8: Scenario 8: 3+4 direction skew + exit concentration

Scenarios per case: 5
Single-cross AGV counts: 20
Double-cross AGV counts: 24

Single rows: 20
Double rows: 20

Burst scenarios set half of the AGVs to early departures sampled from 0-2 steps; the rest are sampled from 3-8 steps.
Direction-skew and exit-concentration scenarios sample the dominant direction count and randomly distribute the remainder.
Every non-dominant point receives at least one AGV, and no non-dominant point exceeds half of the non-dominant remainder.
