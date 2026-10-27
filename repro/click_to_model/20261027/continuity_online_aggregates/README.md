# Actual online continuity integration: 180 synthetic RGB-D frames

This was one fresh GPU run of the new production entry with `--output-policy
continuity`, SAM-score initialization, FoundationPose and TAPNext. The policy
consumed current typed events during tracking. It was not merely an offline
replay. All 180 inputs, 179 tracking frames and 756 registration candidates were
retained. There were no denied reads or unknown output slots.

162 inputs were published and 18 explicitly abstained; tracking-only coverage
was 161/179 (89.944%). The initialization was published. Frames 37–54 abstained
and publication resumed at 55. All seven trust flags stayed false. A published
working pose is not a certification of unique physical instance identity.

The raw audit checked every publication event against JSON, NPZ and the actual
candidate using dtype, shape and exact bytes. A separately invoked frozen v4
gate on this run's current events matched all 180 online decisions. The actual
initialization's 29 arrays and all 180 candidate poses also matched the original
baseline byte for byte. Therefore the existing audited numerical errors can be
reused; no new GT file or geometry calculation was needed. Candidate accuracy
did not improve. Existing sparse-normal replay has its own source binding; it
is not silently treated as the original baseline or this new online run.

Tracking-only results (all 179 frames remain in the denominator):

| Fixed metric | Raw errors | Published errors / published known | Correct rejected | Wrong rejected |
|---|---:|---:|---:|---:|
| ADD <= 0.1 diameter | 2 | 0 / 161 | 16 | 2 |
| MSSD < 0.1 diameter | 7 | 0 / 161 | 11 | 7 |
| MSPD < 5 normalized pixels | 11 | 0 / 161 | 7 | 11 |

Zero observed published risk trades away correct outputs and is not a calibrated
risk guarantee. These are correlated development frames, not independent trials.
`threshold_counts.csv` has 126 rows: 21 fixed thresholds, all-input/tracking/init
scopes and raw-candidate/actual-publication variants. Undefined risks remain
null (empty CSV cell), not zero. `per_frame.csv` retains the complete 180 clocks,
availability, byte-identity qualification and reused numeric errors, with no
pose matrices. ADD is inclusive; MSSD and MSPD thresholds are strict. Diameter
is 0.1649631473996541 m; MSPD is normalized to 640-pixel width.

Actual parent wall time was 40.962301 s and child wall time 38.224619 s.
The reported 11.444965 tracking FPS includes current input, session evidence,
FP and online policy/evidence work but excludes final output serialization.
180/parent wall is about 4.394284 inputs/s; neither figure is deployment FPS.
`timing.json` preserves exact timings and nested initialization boundaries.

The final full CPU regression passed 1658 tests. The earlier CPU run's missing
ignored fixtures/resource aliases and strict parser incompatibility are retained
in source-hash history. The first launcher draft never ran a GPU attempt: two
CPU counterexamples exposed candidate-byte coverage accepting float64/signed
zero differences. V2 fixes this and tests the real lazy-import call order; the
origin implementation itself needed no change. No failed attempt was erased.

Only this continuous-visible synthetic case was executed online here. This does
not newly validate complete-occlusion identity recovery, real-world multi-instance
identity, point-free textureless objects, or neural weight adaptation. The package
contains only numerical results, source hashes and documentation; no raw poses,
GT matrices, images, masks, model/weight files, logs, code or local absolute paths.
