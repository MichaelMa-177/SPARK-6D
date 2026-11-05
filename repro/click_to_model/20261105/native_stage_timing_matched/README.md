# Native fixed30 hash timing with a common outer observer

This is a new pair of serial instrumented runs, each with 30 inputs, 29 tracking frames and all 756 initialization candidates. Both arms use the same observer that omits only the full-model hash's per-tensor child timing spans. Every original full-model hash and guard still executes. The second run batches same-device state bytes for transfer while retaining every full state hash and guard. The algorithm, thresholds and update policy are held fixed by the experiment launchers.

| Observed measure | Original hash + common observer | Batched hash + common observer |
|---|---:|---:|
| Tracking total, 29 frames | 20.235211 s | 18.620153 s |
| Tracking p50 | 0.580633 s | 0.522045 s |
| Tracking p95 | 1.009937 s | 0.950760 s |
| Parent elapsed | 45.010658 s | 43.509481 s |
| Tracking full-model hash, 220 calls | 9.016768 s | 7.304999 s |
| Peak CUDA allocated | 2749754368 bytes | 2749754368 bytes |
| Peak CUDA reserved | 5041553408 bytes | 5041553408 bytes |
| Whole-child peak RSS | 5743024 KiB | 5740960 KiB |
| Recorded timing spans | 12410 | 12410 |

Each arm passed 211 exact array checks against the original fixed30 run: 29 initialization arrays, 90 RGB/depth/K arrays, 30 saved poses, and 62 publication arrays. All 11 full structure comparisons passed, including 29 prepared updates, 232 before/next private-head digest signatures, every proposal scalar, callback qualification, and publication decisions. No difference was ignored or given a tolerance. Epoch path identity fields are the only normalization. Both finish with holder version 29 and 45 callbacks.

This observed tracking difference is -1.615057 s (-7.98%). It is a single comparison, not a general speed result. Both arms record the same number of spans, every stage/scope call count matches, and all span IDs, parents, labels, frame IDs and callback ordinals match. `observer_counts.json` retains the complete call-count comparison. The per-tensor full-state timing label is absent in both arms; the 221 outer whole-model hash calls remain present. Thus this comparison addresses the unequal observer span coverage of the earlier pair. One run per arm still leaves run order, warm state and runtime variation uncontrolled.

`stage_timing.csv` contains all, tracking-frame and outside-tracking scopes. Inclusive stages overlap: surface includes 90 evidence-sink spans per arm, and sink includes compression. Do not add overlapping rows. Exclusive rows partition only the recorded nested work. CUDA values are current-stream elapsed intervals, not kernel sums. Stage totals were independently reconstructed from all span rows and parent/child relations.

Tracking totals use all 29 saved latencies. Parent elapsed also includes startup, initialization and inherited source/asset checks. Peak memory covers the entire child and can be set by initialization. This prefix contains no partial occlusion. No GT, accuracy evaluation, real-video robustness evidence or deployment FPS claim is added by this timing experiment.

The unchanged parity analyzer has CPU contracts for signed-zero byte differences, private-head/proposal changes, epoch normalization and corrupted nested timing. The added observer comparator rejects changed label counts or changed parent topology even when totals match. Preparation failures from the earlier unmatched observer analysis remain in its original record. This package is a separate common-observer experiment and does not overwrite the earlier pair.

The package includes only aggregate numbers, stage counts/timings and artifact hashes. No images, pose matrices, GT arrays, model, weights, logs or source code are included. The two arm parity receipts and the frozen plan / timing / execution receipts are identified in `sources.json`.
