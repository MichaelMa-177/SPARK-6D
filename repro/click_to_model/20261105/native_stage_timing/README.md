# Native fixed30 full-state hash timing comparison

Two serial instrumented runs, each with 30 inputs, 29 tracking frames and all 756 initialization candidates. The second run batches same-device state bytes for transfer while retaining every full state hash and guard. The algorithm, thresholds and update policy are held fixed by the experiment launchers.

| Observed measure | Original hash + timing probe | Batched hash + same probe |
|---|---:|---:|
| Tracking total, 29 frames | 21.263443 s | 19.394499 s |
| Tracking p50 | 0.780366 s | 0.609424 s |
| Tracking p95 | 1.038568 s | 0.974269 s |
| Parent elapsed | 59.563989 s | 44.590903 s |
| Tracking full-model hash, 220 calls | 9.425237 s | 7.397789 s |
| Peak CUDA allocated | 2749754368 bytes | 2749754368 bytes |
| Peak CUDA reserved | 5041553408 bytes | 5041553408 bytes |
| Whole-child peak RSS | 5745616 KiB | 5747032 KiB |
| Recorded timing spans | 42024 | 12410 |

Each arm passed 211 exact array checks against the original fixed30 run: 29 initialization arrays, 90 RGB/depth/K arrays, 30 saved poses, and 62 publication arrays. All 11 full structure comparisons passed, including 29 prepared updates, 232 before/next private-head digest signatures, every proposal scalar, callback qualification, and publication decisions. No difference was ignored or given a tolerance. Epoch path identity fields are the only normalization. Both finish with holder version 29 and 45 callbacks.

This observed tracking difference is -1.868944 s (-8.79%). It is a single comparison, not a general speed result. Batching removes calls to the per-tensor identity wrapper, reducing recorded spans from 42,024 to 12,410. Reduced observer work contributes to timing differences; the whole change cannot be attributed to transfer batching. The 221 outer whole-model hash calls remain present in both runs. Their inclusive wall times are the comparable measured boundary; their exclusive times have different child-span coverage.

`stage_timing.csv` contains all, tracking-frame and outside-tracking scopes. Inclusive stages overlap: surface includes 90 evidence-sink spans per arm, and sink includes compression. Do not add overlapping rows. Exclusive rows partition only the recorded nested work. CUDA values are current-stream elapsed intervals, not kernel sums. Stage totals were independently reconstructed from all span rows and parent/child relations.

Tracking totals use all 29 saved latencies. Parent elapsed also includes startup, initialization and inherited source/asset checks. Peak memory covers the entire child and can be set by initialization. This prefix contains no partial occlusion. No GT, accuracy evaluation, real-video robustness evidence or deployment FPS claim is added by this timing experiment.

The preparation reader initially assumed packet descriptors used raw-byte SHA only; that CPU self-pair failed and was retained locally. It was corrected to the unchanged producer's dtype + shape + bytes digest. The corrected self-pair and four mutation/accounting contracts passed before these terminal analyses. This is separate from the repaired profiler v1 exception handling, preserved in v2 with 4 normal and 9 failure CPU contracts. No failed GPU attempt was discarded in this two-arm experiment.

The package includes only aggregate numbers, stage counts/timings and artifact hashes. No images, pose matrices, GT arrays, model, weights, logs or source code are included. The two arm parity receipts and the frozen plan / timing / execution receipts are identified in `sources.json`.
