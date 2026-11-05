# Current Cutie / TAPNext++ frontend timing

On the same 101-frame normal mustard development prefix, three fresh processes per
model completed all 606 planned inputs. Each process reports frame 0 initialization,
frames 1–10 warmup, and frames 11–100 primary timings separately. There are 270 primary
observations per model. Model order was Cutie/TAPNext++, TAPNext++/Cutie, Cutie/TAPNext++.

| Current frontend | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Cutie | 12.2552 | 11.6571 | 15.2891 | 17.2934 |
| TAPNext++ | 41.0387 | 41.8788 | 46.1483 | 47.4491 |

The common outer timer brackets the actual public frontend call with CUDA synchronization.
It includes preprocessing, inference, CPU result transfer and adapter validation/copies.
Decode, model loading, file output, depth geometry and 6D refinement are excluded.
Initialization, warmup, model loading, CUDA startup and peak memory are retained in
`summary.json`; per-repetition distributions are also in `latency_by_repetition.csv`.
The GPU/software identity is recorded per successful child in `summary.json`.

Cutie produces a full-resolution 640×480 single-object mask and probabilities.
TAPNext++ resizes internally to 256×256 and tracks 64 fixed first-mask points,
returning coordinates and logits. Both use FP32 without compilation or AMP.
These are the current application configurations, with different output tasks and
internal resolutions. Results do not establish a universal model-speed ranking,
equal tracking accuracy, unique instance identity, or end-to-end SPARK-6D FPS.
Three repetitions and correlated video frames support descriptive reporting only.

The previous attempt failed CUDA initialization in every child: 606 planned inputs,
zero processed inputs and zero performance observations. Its errors and source hashes
are preserved separately; they are not averaged into the successful latency results.
The complete attempt uses the same frozen plan. No hidden retry or discarded slow frame
occurs within either recorded attempt.

This package contains numeric aggregates, explanatory text and source hashes only.
It includes no RGB, masks, raw point trajectories, model weights, source code or logs.
The exporter independently recounted every reported mean and linear-interpolated
percentile with Python sorting/math, without importing the producing benchmark or NumPy.
This checks numeric integrity, not hardware timer correctness or neural output accuracy.
