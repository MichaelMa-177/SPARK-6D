# Three actual normal-checker arms: same poses, corrected rank, recovered cost

Three arms ran the same two synthetic streams: original CAD-normal baseline, full-image measured-normal v2, and sampled-stencil measured-normal v3. Each arm completed **360 inputs / 358 tracking frames**. Direct checks confirm **all 360 actual pose triplets are byte-identical**, and all 358 per-frame neural iteration counts match. All 378 paired target-metric tables are diagonal: **no pose or accuracy gain**.

Measured-normal diagnostics change local full-rank counts from **155 to 98 /179** on the continuous stream and **147 to 131 /179** on the ambiguous stream. Sparse v3 preserves the actual v2 final and per-iteration evidence exactly, including unknown ranks. This is a conditional geometric diagnostic change, not physical-instance verification.

Actual instrumented timings, in baseline / full-map-v2 / sparse-v3 order:

| Stream | Initialization seconds | Tracking seconds | Measured tracking FPS | Parent wall seconds |
|---|---:|---:|---:|---:|
| Continuous | 16.102 / 36.778 / 14.446 | 15.504 / 18.167 / 15.300 | 11.546 / 9.853 / 11.699 | 40.650 / 63.775 / 38.430 |
| Ambiguous | 12.786 / 35.216 / 13.088 | 13.573 / 15.682 / 13.625 | 13.188 / 11.414 / 13.138 | 36.872 / 59.547 / 35.357 |

Sparse v3 removes most of the extra full-map cost in these runs. Differences versus baseline are small and mixed across scopes. These arms ran sequentially once, so cache/warmup/order/system-load variation is not controlled; this is not a statistically established general speed improvement. No micro-kernel speed multiplier is treated as FPS.

Timing scopes overlap. Registration and admission are contained in initialization; parent and child include broader work. Do not add scopes or call instrumented tracking FPS deployment performance.

The ambiguous stay/exchange worlds explain the same observed video and use the same saved inference per arm. They are not additional independent videos or samples. Under exchange, all 70 post-opening frames still follow the distractor. The adjacent-switch counter can be zero because neither-associated gaps break the chain; zero does not mean preserved identity. All seven trust flags remain false, explicit abstention is absent, and empty verified risk remains null.

Files retain 66 five-state association migration tables, 378 fixed metric paired tables, 42 actual-rank phase rows, six actual timing rows, and source hashes. Initialization, all inputs, all tracking and all fixed occlusion phases are explicit. The earlier full-map-v2 package and failed posthoc prefix remain unchanged. Its narrow timing-path repair did not change numerical errors. A separate prior aggregate-review elapsed-time typo was documented by erratum and is not consumed as experiment timing here.

This bounded check recomputes aggregates from saved errors and actual diagnostic/pose bytes, without rereading GT or recomputing geometry. It is author-side independent-formula checking, not external experimental replication. No raw poses, GT matrices, point trajectories, images, or machine-local paths are published.
