# Pre-loss measured anchors: development results

**Reused development data. Every state remains tentative; object identity is not certified.**

Verification status: `verified_against_pinned_independent_reviews`. A successful arithmetic audit does not establish identity or generalization.

Six runs compare original TAPNext + adaptive FoundationPose against the same pipeline with measured pre-loss anchors. Both arms use the same known CAD mesh, exact initial input/pose arrays, and exact original point stream in each case. The two mustard cases share one original sequence; these are not three independent captures. No held-out test or repeated-trial confidence interval is claimed.

| Case | Tracking frames / arm | Failures, original → pre-loss | Mean ADD (mm), original → pre-loss | Tracking inputs/s, original → pre-loss |
|---|---:|---:|---:|---:|
| Cracker / blank RGB-D | 374 | 18 → 18 | 9.6135 → 9.6148 | 14.41 → 10.74 |
| Mustard / foreground occlusion | 736 | 237 → 29 | 121.9083 → 9.7670 | 15.78 → 9.42 |
| Mustard / source stride 4 | 184 | 0 → 0 | 3.6927 → 3.7155 | 15.42 → 8.99 |

**Full denominator:** 255/1,294 failures (19.71%) → 47/1,294 (3.63%). All 208 fewer failures occur in the mustard occlusion case. Cracker remains at 18 failures, mustard remains at 29, and stride-4 remains at zero. The other two cases do not improve failure counts and their mean ADD slightly increases. Pre-loss anchors reduce measured tracking throughput in all three cases.

The mustard five-consecutive-input recovery delay decreases from 208 inputs to zero. The first correct five-input run starts at selected input 338 and is confirmed at 342 for the original arm; it starts at 130 and is confirmed at 134 for pre-loss anchors. Reappearance is input 130. This is retrospective GT-based evaluation, not an online correctness decision. The 29 remaining mustard errors occur at inputs 101–129 and remain in the full 736-frame denominator.

Primary metric: ordinary absolute ADD over the frozen `trimesh.load(process=False)` CAD vertex samples (8,291 cracker; 10,983 mustard), preserving coordinate multiplicities including UV seam splitting. The original cracker OBJ has 8,291 vertex rows but 7,866 unique coordinates; its loader rows have the same order and weighting. The mustard OBJ has 7,866 rows and 7,866 unique coordinates, expanded to 10,983 samples by the loader. Thus neither metric should be described as averaging unique coordinates. The source field `all_original_CAD_vertices` is a legacy name for loader sample counts. The separate raw-OBJ sensitivity diagnostic averages original OBJ vertex rows (including cracker row duplicates) and does not replace the frozen primary metric. A frame fails when ADD is strictly greater than 0.1 times the original model diameter: 26.9573 mm cracker, 19.6463258 mm mustard. There is no alignment, scale fit, GT repair, birth realignment or symmetry minimization. Missing/invalid estimates count as failures; all six actual runs have none. Frame zero is evaluated separately and excluded from the tracking denominator. All six initializations pass. Secondary unhidden/post-intervention metrics are labelled explicitly in the JSON and do not replace the primary denominator.

Injected intervals use selected input indices and half-open ranges: cracker [20,40) zeros the entire RGB-D input; mustard [100,130) uses the existing foreground-occlusion bytes (RGB=32, depth=0.25 m). Stride-4 uses original source indices 0,4,…,736 with no added occlusion. Figure trajectories contain every tracking input, use a common logarithmic millimetre axis, and show the failure thresholds and occlusion intervals.

Seed-source counts are policy selection outcomes, not identity admission or proof that neural refinement accepted a seed. The JSON and CSV retain original, secondary, fallback, conflict, missing and initialization counts, plus manager states, secondary calls, births/readiness, availability and exact-pairing results. Availability does not establish accuracy. All three initializations are paired; all 375/737/185 original input-stream comparisons are exact, with zero mismatched or unknown inputs.

Timing boundaries: reported tracking inputs/s equals tracking inputs divided by summed measured frame wall times. It includes decoding, synchronized original and actual secondary point calls, measured-depth fitting, CAD support proposals, pre-loss lifecycle, adaptive FoundationPose and per-frame evidence writes. **It excludes the `frames.json` journal rewrite after EVERY frame**, as well as load, registration, first input, final journal rewrite and finalization. The source summary boundary wording is retained separately for provenance; this statement specifies the actual code boundary. This measured frame rate is not end-to-end throughput. The CSV/JSON separately retain child and parent all-input throughput (denominators 375/737/185 including frame zero), frame median/p95, parent launch-to-child-completion wall, child wall, FP model-setup load, registration, first input frontend call, first input frame wall and CUDA peaks. Child wall includes validation/import/load/registration/all inputs/all per-frame journals/audit/artifact hashing and excludes the final report write. Parent wall includes child processing and subprocess overhead; it is per run, not total experiment duration. The first point frontend call includes lazy startup plus inference; point-backend object construction is not complete checkpoint loading. Fresh child processes do not remove filesystem/JIT/cache effects: the first FP setup takes 20.45 s while later ones take about 6.47–7.00 s. Cracker parent-observed throughput slightly increases (4.66 to 4.74 inputs/s) despite slower measured tracking frames; this cache/order-confounded observation is not a speedup claim. There are no isolated cold-start or production FPS claims.

Files: `summary.json` preserves exact scalar summaries, methodology, all pairing states and source hashes; `per_case.csv` has one row per run; `raw_OBJ_sensitivity_metrics.csv` contains 30 aggregate rows (five metric modes for six runs); `figures/` contains two scientific figures in PNG, SVG and PDF. `MANIFEST.json` hashes every payload file except itself. The figure curves contain GT-derived ADD errors, but the package contains no raw poses, GT arrays, input images, meshes, weights, source code or absolute local paths. The local builder is deliberately outside this payload.

The package builder only reaggregates existing posthoc error arrays and report timings; it never reads GT files or reruns inference. Independent GT recomputation and event/timing audits are separate prerequisites for verified status.

Comparison source SHA-256: `af5bf1ae591ba8629540d02b505063fad9c2fefa1c4059a19719f650e1036c26`.
Independent raw audit receipt SHA-256: `28a3cae58f649388f004a6b24077d55f1b7f82c44aae6a1cb00afc9d82ecd3c5`.
Independent actual metric/event/timing review receipt SHA-256: `ec415e80c4644a71593b9a12ac50659a77b12f59b3efe76f7f6ddf020e8eb9ad`.
All remaining provenance pins appear in `summary.json`.

Raw OBJ vertex-row sensitivity (subsequent diagnostic; primary metric unchanged):

| Case | Failures, original → pre-loss | Raw-row mean ADD (mm), original → pre-loss | Classification changes vs frozen primary |
|---|---:|---:|---:|
| Cracker / blank RGB-D | 18 → 18 | 9.613517 → 9.614840 | 0 / 0 |
| Mustard / foreground occlusion | 237 → 29 | 121.917731 → 9.850210 | 0 / 0 |
| Mustard / source stride 4 | 0 → 0 | 3.768439 → 3.794591 | 0 / 0 |

All six absolute failure classifications and recovery starts/confirmation frames are unchanged under raw OBJ vertex-row weighting. The main trajectories and primary table continue to use the frozen loader weighting. The sensitivity CSV also reports secondary relative-pose and point-only diagnostics; those are not the primary absolute neural-pose claim.
