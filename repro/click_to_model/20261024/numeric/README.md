# Numerical experiment increment, 2026-10-24

This package contains reviewable numerical CSV exports of 6 development stress runs, 4 complete LMO file-order runs, and one 60-input generated-mesh motion diagnostic. It includes failures. It does not establish a successful robust Click-to-Model system or publication-level performance.

`summary.csv` reports 19 metric rows. `inventory.csv` lists all 41 numerical tables and their verified export counts. CSV fragments repeat their header; concatenate data rows in part order. Adjacent schema files specify each original dtype, shape, and layout: `matrix_rows` stores 16 matrix entries per row, `aligned_vectors` stores one aligned sample per row, and `flat_values` stores `array_key,flat_index,value`. All retained numerical values were exported with floating-point round-trip verification. String arrays are omitted; numerical source frame IDs remain available. The package contains no source code, logs, local absolute paths, provenance digests, model weights, input images, meshes, or binary archives.

## Development stress: recovery remains unsolved

All six runs reuse the same mustard development trajectory and initial CAD/mask. The three fixed conditions are unchanged input, every fourth original input, and synthetic foreground covering selected inputs `[100,130)` with RGB=32 and depth=0.25 metres. These synthetic inputs are not physical occlusion annotations. Stride4 enlarges between-observation displacement; it does not simulate exposure blur or establish sensor frame rate.

| Condition | Baseline ADD mean mm | Adaptive ADD mean mm | Baseline failures | Adaptive failures |
| --- | ---: | ---: | ---: | ---: |
| none | 3.674 | 3.635 | 0/736 | 0/736 |
| stride4 | 366.336 | 355.593 | 166/184 | 165/184 |
| synthetic occlusion | 354.496 | 300.038 | 636/736 | 636/736 |

Recovery requires three consecutive GT-defined inputs after input 130 with ADD at most 0.1 original CAD diameter. Neither occlusion run recovered during the remaining 607 inputs; `recovery.csv` preserves these right-censored outcomes. All selected GT-defined tracking inputs, including corrupted and rejected predictions, remain in the denominator. Registration is separate and excluded from tracking metrics.

## Real LMO objects: original file order with large viewpoint changes

Each object/method run inferred all 1214 RGB-D inputs, including inputs lacking GT. A first visible mask was used only for initialization; subsequent inference did not receive GT poses or future masks. Objects 10 and 11 belong to the same scene, so they are not independent capture sequences. The original file order is preserved. Temporal continuity and uniform timestamp spacing have not been established; these results are large-viewpoint-change file-order stress evidence, not ordinary contiguous-video results.

| Object / method | ADD mean mm | ADD failures | Discrete MSSD mean mm | Discrete MSSD failures |
| --- | ---: | ---: | ---: | ---: |
| 10 baseline | 648.281 | 1174/1174 | 703.527 | 1174/1174 |
| 10 adaptive | 303.071 | 1166/1174 | 349.077 | 1171/1174 |
| 11 baseline | 377.594 | 902/902 | 453.971 | 902/902 |
| 11 adaptive | 402.377 | 902/902 | 479.408 | 902/902 |

All metrics use every original model vertex, unchanged raw GT rotations, and all GT-defined tracking inputs. Missing or invalid predictions count as failures. Objects 10/11 have 1174/902 GT-defined tracking inputs respectively. Geometry support, temporal admission, and memory commit flags do not filter accuracy and are not independent identity truth.

ADD uses corresponding vertices without symmetry equivalence. ADD-S uses nearest predicted vertices for each GT vertex, following the BOP ADI direction; it is not an exact declared-symmetry metric. MSSD minimizes maximum corresponding-vertex error over identity plus the published discrete original-CAD symmetry transforms, including their translations. All reported failure rules use error greater than 0.1 original BOP diameter. This diagnostic MSSD threshold is not official multi-threshold BOP AR, and first-mask sequential tracking is not the official BOP challenge protocol. Continuous symmetry and independent texture strata are not demonstrated.

## Generated mesh: relative motion only

The generated-mesh development prefix has 60 inputs and 59 evaluated tracking inputs: mean first-frame-gauge-aligned motion ADD is 5.498 mm, with 0/59 failures. It omits the other 315 inputs of that source sequence and is not a full-sequence result.

Generated and original CAD models have different coordinate gauges. This diagnostic uses exactly one post-hoc gauge, `C = inverse(P_generated_0) * GT_original_0`, followed by `P_aligned_t = P_generated_t * C`. Only the first GT pose determines this fixed gauge. No per-frame/trajectory alignment or scale fitting is performed. Initial pose error is cancelled by construction and is not accuracy evidence. The metric measures relative motion on original CAD vertices; it does not measure absolute generated-mesh pose, initialization accuracy, mesh shape/hidden-surface accuracy, or physical-instance identity. The run's click provenance remains a development limitation; no general click-selection success rate is established.

Single runs are retained without best-run selection. All ten complete stress/LMO runs and the meshless diagnostic passed their recorded source/inference evidence checks; this says the evidence contract is valid, not that the poses are correct. Failures above remain unresolved.
