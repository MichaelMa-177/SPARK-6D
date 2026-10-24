# Causal BootsTAPIR RGB-D candidate probes

This independent numerical increment contains four frozen causal point-tracker probes and their post-hoc evaluations. It is separate from the SPARK-6D end-to-end tracking runs. All 3D seed proposals were deliberately rejected to check that rejecting pose memory does not rewind learned 2D query state. No experiment here proves end-to-end pose recovery or physical-instance identity.

The existing causal BootsTAPIR model consumed one RGB frame at a time. Initial queries and query features remained fixed; no later-frame query replacement or GT-selected refresh was used. Current depth lifts visible queries to 3D, and rigid fitting proposes full SE(3) candidates. Learned visibility probabilities are not calibrated identity confidence. This is an evaluation of an existing learned backend, not a claimed new tracking algorithm.

`inventory.csv` lists 12 tables: each probe's initial queries, raw observations, and per-frame post-hoc metrics. The exports retain 266,038 numerical values. Adjacent schema files describe dtype, shape and `matrix_rows`, `aligned_vectors`, or `flat_values` layout. CSV fragments repeat headers and preserve all data rows in part order. All numbers are checked by round-trip reconstruction; string arrays are excluded. There are no source files, logs, local absolute paths, provenance digests, weights, images, meshes, or binary archives.

## Results and exact scope

| Probe | Inputs / tracking denominator | Primary candidate metric | Failed candidates | Scope |
| --- | ---: | --- | ---: | --- |
| cracker clean | 80 / 79 | first-frame-gauge-aligned motion ADD | 0/79 | Prefix of a 375-input source |
| cracker synthetic blank | 80 / 79 | first-frame-gauge-aligned motion ADD | 20/79 | Same prefix; RGB and depth zero on inputs `[20,40)` |
| mustard synthetic occlusion | 737 / 736 | raw original-CAD ADD | 639/736 | Complete selected/source stream; existing foreground-plane inputs `[100,130)` |
| mustard stride4 | 185 / 184 | raw original-CAD ADD | 157/184 | Complete predefined every-fourth-input stream of a 737-input source |

The primary denominator includes every selected GT-defined tracking input, including hidden, unavailable and invalid candidates. Missing candidates are failures, and no finite-only mean is substituted for a stream containing missing candidates. Empty means in `summary.csv` are intentional. Registration input zero is excluded from tracking scores. Failure means error greater than 0.1 original CAD diameter.

The cracker clean motion ADD mean is 5.280 mm. Cracker blanking produces no valid 3D candidate for 20 hidden inputs; all 40 post-blanking inputs have successful motion ADD, with post-blanking mean 5.596 mm. The first five consecutive correct candidates start at reappearance input 40 and confirm at input 44. This is only a candidate-level short-prefix recovery observation.

The longer mustard occlusion probe has no valid 3D candidate in any of the 607 inputs from reappearance input 130 to the end. Its candidate recovery is right-censored. The stride4 probe also fails on 157 of 184 tracking inputs. The short cracker result therefore does not establish long-sequence recovery. Secondary unhidden/post-event metrics are labelled separately and never replace the primary denominators.

## Coordinate gauges and GT boundaries

Cracker probes initialize anchors in the first camera's coordinate frame. They have no independently bound original-CAD absolute pose. Post-hoc evaluation uses one fixed gauge, `C = inverse(P_0) * GT_0`, and scores `P_t * C` on original CAD vertices. This cancels first-frame pose error by construction and measures relative motion only. It cannot establish initialization, absolute pose, generated-shape or identity accuracy.

Mustard probes initialize from an independently frozen runtime first-frame registration in the original CAD coordinate system. Their raw candidate ADD is therefore coordinate-bound and reported separately from the optional first-frame-gauge-aligned motion diagnostic. GT was accessed only after inference; it did not supply the registration or candidate seeds. There is no scale fit, per-frame/trajectory alignment, or GT-rotation repair.

`summary.csv` contains all 16 reported metric/scope rows, including zero-denominator secondary scopes. `recovery.csv` specifies five-consecutive-candidate confirmation; it does not use or claim the separate end-to-end controller's recovery criterion. All four probes passed their recorded source/evidence checks. That validates the evidence contract, not the candidate poses. No newer multi-cohort experiment or unexecuted result is included.
