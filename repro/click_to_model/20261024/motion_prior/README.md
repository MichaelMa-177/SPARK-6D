# Constant-velocity reference development experiment

This numerical increment contains all six frozen runs of adaptive SPARK-6D with the trusted-view point backend, comparing the existing temporal reference with an optional constant-velocity reference. The three fixed input protocols use the same reused mustard trajectory. These are single development runs, not independent sequences or timing replications.

The prior is formed only from the previous two consecutive successful correspondence commits. It changes the reference for the candidate's existing translation and rotation tests; their bounds remain 0.06876214029824114 metres and 0.8 radians. Independent point consensus, current CAD support and the subsequent neural-correction checks remain required. A failed commit clears the motion history. The prior does not replace the raw neural pose and does not extrapolate through lost frames. Equal spacing of selected inputs is an assumption.

| Protocol | Inputs / tracking | Final ADD failures, reference / CV | Final mean ADD, reference / CV |
| --- | ---: | ---: | ---: |
| Original selected stream | 737 / 736 | 0 / 0 | 3.635 / 3.635 mm |
| Every fourth source input | 185 / 184 | 163 / 0 | 644.324 / 3.635 mm |
| Synthetic foreground plane on inputs [100,130) | 737 / 736 | 636 / 636 | 300.038 / 300.038 mm |

All selected GT-defined tracking inputs remain in the primary denominator, including hidden inputs and uncertain states. Registration is excluded. ADD uses every original CAD vertex and unmodified raw GT. Failure means ADD greater than 19.646325799497472 mm. No trajectory alignment, scale fit, GT-frame reordering or GT-rotation repair is used. The sampled runtime diameter and exact metric diameter happened to agree in these runs; they were reconstructed and checked separately.

## Candidate mechanism check

The independent post-hoc check evaluates each stored point candidate itself in the original CAD frame: original pose = centered pose multiplied by the translation of negative mesh bounding-box center. The saved original and centered registration poses verify this conversion. Candidate correctness never inherits the final neural pose's correctness.

The two-reference table compares the same candidate with both logged references within the actual CV trajectory. There are 734, 182 and 97 eligible inputs for the original, stride4 and occlusion CV runs respectively. These eligible counts do not replace the complete tracking denominators. Separate method runs develop different memories and candidates and are not strict framewise counterfactuals.

Only one eligible candidate changes its temporal admission: selected input 22, original source input 88 of the stride4 stream. The previous reference rejects it, while CV admits it. Its own original-CAD ADD is 6.244 mm, and actual seed admission and memory commit both succeed. The remaining eligible candidates pass both references. No candidate changes from accepted to rejected in these recorded runs; direction-dependent rejection is therefore not established by this real-data sample.

Stride4 candidate failures change from 162/184 missing candidates to 0/184; the existing-reference run has one additional correct candidate rejected by the temporal gate. Both original-stream runs have 0/736 candidate failures. Both occlusion runs have 637/736 missing candidates and no valid candidate after reappearance input 130. Their final-pose recovery remains right-censored over all 607 remaining inputs, using three consecutive successful final poses as the recovery rule.

Every candidate that exists in these six runs satisfies the ADD threshold. Consequently, zero observed false candidate admissions does not test rejection of incorrect present candidates or repeated-instance distractors. This experiment supports one useful motion-prior mechanism case; it does not establish general recovery, relocalization, unseen-scene robustness or physical-instance identity.

## Numerical contents

`inventory.csv` contains 30 tables: each run's raw final poses, tracking latencies, raw neural iterations, canonical per-frame metrics and independent candidate/innovation arrays. Adjacent schemas describe dtype, shape and the CSV layout. Split parts repeat the header and preserve all numerical rows in part order. String arrays are excluded. There is no source code, log, local absolute path, provenance digest, model weight, image, mesh or binary archive.

`summary.csv` keeps final-pose scores separate from candidate scores. Empty candidate means indicate missing predictions in the primary population; no finite-only average substitutes for the failed stream. Fields old0_cv0, old0_cv1, old1_cv0 and old1_cv1 count the two-reference temporal cells; actual seed and memory fields separately record completed admissions. The memory-versus-candidate counts associate a commit with its input candidate and do not substitute for final-pose accuracy. Candidate CAD observation diagnostics were not serialized separately, so a temporal pass alone must not be treated as actual seed admission.

Within candidate arrays, gate/state values -1, 0 and 1 mean unrecorded, false and true. Missing poses use NaN; missing candidate errors use positive infinity on GT-defined inputs. Registration has no tracking candidate. Full input indices, masks and availability flags are retained. `recovery.csv` contains both unsuccessful occlusion results. All six source/evidence contracts passed; that validates the records, not broader tracking claims.
