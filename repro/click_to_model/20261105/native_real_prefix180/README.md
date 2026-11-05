# Actual native private-head refinement on four real 180-frame prefixes

**This development result is mixed, small in accuracy effect, and substantially more expensive in the observed instrumented timing.** Scissors mean tracking ADD worsened from 2.217093 to 2.262072 mm; clamp ADD increased slightly. Gelatin improved from 3.207500 to 3.151756 mm. Across 716 tracking slots, stored ADD increased on 113, decreased on 192, and was exactly equal on 411. Exact scalar signs are descriptive; no significance claim or epsilon threshold is implied. No production-default, heldout, identity-verification, or CVPR-level claim follows from this experiment.

The native private-head intervention ran through the actual frozen GPU runtime on all four objects. The comparison baseline is each original full real run's first 180 actual inputs and outputs. The same frozen synthetic-run parameters were used without object-specific tuning. Each object's mesh, metric units, centering, intrinsics and declared symmetries were bound to its real capsule. The 116 initial arrays and all 720 RGB/depth/K input pairs were byte-exact against baseline. The independently recomputed 720 baseline scalar rows exactly matched the prior full-run posthoc prefix in every field.

## Tracking results

Each row has **179 tracking inputs**. Numbers show baseline → native. Errors are candidate errors, including candidates that were abstained; publication is listed separately.

| Scene/object | Mean ADD (mm) | MSSD < 0.1D errors | MSPD < 5 px errors | ADD ≤ 0.1D errors | Published | Mean recorded tracking latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| 51/17 scissors | 2.217093 → 2.262072 | 0 → 0 | 8 → 7 | 0 → 0 | 179 → 179 | 0.125478 → 0.709128 |
| 54/19 clamp | 54.980768 → 54.980790 | 0 → 0 | 179 → 179 | 179 → 179 | 173 → 173 | 0.176672 → 0.996413 |
| 57/21 foam | 3.508529 → 3.508347 | 0 → 0 | 44 → 44 | 0 → 0 | 179 → 179 | 0.176475 → 0.991557 |
| 58/8 gelatin | 3.207500 → 3.151756 | 0 → 0 | 75 → 74 | 0 → 0 | 179 → 179 | 0.141882 → 0.815531 |

The full grid has no correct→wrong threshold crossing, but continuous errors still worsen on many frames. Only three tracking metric/threshold cells gain wrong→correct transitions: scissors MSPD < 5 px (+1), gelatin MSSD < 0.05D (+3), and gelatin MSPD < 5 px (+1). These counts overlap across metrics and scopes and **must not be summed as five distinct improved frames**. Gelatin's fixed BOP tracking subset gains one MSPD < 5 px correct slot. All other tracking or fixed-BOP threshold cells retain their original counts; all-input cells include the same tracking changes.

Both arms publish 710/716 tracking candidates. At MSSD < 0.1D, all 716 candidates are correct but six are rejected. At MSPD < 10 px, 23 incorrect candidates are still published and six correct candidates rejected. At ADD ≤ 0.1D, all 179 clamp candidates fail, including 173 published outputs; the symmetry-aware MSSD result does not establish resolved absolute orientation. Strict MSSD < 0.05D published errors change 236→233; MSPD < 5 px published errors change 300→298. All seven runtime trust flags remain false; tentative availability does not establish physical identity.

## Adaptation and refusal outcomes

There are 1,170 callbacks: 438 qualified and 732 refused. Refusals are 726 `insufficient_disjoint_surface` and six `current_candidate_permission_absent`; all remain counted. Of 716 tracking slots, 308 produce gradient proposals, 205 commit and 103 reject; 408 produce no gradient proposal because the frame contains an unqualified callback. Commit counts are scissors 47, clamp 9, foam 26 and gelatin 123. A lower recorded check loss alone does not imply acceptance; the package preserves rejected proposal loss directions as telemetry rather than inventing a sole rejection reason.

The CPU raw audit checks transfer, actual candidate permission, publication, and holder progression. It does not independently recompute gradients or reload unavailable terminal live model state. Gradient/check losses retain their runtime-telemetry limitation. Qualification/refusal counts are callback counts; proposal/commit counts are tracking-frame counts and use different denominators.

## Protocol, timing and contents

The fixed population is four known development sequences, 720 inputs, 716 tracking inputs and four initializations per arm. BOP first-180 input counts are 8/13/9/15 (45 total); BOP tracking counts are **7/12/8/15 = 42**. Original BOP vertices and symmetry geometry are unchanged. The 21 original thresholds are MSSD/D 0.05 through 0.5 with strict `<`, MSPD 5 through 50 px with strict `<`, and ADD ≤ 0.1 object diameter. Original floating-point threshold values are retained exactly. Unknown slots, abstentions, undefined ratios (`null`), and all adverse outcomes remain represented.

Timing uses each baseline's first 179 real latency samples. Baseline prefix parent and child wall times are unobserved and remain `null`; full-run wall times are never substituted. Native wall times describe its actual 180-frame run. GPU cases ran serially, while single-thread CPU raw audits/source preparation could overlap. Baselines were not simultaneous paired timing runs. These are descriptive instrumented measurements, not a precise speedup estimate or deployment FPS.

- `arm_cells/`: all **672** original per-case/arm/scope/threshold cells, preserving each scope object exactly.
- `paired_cells/`: all **336** original paired cells, preserving candidate 3×3, published 3×3, availability 2×2, and publication/label 6×6 tables exactly. Published unknown includes abstention; the 6×6 table keeps these states distinct.
- `threshold_changes/`: all paired threshold deltas, including zeros and regressions if present.
- `continuous_error/`: aggregate baseline/native/delta distributions and better/worse/equal counts for every case/scope/metric; no per-frame labels.
- `pooled_scopes/`: count sums over all four objects, with each object's original ADD threshold retained.
- `run_summary.json`: exact denominators, descriptive timings, source-geometry scalar metadata, and adaptation/refusal counts.
- `source_hashes.json`, `cell_inventory.json`, and verification receipts: provenance and completeness checks.

Every file is smaller than 70,000 bytes. Only numerical aggregates, paired counts, source hashes and documentation are included. There are no GT matrices, pose arrays, trajectories, images, meshes, weights, logs, or new source-code files. The first local builder attempt stopped at a failed proposal-count assertion because refusal records also occupy `proposal_recorded`; its failed directory is retained locally. The corrected builder counts gradients by the original collector's `gradient_norm` field and preserves non-gradient refusal reasons. No runtime result, metric value, threshold or protocol was changed.
