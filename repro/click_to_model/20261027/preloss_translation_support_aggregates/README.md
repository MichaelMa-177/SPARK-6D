# Translation-triggered secondary support: one synthetic development sequence

Two fresh online runs use the same 180 RGB-D inputs, click, mesh, SAM-score rule,
TAPNext assets and FoundationPose configuration. Only the experimental secondary
support birth policy changes. Initialization is frame 0; all 179 tracking frames
remain in each denominator. No ground-truth alignment or pose substitution is used.
This small improvement on one known synthetic development segment is not evidence
of general relocalization, held-out accuracy, or physical-instance identity recovery.

Tracking results (baseline → birth policy):

| Quantity | Baseline | Birth policy |
|---|---:|---:|
| Published / 179 | 161 | 162 |
| Abstained / 179 | 18 | 17 |
| Mean candidate ADD (mm) | 2.662753 | 2.516243 |
| Candidate wrong at ADD ≤ 0.1D | 2 | 1 |
| Correct candidates rejected at ADD ≤ 0.1D | 16 | 16 |
| Candidate wrong at MSSD < 0.1D | 7 | 6 |
| Correct candidates rejected at MSSD < 0.1D | 11 | 11 |
| Candidate wrong at MSPD < 5 normalized pixels | 11 | 8 |
| Correct candidates rejected at MSPD < 5 normalized pixels | 7 | 9 |
| Instrumented tracking time (s) | 15.6401 | 24.9195 |
| Instrumented tracking throughput (frames/s) | 11.4450 | 7.1831 |
| Child wall time (s) | 38.2246 | 47.5062 |
| Parent wall time (s) | 40.9623 | 50.1652 |

At those three thresholds, published wrong and unknown counts are zero in both
runs, but correct rejection remains substantial. Availability is tentative; all
seven trust flags stay false, verified count is zero, and verified risk is null.
Zero errors in selected outputs does not establish calibrated risk or unique identity.

The actual registration has 29/29 matching initialization arrays. All 180 actual
candidate poses were compared: 162 byte-exact and 18 different, at frames 37–54.
The new secondary stream is born at frame 4 and becomes ready at frame 7; this is
an observed runtime event, not a GT-selected threshold. Temporary abstention changes
from frames 37–54 (resume at 55) to 37–53 (resume at 54). At frame 54 the new candidate
ADD is **2.296520 mm**, worse than **2.067845 mm**, despite earlier publication.
The predefined partial-occlusion interval [100,130) is retained separately.

Both target and distractor are fixed persistent objects. Membership uses each
object's ADD ≤ 0.1D independently, preserving target/distractor/both/neither/unknown;
there is no nearest-GT reassignment. Summary metrics cover both objects on all 180
inputs, 179 tracking inputs, and the single initialization. The full original mesh
has 386 vertex rows, 768 faces, diameter 0.1649631473996541 m, and the inherited
identity symmetry specification. MSPD is normalized by 640 / image width (480).

`thresholds.csv` contains 252 target-correctness cells: 126 per arm, from 3 scopes ×
21 frozen thresholds × 2 output variants. ADD uses ≤; MSSD/D and normalized MSPD
use strict <. `paired_transitions.csv` contains all 126 paired C/W/U cells and all
nine transitions. Its unknown category includes explicit abstention in the gated
variant; threshold CSV columns keep abstention separate from unknown metric values.
CSV blank values represent null, never zero risk for an empty selected population.

`summary.json` includes both arms, target/distractor errors and association counts,
actual availability, timing boundaries, initialization pairing and the frame-54
counterpoint. `sources.json` binds the frozen posthoc, raw audits, actual scalar rows
and publication evidence by SHA-256. This exporter independently recounted every
cell and paired transition from those rows without importing the producer's counting
functions; it did not repeat geometric GT calculations. All 360 scalar and decision
rows and all 358 tracking latency entries were checked. Source bytes were unchanged.

Timing includes instrumentation, current-frame continuity decisions and publication
evidence. Tracking timing excludes final output serialization. Initialization,
registration and admission timers overlap; do not add them. Child/parent totals
include broader setup and output work. Cold loads, cache and scheduling are not
controlled. These numbers are not deployment FPS or a causal performance estimate.
The birth run is slower in all three reported wall-time scopes.

Only numeric aggregates, this explanation, source hashes and the package manifest
are included. No raw poses, GT matrices, point tracks, images, models, weights, source
code, logs or absolute local paths are included. Correlated frames are not independent
trials; no confidence interval, parameter selection or general efficacy claim is made.
