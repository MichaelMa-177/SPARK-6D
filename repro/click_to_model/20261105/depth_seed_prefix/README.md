# Current-depth seed: three-frame development comparison

Two existing runs share the first three RGB-D inputs, click, mesh and 29
byte-identical initialization arrays for YCB-V scene 57 / object 21 (foam brick).
Each arm includes one initialization and two tracking frames. No runtime read GT.
The baseline is a prefix of an existing 1,871-input run. This is a very short
known development case, not a held-out or relocalization benchmark.

| Two tracking frames | Original point seed | Current-depth seed |
|---|---:|---:|
| Mean ADD (mm) | 3.539691 | 3.498169 |
| Mean symmetry-aware MSSD (mm) | 4.433953 | 4.324687 |
| Mean MSPD (640-pixel width) | 4.268573 | 4.205817 |
| Wrong at MSSD < 0.05 diameter | 1 | 0 |
| Wrong at MSSD < 0.1 diameter | 0 | 0 |
| Published / tracking inputs | 2 / 2 | 2 / 2 |
| Instrumented two-frame tracking time (s) | 0.284373 | 0.924077 |

All five retained error metrics improve on both tracking frames; initialization
is unchanged. The ADD improvements are only about 0.002308 and 0.080737 mm.
The first tracking frame crosses the strict 0.05-diameter MSSD boundary by a
small margin. Preserve this magnitude and both looser-threshold zero-error
results rather than interpreting the crossing as broad success.

The depth proposals were actually supplied to FoundationPose before its real
refinement returns. They took eight and 24 local iterations respectively. The
second reached its iteration cap with oscillating correspondence counts.
Neither proposal has a convergence or physical-instance identity certificate.
Both arms remain tentative; publication is not identity verification.

`thresholds.csv` retains 336 rows: two arms, four scopes, 21 frozen thresholds
and candidate/published variants. `paired_transitions.csv` retains all 168
paired C/W/U cells. Scopes are initialization, all inputs, all tracking and the
fixed official BOP19 tracking subset. That last subset has **zero frames** here:
counts remain zero and rates remain null/blank, not a perfect official result.
ADD uses <=; MSSD and MSPD use strict <. Abstention is retained explicitly.

`frame_directions.csv` retains each input's improvement/worsening/unchanged/
unknown direction for all five metrics, without raw poses or GT matrices.
`summary.json` gives aggregate errors, counts, actual iteration limits and
timing. The original 8,455-vertex mesh and two declared symmetry transforms are
used; no GT alignment, alternate hypothesis selection or pose repair occurs.
The recomputed baseline prefix matches its earlier full-sequence posthoc in
every field.

Timing includes observer, evidence and refinement overhead. It is slower in the
depth-seed run. Baseline whole-process times in the summary remain scoped to
1,871 inputs; no three-frame baseline process time is invented. The two recorded
tracking latencies per arm are compared separately. These are not deployment
FPS or controlled causal speed measurements.

The exporter independently recounts the six scalar rows and all threshold,
paired, direction and timing entries without importing the producer's counting
functions. It does not independently recompute BOP geometry. Sources are pinned
by sanitized hashes. The package contains only aggregate numerical results,
per-input direction labels, source hashes and this explanation: no source code,
raw poses, raw GT, point tracks, images, model files, weights, binaries or logs.
