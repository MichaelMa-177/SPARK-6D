# Full180 native-head adaptation: development results

Both runs consume the same 180 RGB-D inputs, first-frame click and mesh. All 29
initialization arrays and all 180 actual input packets are paired exactly.
The treatment extends the unchanged fixed30 native adaptation to the full known
synthetic sequence, including partial occlusion [100,130). It updates 3,078
object-specific head parameters and applies accepted parameters to the next frame.
No learning rate, sampling, trust or geometry threshold changed. Runtime did not
read GT. Initialization remains separate from all 179 tracking frames.

| Quantity (179 tracking frames) | Baseline | Native adaptation |
|---|---:|---:|
| Published / 179 | 161 | 161 |
| Abstained / 179 | 18 | 18 |
| Mean ADD (mm) | 2.662753402 | 2.497705650 |
| Candidate wrong, ADD <= 0.1D | 2 | 0 |
| Published wrong, ADD <= 0.1D | 0 | 0 |
| Correct candidate abstained, ADD <= 0.1D | 16 | 18 |
| Candidate wrong, MSSD < 0.1D | 7 | 7 |
| Published wrong, MSSD < 0.1D | 0 | 0 |
| Correct candidate abstained, MSSD < 0.1D | 11 | 11 |
| Candidate wrong, MSPD < 5 normalized pixels | 11 | 11 |
| Published wrong, MSPD < 5 normalized pixels | 0 | 0 |
| Correct candidate abstained, MSPD < 5 normalized pixels | 7 | 7 |
| Instrumented tracking time (s) | 15.640065 | 122.081870 |
| Instrumented tracking throughput (frames/s) | 11.444965 | 1.466229 |

| Phase | Tracking inputs | Baseline mean ADD (mm) | Adapted mean ADD (mm) |
|---|---:|---:|---:|
| before_occlusion | 99 | 3.151163424 | 2.950665337 |
| occlusion | 30 | 1.935554969 | 1.829492670 |
| after_occlusion | 50 | 2.132020617 | 2.001773258 |

Candidate ADD decreases on 171 tracking frames, increases on
7, and is exactly equal on 1. All paired metric
changes are retained in summary.json. MSSD/MSPD headline failure counts and
publication availability do not improve. ADD-correct candidates withheld by the
gate increase. The larger instrumented runtime is retained; this is not evidence
of a general stable-tracking, identity-recovery or speed improvement.

The three tracking phases are [1,100), [100,130), [130,180), with 99/30/50 inputs.
They partition all 179 tracking slots and exclude initialization. All 21 frozen
thresholds are reported for all180/tracking179/init1, yielding 252 main threshold
cells and 126 paired-transition cells. Phase files add 252 threshold cells and
126 paired cells. Target and distractor errors, candidate/published populations,
unknowns and abstentions are retained. ADD uses <=; MSSD/MSPD use strict <. Empty
CSV fields represent null, including risk with no published population.

There are 268 actual native callbacks, 161 accepted
updates and final private version 161.
All 19 callbacks lacking current permission made zero private forward
calls and left effective output unchanged. The raw audit checks next-frame
version continuity and actual-return-to-saved-pose bytes. A final-frame committed
update has no subsequent observation here. Update-only seconds were not measured
and remain null; full tracking latency must not be relabeled as training time.
Phase update counts and first/last source-frame indices are included in telemetry.

An earlier v1 short experiment exposed stale AMP-cached private weights after SGD.
That invalid shadow validation remains excluded. This experiment inherits the
previously checked private cache-disabled objective. Shadow tensors and individual
parameter gradients were not persisted, so proposal losses remain source-verified
telemetry rather than independently recomputed CUDA evidence.

All seven trust flags remain false. Published poses are tentative. This is one
known synthetic development sequence with correlated frames; it does not establish
heldout accuracy, physical instance identity, general relocalization or CVPR-level
success. Whole-process peak memory includes models, initialization and full journals.
Timing includes instrumentation and evidence work; it is not deployment FPS.

This exporter independently recounts 360 scalar and publication rows, all 504
threshold cells, 252 paired cells, 72 phase-error summaries and 358 tracking
latencies. It imports no producer counting functions and does not open original
GT matrices, raw poses, images, weights or mesh. It does not recompute geometric
GT errors. Source bytes are hashed before and after export. The earlier CPU raw
audit import-path failure remains preserved; corrected invocation used the same
audit source and did not rerun GPU inference.

Only numeric aggregates, source hashes and this explanation are included. No
source code, raw pose/GT matrices, images, model, weights or logs are packaged.
