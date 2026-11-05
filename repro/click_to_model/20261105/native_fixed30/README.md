# Fixed30 object-specific native-head adaptation: development results

The treatment updates 3,078 private translation/rotation-head parameters, validates
a shadow proposal on fixed current-depth surface samples, and applies accepted
parameters only to the next frame. The original pretrained backbone is unchanged.
The baseline is the exact first 30 inputs of the previously completed 180-input
online run; the treatment is a separately executed 30-input run. Initialization
matches in all 29 checked arrays, including SAM outputs and three 252-hypothesis
registration banks. Runtime inference never read GT.

| Tracking quantity (29 frames per arm) | Baseline | Native adaptation |
|---|---:|---:|
| Published /29 | 29 | 29 |
| Mean ADD (mm) | 2.029257511 | 2.006519832 |
| Mean MSSD / diameter | 0.01678068111 | 0.01665091616 |
| Mean MSPD (normalized640 pixels) | 1.870757398 | 1.856444519 |
| Instrumented tracking time (s) | 2.503435 | 20.001901 |
| Instrumented tracking throughput (frames/s) | 11.584085 | 1.449862 |

Both arms have 29/29 correct tracking candidates and 29/29 published candidates at
ADD <=0.1D, MSSD <0.1D and MSPD <5 normalized pixels. ADD decreases on
27 tracking frames, increases on1, and is exactly equal on1.
The mean reduction is only0.022737679 mm; headline success counts do not improve.
The substantial instrumented latency increase is retained. This short known
synthetic development prefix is not a heldout test, a relocalization experiment,
proof of physical identity, or evidence of general accuracy improvement.

All 21 frozen thresholds and all 30 / tracking 29 / init 1 populations are retained:
`thresholds.csv` has 252 cells and `paired_transitions.csv` has 126 paired cells.
ADD uses an inclusive threshold; MSSD/MSPD use strict thresholds. Availability,
unknown errors and abstention remain separate. Risk for an empty population is
null, represented by an empty CSV field. The summary includes both persistent
target and distractor errors and all paired better/worse/equal counts.
All seven trust flags remain false; published outputs are tentative.

The actual run records 29 accepted updates and final private version 29. Effective
pose changes are bound to returned and saved poses by the raw review. Proposal
gradients/shadow losses remain source-verified telemetry: shadow parameter tensors
and per-parameter gradients were not saved for independent CUDA recomputation.
Telemetry and whole-process peak memory are kept separate from GT performance.

An earlier three-input v1 experiment exposed stale AMP cached private weights
after SGD. Its apparent unchanged check loss did not validate the updated model.
That failure is preserved. The corrected private autocast disables its cache;
CPU freshness checks and subsequent real GPU execution were completed before
this run. No learning-rate or geometry threshold was retuned using these labels.

Timing sums only the first 29 actual tracking latencies in each arm. Baseline
full 180 parent/child totals remain explicitly labeled 180; its prefix parent wall
is null. Treatment parent/child totals cover 30 inputs. These instrumented scopes
contain observation, rendering, gate, gradient and evidence work and are not
deployment FPS. Cache/scheduling differences are uncontrolled; a causal slowdown
factor is not established.

This package contains numeric aggregates, source hashes and this explanation only.
No source code, raw pose/GT matrices, images, models, weights or logs are included.
The exporter independently recounts 60 scalar and publication rows, 252 threshold
cells, 126 paired transitions and 58 latency entries; it does not rerun the geometric
GT evaluator. Source files are hashed before and after export.
