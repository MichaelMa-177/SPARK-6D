# Wood: actual translation-head adaptation

This fixed development control compares the actual SAM-score Wood run with its
actual private translation-head adaptation run. It contains 1,829 inputs: one
initialization and all 1,828 tracking frames. The predeclared 75 BOP tracking
targets are reported separately. The 29 initialization arrays are byte-identical;
1,820 tracking poses changed. No baseline trajectory is substituted.

| Tracking metric | SAM-score baseline | Adaptation |
|---|---:|---:|
| MSPD < 5 normalized pixels | 283 / 1,828 | 279 / 1,828 |
| MSPD < 10 normalized pixels | 1,622 / 1,828 | 1,624 / 1,828 |
| MSSD < 0.05 diameter | 1,669 / 1,828 | 1,669 / 1,828 |
| Mean ADD | 6.708880 mm | 6.714021 mm |
| Mean MSSD | 8.615140 mm | 8.620865 mm |
| Mean MSPD | 7.354137 px | 7.363669 px |

The changes are mixed and do not establish a useful accuracy gain. The stricter
MSPD count and all three mean errors worsen. On the fixed 75 BOP targets, MSSD
< 0.05 diameter falls from 72 to 69 correct; MSPD < 5 pixels falls from 14 to 13.
MSSD < 0.1 diameter and ADD <= 0.1 diameter remain correct for all tracking frames.
Threshold equality is strict for MSSD/MSPD and inclusive for ADD. Symmetry-aware
MSSD uses all 8 declared finite transforms and 8,920 original model vertex rows.
Plain ADD can also change with the chosen symmetry representative.

The private 1,539-parameter head updates on 1,815 frames and resets on 13. The
saved final-callback AMP comparison is nonzero on 1,815 frames; all 1,828 selected
head outputs match the original raw audit. These are execution checks, not proof
of useful refinement. Earlier callbacks have version assertions and counts, but
not separately saved head features. Original pretrained weights are unchanged.

Actual tracking wall increases from 352.213 to 457.643 seconds (5.190 to 3.994
instrumented tracking FPS). Parent wall increases from 395.336 to 500.985 seconds.
Recorded adaptation work is 104.376 seconds contained within tracking wall.
These overlapping timers must not be summed. This sequential one-run comparison
does not establish deployment throughput or isolate all timing causes.

All seven trust flags remain false; there is no explicit abstention intervention.
Every working output is retained, including wrong and unknown cases. Self-verified
coverage is zero and its error risk is undefined, never a successful abstention.
The first CPU posthoc attempt failed on an absent summary field before reading GT;
the versioned metadata fix preserves that failure and reuses the same GPU outputs.

`threshold_counts.csv` contains 252 count rows for both arms, three scopes and both
working/self-verified selections. `paired_transitions.csv` retains all 63 full 3x3
correct/wrong/unknown transitions. `continuous_errors.csv` includes all-input,
initialization, tracking and fixed-BOP means, quantiles, maxima and improved/worsened
counts. Independent scalar review checks the existing evaluated numbers and
clocks; it does not repeat GT geometry, neural inference or the raw-data audit.

Only numeric aggregates and source hashes are included: no raw poses, GT matrices,
per-frame trajectories, images, weights, executable source or local filesystem
paths. This is one known development sequence, with no held-out result,
independent-frame confidence interval or physical-identity verification claim.
