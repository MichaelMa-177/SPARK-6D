# Tentative publication during a two-stream conflict: development replay

This output-only rule added **41 tentative publications**, all on the foam object. It also added **2 wrong publications at MSSD < 0.05 diameter** and **29 wrong publications at MSPD < 5 normalized pixels**. These regressions are retained in every table. The result is a coverage/error tradeoff on known development data, not an improvement in the pose trajectory. The rule is not enabled as the production default.

The underlying poses are unchanged byte-for-byte. Among the 41 newly published candidates, all pass ADD <= 0.1 diameter, MSSD < 0.1 diameter and MSPD < 10 normalized pixels. Passing these looser criteria does not remove the stricter-threshold failures. All seven runtime trust flags stay false.

## Main numerical results

Across all 7,522 real tracking frames, publication increases from 6,497 to 6,538 (coverage 86.373305% → 86.918373%). For the foam object alone, it increases from 1,819 to 1,860 of 1,870 tracking frames (97.272727% → 99.465241%). Other real objects are unchanged. No initialization changes.

| Threshold | Wrong publications, baseline → research | Newly wrong | Error risk among known published, baseline → research | Rejected threshold-passing candidates, baseline → research |
|---|---:|---:|---:|---:|
| MSSD_over_diameter < 0.05 | 1886 → 1888 | 2 | 0.29028783 → 0.28877333 | 169 → 130 |
| MSSD_over_diameter < 0.1 | 1 → 1 | 0 | 0.00015392 → 0.00015295 | 996 → 955 |
| MSPD_640px < 5.0 | 3383 → 3412 | 29 | 0.52070186 → 0.52187213 | 24 → 12 |
| MSPD_640px < 10.0 | 504 → 504 | 0 | 0.07757427 → 0.07708779 | 490 → 449 |
| ADD_over_diameter <= 0.1 | 1588 → 1588 | 0 | 0.24442050 → 0.24288773 | 685 → 644 |

At MSSD < 0.1 diameter, the original raw trajectories have 30 failures across the real tracking frames. Both output policies still publish 1 of these failures and reject 29; rejected threshold-passing candidates decrease from 996 to 955. No candidate pose was corrected.

The existing fixed BOP19 tracking subset has **297** frames, from 74 + 74 + 74 + 75. It contains **zero interventions**; all its metrics and publication decisions are unchanged. The historical v1 posthoc plan manually stated 296 although its counted scopes and baseline comparisons already used 297. Version 2 derives this metadata from the prior frozen scope, adds consistency assertions and explicit trust counts. All v1 numerical threshold and paired tables are preserved and unchanged.

## Fixed scope and same-appearance control

Six unique input streams comprise four full real runs (7,526 inputs / 7,522 tracking frames) and two historical synthetic streams (360 inputs / 358 tracking frames): 7,886 unique inputs and 7,880 unique tracking frames overall. The hidden synthetic stream has two world annotations, `stay` and `exchange`, for identical observations. Thus the label join has 8,066 rows / 8,059 tracking labels, not 8,066 independent observations. Only real cases are pooled.

The continuous synthetic stream retains 161/179 tracking publications. The hidden stream retains 54/179 in each world. At MSSD < 0.1 diameter, both policies publish zero wrong candidates in each world; rejected correct/wrong candidates are 11/7 for continuous, 78/47 for hidden stay, and 1/124 for hidden exchange. All synthetic results are unchanged.

These historical synthetic records have **no genuine secondary bridge**, so they do not challenge the new two-live-branch rule with an impostor. They test preservation of historical behavior and the identity-loss latch only. A logical CPU counterexample already shows that one wrong candidate agreeing with one wrong stream can pass the new tentative rule; this policy does not verify identity or geometry.

## Frozen rule and causal limit

The rule permits tentative output only when the original publication veto is a two-stream motion conflict, current observation support is valid, identity loss is not latched, both original and secondary branches are live, and exactly one branch passes the unchanged candidate-fit criterion. It changes only saved publication status/reason/availability. Candidate bytes, branch proposals, conflict, point seed, thresholds, neural refinement, memory and trust flags stay unchanged.

The candidate participates in choosing the agreeing branch. Its dependencies can overlap with branch tracking or earlier pose state; the check is not independent evidence. Development labels motivated the rule before this experiment. There is no held-out result, pose-accuracy gain, trusted relocalization claim, inference timing improvement or CVPR-level performance claim here.

## Metrics and files

`summary.json` contains all cases/worlds, four scopes, 21 original thresholds, three output variants, six-cell publication/error counts, risk/coverage, paired transitions, synthetic association summaries and explicit trust counts. `threshold_counts.csv` has 2,016 variant rows; `paired_transitions.csv` has 672 threshold/scope rows, each containing 12 availability/label cells. These are counts, not per-frame poses or labels.

The three variants are the qualified raw candidate, the original publication decision and the research publication decision. Each fixed scope includes correct, wrong and unknown candidates crossed with publication/abstention. `published_wrong / N` retains the full denominator; selective error uses its explicitly stated published/known denominator. Unknowns remain explicit, with lower/upper risk bounds. Every zero-denominator rate is null (blank in CSV). `abstained_wrong` means a rejected candidate fails that metric, not that its physical identity was verified. `abstained_correct` means a threshold-passing candidate was withheld.

ADD uses <= 0.1 of each object's original diameter. MSSD uses the unchanged strict diameter-relative grid 0.05 through 0.50. MSPD uses the unchanged strict 640-pixel-normalized grid 5 through 50. Original floating-point threshold values are preserved. Synthetic BOP scopes are undefined and marked `defined=false` / `scope_defined=false`; their empty N=0 placeholders and null rates are not BOP performance estimates.

All 336 real baseline threshold/scope cells match the previous posthoc summary exactly. An independent reviewer recomputed the complete numerical join and checked v1/v2 numerical equality. Sources and review are identified by hashes in `sources.json`. This package contains only aggregate numerical results, paired counts, provenance hashes and explanations. It contains no raw GT, per-frame trajectory, pose matrix, image, weight, model, log or source-code file.
