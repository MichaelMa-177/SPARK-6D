# TUD-L static click initialization: coverage failure retained

This package contains actual results for **96 static assistant-click observations**:
3 supplied CAD objects × 8 recorded lighting labels × 4 observations per cell.
The 24 cells are not 24 videos. Lighting and viewpoint vary together, so these
results do not establish a causal effect of lighting, continuous tracking,
relocalization, or physical-instance identity. This is a development pilot.

At the frozen threshold **error ≤ 0.1 × official model diameter**, both full-vertex
ADD and identity-only MSSD give the following results:

| Outcome | Full 96-observation denominator |
|---|---:|
| Actual FoundationPose top-1 geometrically correct |95/96 (98.9583%)|
| Observations with a qualified correct candidate in the bank |95/96|
| Qualified trusted admissions |0/96|
| Correct admitted poses |0/96|
| False rejections under the frozen geometric label |95/96|
| True rejections under that label |1/96|

**Useful admission coverage failed.** Zero false acceptances cannot establish
success when nothing is accepted. Accepted-output conditional accuracy is
undefined (`null`), not 100%. The 96 labels are known. Geometric correctness does
not verify physical identity or uniquely resolve absolute orientation.

The PNG/SVG/PDF figures show the actual top-1, admitted output and qualified bank
recall separately, preserving all four inputs in every object/light cell. CAD 1
has 31/32 top-1 successes; CAD 2 and CAD 3 each have 32/32.

## Appearance preferences: show coverage with conditional error

Phase 2 retains **384 requests**: 192 appearance requests and 192 pure CPU replays,
covering the same 96 banks under albedo and fixed-light rendering. These requests
and candidate pairs are not independent experimental samples. Each request has
252 candidate slots and 31,626 unordered pairs; every variant/channel/metric
therefore retains **6,072,192 pair slots**. Channel and metric rows reuse those
slots and must not be summed as distinct observations.

For ADD, there are 749,312 informative pairs (one correct and one incorrect
candidate) per variant. The appearance gradient channel has 24,273 correct
preferences and 3,225 wrong preferences: decisive informative coverage is
3.6698%, with conditional error
11.7281%.
It leaves 709,610 informative pairs unknown and 12,204 with fold disagreement.
The affine channel has only 42 correct and 13 wrong decisive preferences:
coverage 0.007340% and conditional error
23.6364%.
Thus a small conditional error, or a large unknown fraction, is not sufficient
evidence of a useful selector. This diagnostic did not rerank candidates,
change admission, update trusted memory or make new neural predictions.

`pair_counts.csv` includes all/albedo/fixed-light aggregates for both channels
and both audited metrics. The all row overlaps the two rendering subsets.
`pair_transitions.csv` records appearance→replay states; this pure replay is
not a new learned method. Only the interrupted request changes qualified
states to unknown; all completed request states are retained.

One request, `obs000_fixed_light` replay, was interrupted and remains unqualified:
**all 31,626 of its pairs remain unknown**, with no retry or denominator removal.
The pause/recovery amendment started only the 190 previously unstarted replays;
it did not rerun completed appearance work or the completed replay. The other
383 requests are qualified. Original continuous wall time is unavailable, and
no deployment FPS is claimed.

## Numerical audit and limits

An independent posthoc audit passed for all 24,192 candidates and 96 actual top-1
outputs under ADD and identity-only MSSD, 48,577,536 metric-pair state slots,
242,887,680 stored pair-array elements and 24,288,768 transitions. Maximum
absolute numeric difference was 4.440892098500626e-16 m;
bound sources and inputs were unchanged. CAD vertex-row counts were 32,456,
52,204 and 3,471. Model coordinates were converted mm→m once; raw rotations
were not repaired. This audit used GT after the runtime/raw gates; the package
builder and plotting tool do not open GT pose files.

**ADI is a separate producer-only auxiliary diagnostic and was not independently
numerically audited.** Its counts are explicitly isolated in `summary.json`;
they do not support the independently checked primary claims or figures.
No threshold was tuned from this report. Pair counts must not be treated as
independent samples for confidence intervals or significance claims.

The package contains 11 payload files plus `manifest.json`: aggregate JSON/CSVs,
this explanation and six generated figures. It contains no raw predictions,
GT poses, source images, source code, models, weights or absolute local paths.
`initialization_cells.csv` has 48 rows: 24 object/light cells × 2 audited
metrics, with the three plotted measures and other counts in columns. The
plot tool internally emitted 144 rows (24 cells × 2 metrics × 3 measures);
that redundant internal CSV is not part of this package.

Source and audit hashes are in `summary.json`; payload hashes are in the manifest.
