# Four real RGB-D sequence results

Four fixed YCB-V sequences: 51/scissors, 54/large clamp, 57/foam brick, and
58/gelatin box. All 7,526 inputs completed and passed full raw-data review.
This includes 7,522 tracking frames, four initializations and provenance for
3,024 initialization hypotheses. Geometry errors evaluate the delivered
candidate, not a GT-selected alternative hypothesis. Runtime did not read GT.

| Tracking population | Candidate correct / wrong / unknown | Published correct / wrong / unknown | Correct candidates abstained |
| --- | --- | --- | --- |
| All frames, MSSD < 0.1D | 7492 / 30 / 0 | 6496 / 1 / 0 | 996 |
| BOP19 subset, MSSD < 0.1D | 297 / 0 / 0 | 268 / 0 / 0 | 29 |
| All frames, MSPD < 10 px | 6483 / 1039 / 0 | 5993 / 504 / 0 | 490 |
| BOP19 subset, MSPD < 10 px | 253 / 44 / 0 | 238 / 30 / 0 | 15 |

MSSD is normalized by the original object diameter D; MSPD uses 640-pixel
image width. Every MSSD threshold 0.05..0.50 and MSPD threshold 5..50, plus
auxiliary ADD <= 0.1D, appears in `thresholds.csv`. Initialization, all inputs,
all tracking and the fixed BOP19 tracking subset are separate. CSV empty fields
represent undefined/null quantities, not zero risk.

The gate withheld 1,025 tracking poses and retained 6,497. At MSSD < 0.1D it
withheld 29 erroneous candidates and 996 correct candidates. The official
subset contains no erroneous candidates at this threshold, yet 29 are withheld.
This is a substantial false-rejection cost; it is not an overall accuracy gain.
The large clamp's ADD classification differs strongly from its symmetry-aware
MSSD classification. Both are retained, without choosing a favorable metric.

These are development measurements from a single algorithm arm. They do not
establish physical instance identity, automatic relocalization, baseline gains,
heldout performance or CVPR-level success. Full-video labels may have lower
quality than the separately reported official BOP19 subset. Consecutive frames
are correlated. Timing is instrumented and excludes some output serialization;
overlapping timing scopes must not be added or called deployment throughput.

`summary.json` contains counts and observed timing; `sources.json` pins the
underlying raw reviews and posthoc records. This package contains no images,
raw poses, GT poses, point tracks, weights, binaries or source code.
