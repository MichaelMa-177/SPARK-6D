# Neural adaptation × fixed causal output gate

Four combinations are evaluated on one previously recorded synthetic
continuous_visible/stay stream: baseline candidate, baseline + gate, neural
adaptation candidate, and neural adaptation + the SAME frozen gate. There are
180 observed frames (179 tracking) shared by two actual inference traces. The
360 per-arm rows are not independent observations. No hidden-world adaptation
run is added. Both initializations are retained and separately counted.

The gate source and every threshold are unchanged. It consumes each trace's
own current RGB-D, point evidence and candidate; evaluation uses that trace's
own already computed errors. No GT is supplied to the gate. It only changes
whether a candidate is published; future tracking and adaptation are unchanged.
This composition is a CPU replay, not a new GPU or closed-loop experiment.

Tracking-only results:

| Combination | Published /179 | Coverage | ADD errors among published | Correct rejected | Wrong rejected | Mean ADD on published (mm) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline candidate | 179 | 100% | 2 | 0 | 0 | 2.662753 |
| Baseline + gate | 161 | 89.94% | 0 | 16 | 2 | 2.025738 |
| Adaptation candidate | 179 | 100% | 2 | 0 | 0 | 2.677744 |
| Adaptation + gate | 161 | 89.94% | 0 | 16 | 2 | 2.021115 |

The publication mask is identical across all 180 frames of the two arms:
temporary abstention on frames 37–54, recovery at 55, no latched identity loss.
These transitions come from the previously frozen causal evidence rule, not
error labels. All seven trust flags remain false. Abstention is never counted
as correct, and zero published error risk comes with 16 rejected correct ADD
candidates in each arm. Published risk is wrong / metric-known published;
with no metric-known output it is undefined (null/empty), not zero.

Adaptation changes 177 of 180 candidate pose arrays. The largest candidate
translation difference is 1.504379 mm. The largest raw matrix-element
difference is 0.01691073179244995 (mixed dimensionless rotation entries/metre
translation entries, not a rotation angle or pure distance). The original
candidate arrays are never replaced by this package or gate.

Over all 179 tracking candidates, mean ADD worsens by 0.014990 mm despite
150 frame pairs improving, 27 worsening and 2 equalling. On the same 161
published frames, mean ADD decreases by 0.004623 mm. The latter conditional
change does not establish an overall accuracy improvement. Mean MSSD also
worsens slightly over all candidates; normalized MSPD mean decreases slightly.
All per-frame numeric error differences and per-combination mean/median/max
statistics are retained, so neither improvements nor regressions are hidden.

ADD uses <=0.1 diameter (0.01649631473996541 m). MSSD uses strict <0.05...0.50
diameter and MSPD strict <5...50 normalized 640-width pixels. All 21 existing
thresholds, three scopes (all inputs/tracking/initialization) and four
combinations appear in `threshold_counts.csv` (252 rows). At MSSD 0.1D each
candidate arm has 7 errors, and each gate rejects 11 correct + 7 wrong; at
MSPD 5px each has 11 errors, with 7 correct + 11 wrong rejected. Both gated
arms have zero published errors at these thresholds as well. The complete
tables, unknown counts and risk denominators are included.

Files: `four_combinations.csv` holds 36 headline rows; `error_statistics.csv`
has 60 scope/combination/metric summaries; `candidate_changes.csv` contains
180 numeric error/pose-change summaries, not pose matrices. The original
adaptation parent and child statuses remain failed, parent return code 1.
Qualification uses the separately sealed complete180 raw recovery and exact
classification of four denied bytecode-cache probes. This does not waive
other failures or rewrite the original execution history.

The original host author independently implemented scalar/set count checks:
all 252 count rows, 360 numeric rows, associations/recovery and exact published
pose-copy checks passed. This is not an independent-author audit. Provenance
hashes bind the actual raw, metric, frozen-policy and numeric-review sources.
No raw GT, poses, images, masks, models, observations, logs, executable source
or absolute local paths are exported.

Known synthetic development data only. No held-out, deployment, identity-solved
or causal adaptation-benefit claim. Smaller selected-subset errors cannot
substitute for all-candidate accuracy or conceal the coverage cost.
