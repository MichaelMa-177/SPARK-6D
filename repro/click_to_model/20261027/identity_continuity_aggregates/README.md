# Causal continuity with explicit output abstention

This package evaluates a fixed research output policy on two previously executed
synthetic RGB-D streams. It preserves every original candidate and every future
tracking update, publishing either that exact candidate or no pose. It is an
offline causal replay, not a new GPU tracking experiment or solved identity.

The rule was fixed before replay: require a surviving locally observable query
chain, sufficient current support and candidate/anchor agreement. Temporary
insufficiency abstains; complete chain loss latches until a new click. Existing
RGB/depth scales are reused without tuning from these labels. Texture is not a
global requirement. API fixes preserved all old counterexamples and did not
change any of the 360 replay decisions. All seven trust flags remain false.

Tracking-only results (179 frames per observed stream):

| Stream / possible hidden world | Published | Coverage | Original ADD errors | Published ADD errors | Correct rejected | Wrong rejected |
|---|---:|---:|---:|---:|---:|---:|
| continuous_visible / stay | 161 | 89.94% | 2 | 0 | 16 | 2 |
| hidden_identity_ambiguous / stay | 54 | 30.17% | 47 | 0 | 78 | 47 |
| hidden_identity_ambiguous / exchange | 54 | 30.17% | 124 | 0 | 1 | 124 |

ADD uses <= 0.1 diameter (0.01649631473996541 m). The published ADD error risk
is zero in these development data, at substantial coverage and false-rejection
cost. Abstention is not a correct pose. The stay/exchange rows share exactly the
same hidden-stream inputs and output; they are alternate possible identity
worlds, not independent observations. There are 360 unique input frames and
358 tracking frames, with 540/537 evaluated-label rows across the three worlds.
Both initializations remain published and are reported separately.

The continuous-visible stream temporarily abstains on frames 37–54 and resumes
at 55 via surviving cues. The other stream latches loss at 55 and never silently
reacquires identity. These times arose from the fixed evidence rule, not stage
labels. No actually bridged secondary branch occurred; that path has synthetic
boundary tests only. Identical textured false matches and flat textureless
objects remain limitations. A local normal rank alone is not an identity gate.

`threshold_counts.csv` includes all 21 existing thresholds, all-input/tracking/
initialization scopes and candidate/gate variants (378 rows). ADD is inclusive;
MSSD < 0.05...0.50 diameter and MSPD < 5...50 normalized 640-width pixels are
strict. `per_world.csv` includes ADD, MSSD 0.1 diameter and MSPD 5 pixels for all
scopes/variants. MSSD 0.1D and MSPD 5px also have zero published tracking errors
here, but reject respectively 11/7 and 7/11 correct/wrong candidates in the
continuous-visible stream. Full values, unknown counts and risk denominators
are retained. Risk is undefined (null/empty CSV field) if none are metric-known.

The original host author's separate scalar/set recomputation checked all 540
numeric rows, 378 threshold rows, associations and recovery counts. This is not
an independent-author metric audit. A separate agent checked seven synthetic
gate protocol cases. Source hashes identify the fixed rule, preserved replay,
original evaluated numeric labels and these checks. No raw GT, poses, images,
masks, models, observations, logs, executable source or local paths are included.

Known development data only: no held-out claim, no causal model-training benefit,
no deployment recommendation, no physical identity certification. Neural
adaptation combined with this gate is a separate pending comparison.
