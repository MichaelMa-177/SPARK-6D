# TUD-L working-pose risk and coverage: development exploration

**The new score-margin gate showed no additional error-removal benefit over the existing top-1 geometry check on this sample.** Increasing its threshold only removed correct outputs. No deployment threshold was selected, and the uncalibrated default continues to return unknown.

The analysis used the same saved FoundationPose banks for 96 static assistant-click observations: 3 CAD objects × 8 recorded lighting labels × 4 observations. These are GT-seen development observations, not independent validation or 24 videos. Lighting and viewpoint are not separately controlled.

| Arm / operating point | Outputs / all inputs | Correct | Wrong | Empirical risk among accepted known labels | Useful correct coverage |
|---|---:|---:|---:|---:|---:|
| Actual FP top-1, all outputs | 96/96 | 95 | 1 | 1.0417% | 98.9583% |
| Original admission | 0/96 | 0 | 0 | Undefined | 0% |
| Existing top-1 geometry qualification only | 95/96 | 95 | 0 | 0% | 98.9583% |
| New gate default, threshold=None | 0/96 | 0 | 0 | Undefined | 0% |
| Exploratory threshold 0 or 0.02 | 95/96 | 95 | 0 | 0% | 98.9583% |
| Exploratory threshold 0.10 | 82/96 | 82 | 0 | 0% | 85.4167% |
| Exploratory threshold 0.20 | 48/96 | 48 | 0 | 0% | 50% |
| Exploratory threshold 0.50 | 7/96 | 7 | 0 | 0% | 7.2917% |
| Exploratory threshold ≥0.64 | 0/96 | 0 | 0 | Undefined | 0% |

These are working-pose output counts, not trusted production state changes. The original admission remains unchanged. Every identity, global-uniqueness, risk-calibration, memory-update and deployment-authorization flag remains false. Zero empirical error on an already inspected development set is not a calibrated risk guarantee.

The sole wrong top-1 is observation 27 (CAD 1, light label 7), also the sole top-1 failing existing geometry qualification. All other 95 were correct under both frozen labels. This is a posthoc development finding, not a prospective generalization claim.

## What the prototype measured

It retained the actual complete 252-candidate bank and selected backend index for every input. Existing masked/unmasked geometry support was reused without threshold changes. The direct top-1 numerical neighborhood uses declared symmetry alignment and the existing 0.015 m / 0.15 rad tolerances; it is not a transitive equivalence relation. The new scalar is:

`q = (top-1 score − highest score outside its neighborhood) / (bank maximum score − bank minimum score)`.

This is a finite-bank ranking feature, not a probability. Outside-neighborhood top-score ties, flat/nonfinite ranges, missing evidence and no outside candidate produce unknown. Same-neighborhood ties may retain a diagnostic without proving uniqueness. The fixed exploratory scan contains exactly 51 thresholds, 0, 0.02, …, 1, applied with `q ≥ threshold`. The default `None` is a separate uncalibrated mode; no scan value was selected for deployment. All 96 actual diagnostics had finite informative q values; geometry eligibility still excluded one input.

ADD and identity-only MSSD use the previously audited complete-model-vertex labels at error ≤0.10 × official diameter. Their counts agree here; the plot displays ADD once. This single-threshold analysis is not official BOP AR. ADI was not used. No GT metric was recomputed by the prototype, no new model was called, and no production/GPU entry was modified.

## File definitions and full denominators

- `per_observation.csv`: 96 rows with q, two correctness labels, existing geometry/texture/symmetry status, four-arm output flags, scan acceptance count, and source-bank SHA. It has no pose arrays or original GT records. The 51-character `exploratory_accept_bits_tau_0_to_1_step_0p02` is ordered by the fixed grid; `1` means that exploratory point would output the working pose. Preserve this column as **text** when reading CSV so leading zeros are retained. Its sum is the row's `exploratory_accept_count`. The default output column remains false regardless of the scan.
- `risk_coverage.csv`: all 4,290 rows. There are 39 overlapping groups: all inputs, 3 CAD groups, 8 light groups, 24 CAD/light cells, 2 texture-status groups and 1 declared-symmetry group. Each has the three baseline operating points plus the new gate's default and 51 scan points, for both metrics. Baselines have no invented threshold curves.
- `risk_coverage_development.png`, `.svg`, `.pdf`: three formats of the same figure. Empty acceptance has undefined risk and is shown only as text in the risk panel, not plotted at zero risk. The right panel's geometry-only line is a coverage reference, not a threshold scan for that baseline.
- `source_hashes.json`: source, protocol, result and review hashes; `manifest.json`: exact payload hashes and sizes.

For each group, N includes every planned input. A is accepted/output count; TP and FP are accepted known-correct/known-wrong counts, U is accepted unknown-label count. We report A/N, TP/N, FP/N, FP/(TP+FP), and the conservative unknown-as-error fraction (FP+U)/A with the relevant denominators. Undefined risks are blank CSV cells, corresponding to JSON null. All labels were known in this run. No input was removed: 96/96 analysis slots completed, with zero failed or unexecuted slots.

Candidate pairs, repeated channel/rendering scores, observations from related scenes, and overlapping group rows are not independent samples. No confidence intervals, significance tests, binomial risk certificate or interpolated AURC are provided. Historical TUD-L appearance/replay requests and their interrupted unknown pairs were not reused as sample counts.

## Scope and limits

The existing texture heuristic labelled 95 top-1 observations textured and one unknown. All runtime symmetry declarations were none; this does not prove physical asymmetry. Thus this run supplies no actual textureless, nonidentity-discrete or continuous-symmetry performance evidence. Finite discrete symmetry behavior has synthetic tests; continuous symmetry is unsupported in this CPU prototype and would retain an unknown slot.

The 28 author CPU boundary tests passed with no skips. Root independently reproduced and checked the repaired outside-neighborhood score-tie case before the protocol was frozen. Independent numerical and export review receipts are bound in `source_hashes.json`. The once-only development analysis preserved all source/input hashes and took 2.105897740 seconds of recorded CPU analysis time; this is not neural inference latency or deployment FPS. No T-LESS input was read.

This package contains numerical summaries, generated figures and provenance only. It contains no raw predicted/GT pose arrays, source RGB/depth images, models, weights, source code or absolute local filesystem paths. The new gate was not introduced into production.
