# Per-channel saturation support ablation — 2026-10-24

This package contains aggregate experimental counts only. It compares the original
whole-patch saturation exclusion with exclusion of individual saturated RGB channels.
The four first-frame captures were already used for development. This is a development
counterfactual, not an independent validation result or a tracking experiment.

Eight requests retain all 2,016 candidate-policy slots and 253,008 unordered pair
slots per appearance channel and correctness metric. Each request has 252 candidates
and 31,626 pairs. Informative pairs contain one correct and one incorrect candidate;
unknowns, fold disagreements, ties and invalid evidence remain in that denominator.
Pairs share candidates and do not constitute independent samples. ADD and MSSD use
the previously frozen labels and threshold (0.1 times the original model diameter).

`pair_counts.csv` contains all eight requests × two appearance channels × two label
metrics × old/new variants (64 rows). `informative_transitions.csv` reports all
nonzero informative outcome transitions; omitted transitions have zero count. The
affine arrays and jointly qualified classifications are unchanged controls. A strict
same-sign comparison in both observed finite spatial folds indicates a direction;
it is not a calibrated probability or a globally selected pose. Nothing here is
used for identity admission or trusted memory.

Only the gradient channel's observed saturation support changes. Original candidate
poses/order, geometry, rendered/observed gradients, contrast floor, minimum support,
spatial folds, shared candidate-pair channel support and affine fits are unchanged.
All eight requests completed and exactly reproduced the original comparisons before
new comparisons were qualified. The posthoc plan was frozen before replay execution.
The independent raw audit precedes evaluation; the independent metric audit checks
the actual saved classifications, transitions and full denominators.

There are no raw images, CAD models, weights, source code, trajectories, individual
pose matrices, per-pixel data or per-pair observations in this public package.

For the gradient channel under MSSD labels, the informative denominator is 80,218. Correct wins change from 1,696 to 5,604, wrong wins from 994 to 6,094, and unknowns from 75,668 to 58,429. More support does not establish better discrimination. This result does not justify enabling the replay as a pose-admission rule.

Numerical limitation: independently regrouping Sobel sums crosses the fixed 2/255 contrast boundary for two observed bowl patches. A diagnostic that changes only these support masks changes 37 / 91 pair direction categories in the two bowl renderings. The original whole-patch comparisons are unaffected. The failed first audit and full counterfactual are preserved locally. The passing audit independently accumulates all nine weighted terms in the specified order and reproduces every actual gradient, gate and pair exactly; it also retains the alternative-order sensitivity. Production thresholds, actual outputs and these aggregate counts were not changed to pass the audit.

## Provenance (SHA256)

- `20261024_channel_appearance_v1/plan.json`: `6f2800d7d20f72feeeb5422ad5e9090ccb81a9738615a6a6587913ec715e3b71`
- `20261024_channel_appearance_v1/execution.json`: `ed58f2c9e9d8b9c821cfde20959771b0753b19f250992efac490e76a854580c7`
- `20261024_channel_appearance_posthoc_v1/plan.json`: `df2e1781b1a75198000a4bc784111a5731c62bcfacc1c9bffd603c20ffe6ea30`
- `20261024_channel_appearance_posthoc_v1/evaluation_v1/comparison.json`: `fed84270dd5370f069271352af65137ff71d0200fb424a4deb35c6f25ad0ba23`
- `20261024_channel_appearance_v1/independent_stress_audit_v2/receipt.json`: `0df80347395d16e0af915a9644751ca4b8cfb099bc76a1138c78f5f175d276a0`
- `20261024_channel_appearance_posthoc_v1/independent_symmetry_metric_audit_v1/receipt.json`: `e65b13285f0ab7ee6f49d34ca28a7fbbf2384ffd80d535f5cb9bdba377d1ce55`
- `20261024_channel_replay_stress_roundoff_diagnostic_v1/counterfactual_report.json`: `e770d22c13a3bfa4d70382232b23b6651363d232a62ae33c73b2c945cddd0bfa`
