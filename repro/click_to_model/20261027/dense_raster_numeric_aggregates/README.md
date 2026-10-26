# Dense mesh raster diagnostics, 2026-10-27

These are numerical aggregates from four fixed real-mesh hypotheses, not a tracking benchmark or a physical ground-truth evaluation. No raw poses, vertices, pixels, images, weights, logs, source code or local paths are included.

The earlier full-candidate diagnostic failed in its first batch: all 9,072 planned rows remain unknown, including 1,512 unavailable rows. The four-candidate reproduction is byte-identical to that batch. All 675 ray-consistency violations lie outside the ideal selected triangle. Subpixel coverage rounding followed by clamped barycentric interpolation explains the mechanism. The 18.46 mm unclamped-oracle extreme is outside-triangle extrapolation and is not a valid finite-surface error.

A separate full-mesh CPU ray reference finds five original-only pixels, six reference-only pixels, and 773 different face selections among overlapping pixels. The largest overlapping depth difference is 8.898 mm. Two contained selected faces are still behind the nearest surface, so merely fixing XY or checking containment is insufficient.

A new float64 whole-mesh ray raster prototype matches the reference coverage and face IDs on all four candidates, on CPU and actual CUDA. Version 2 also fixes an integer-overflow counterexample found by independent review. Eight author analytic groups and nine independent boundary groups pass. Near-parallel rays retain uncertainty; near/far-plane crossings and resource overflows fail explicitly. The original 2e-6 metre guard was not relaxed, and production inference is unchanged.

`per_candidate.csv` reports every candidate, including cold-call timing. Timing is instrumented prototype execution, not application FPS. `summary.json` contains source hashes, negative findings and scope limits. This single-mesh test does not establish pose accuracy, identity preservation, general renderer correctness or CVPR-level validation.
