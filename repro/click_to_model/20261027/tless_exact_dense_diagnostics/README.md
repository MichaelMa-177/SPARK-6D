# Complete T-LESS dense geometry diagnostic

All 7,560 available saved pose hypotheses now have dense surface diagnostics. All 1,512 originally unavailable hypotheses remain unknown, preserving 12 trials and 9,072 planned rows. The earlier failed raster experiment remains preserved separately; it is not relabeled as successful.

The new ray backend uses every mesh triangle, the original integer-pixel camera calibration and the saved float32 camera-space geometry. The first four batched camera attributes and poses match the previous actual GPU capture byte for byte. No GT, neural inference, new pose selection, new acceptance threshold or production change is involved.

`candidate_diagnostics.csv.gz` (gzip-compressed CSV) contains every planned candidate, including unavailable rows. It contains numeric counts, scores and diagnostics, without poses, geometry arrays, images, weights or local paths. `summary.json` provides per-trial/per-mask distributions and source hashes. Count partitions and CSV/prefix agreement were checked across all computed rows. Unknown depth and foreground occlusion remain separate from free-space contradiction. Mask and normal statistics are diagnostics; the click is not a hard gate.

These results establish completion of the geometric diagnostic on the development pilot. They do not establish improved pose accuracy, physical instance identity or generalization. Candidate rows are correlated, and all verified flags remain false. Runtime measurements include instrumentation and are not application FPS.
