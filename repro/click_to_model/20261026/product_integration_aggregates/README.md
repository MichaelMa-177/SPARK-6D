# Product integration: complete 737-input mustard sequence

The opt-in `tentative-preloss` public entry completed all 737 inputs. Its final
737 original-coordinate poses match the frozen experimental run exactly in
dtype, shape and bytes; maximum difference is zero. This is a software migration
check on reused development data, not a new accuracy or generalization result.
The run used a supplied first mask. Fresh-click SAM and live-camera validation
are separate tasks.

The full CPU regression passed 1,609 tests with zero skips. The actual GPU run
made 738 predictor calls and 742 network returns. All 5,910 neural comparisons
are exact. All 22,363 pipeline prerequisite gates pass. The broader comparison
retains 138,365 exact and 1,472 differing entries: every difference is the known
centered-pose storage shape, old `float32[4,4]` versus new raw `float32[1,4,4]`.
The explicitly extracted single-pose values match; this package does not relabel
the complete archive as byte-identical.

The first attempt failed in instrumentation before reading an input. The second
completed registration and failed at its first tracking callback because the
adapter rejected FoundationPose's legitimate singleton batch. Both attempts and
their entire 737-input denominators remain in `attempts.csv`. The correction
preserves the raw batch and selects its sole matrix without changing pose values;
other shapes remain rejected. The third attempt completed without failed,
unknown or unexecuted inputs.

`final_pose_comparison.csv` contains 737 numerical verification rows and hashes,
not pose matrices. Actual outputs and derived coordinate transforms are separate:
121 derived-versus-actual records differ by floating-point rounding (maximum
5.960464477539063e-08), while actual-versus-actual outputs match on every frame.
The old initial derived value was not recorded and is left unavailable.

All identity/orientation verification and trusted-memory flags remain false.
Texture/symmetry validity coverage and same-appearance instance identity remain
open requirements. Instrumented times include evidence serialization and must
not be presented as deployment speed. The original frozen accuracy results are
documented separately in `../preloss_motion_aggregates/`.

This package contains summaries, numerical comparison rows and provenance
hashes only. It contains no images, ground-truth poses, raw trajectories, model
weights, local absolute paths or source code.
