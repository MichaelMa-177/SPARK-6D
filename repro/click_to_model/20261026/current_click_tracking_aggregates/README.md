# Current-click product experiment, 2026-10-26

One fixed assistant-selected click on the first RGB frame initialized SAM2.1
small, FoundationPose and the measured-motion tracking path on a reused mustard
development video with an existing CAD mesh. This is not held-out evaluation.

The successful attempt completed 737 inputs: one initialization and 736 tracking
frames. All three current SAM masks and 756 actual registration candidates were
retained. Only mask 2 passed the existing observation geometry check. Full-bank
admission remained ambiguous, and all seven trust flags remained false.

The first attempt failed in an instrumentation type check after SAM returned
binary float32 masks. It has zero completed, one failed and 736 unexecuted input
slots, plus 756 unknown registration slots. The observer was corrected in a
separately frozen attempt; the original click, production algorithm and failure
evidence were retained. attempts.csv keeps both attempts.

Evaluation uses absolute ADD on all 10,983 canonical trimesh vertex rows,
including the same UV duplicates as earlier metrics. The audited official GT
matrices are unchanged: no alignment, scale fitting, rotation repair or symmetry
minimization. The threshold is 10% of the canonical vertex-pair diameter.
Initialization is separate, and all 736 tracking slots remain in the denominator.

There are 29/736 threshold failures and mean tracking ADD 9.7387 mm. All failures
are within the input intervention [100,130); the other 706 tracking frames and
607 post-intervention frames are below threshold. The first five consecutive
correct post-reappearance frames are 130–134. This does not verify instance
identity. The actual initial mask differs from supplied-mask experiments, so
no same-input improvement or causal attribution is claimed.

Independent raw-data and metric audits passed. All 737 ADD values and failure
labels, four summary groups and AUC values, and recovery indices were checked
independently. The largest ADD difference was 5.56e-17 m (floating-point
rounding). The metric review receipt SHA256 is
`4a2e421db26968469c1ebf5599f694c4a5c828865b0c55c3dad341d273b5efc1`.
Timing includes instrumentation and is not a deployment speed measurement.

The package contains aggregate numbers, per-frame errors, hashes and plots.
It contains no raw poses, raw GT matrices, RGB/depth images, model assets, source
code, logs, weight files or local absolute paths.
