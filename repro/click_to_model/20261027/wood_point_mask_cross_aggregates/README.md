Wood point-mask crossover: one development sequence, three actual runs.

At strict MSPD < 5 px (640 px normalization), all-tracking counts are legacy 360/1828, SAM-score 283/1828, crossover 303/1828. This is partial recovery, not full restoration. Fixed BOP75 is a separate subset and initialization is a separate row. The crossover keeps SAM-score current registration/working pose and changes the initial GFTT mask plus original depth-anchor foreground mask to the same current registration legacy choice. It does not install old queries or poses.

The 126 rows contain all nine paired transitions, including unknowns, at every fixed threshold for both previous actual arms. Core metrics include negative and positive changes, plus ADD as a secondary symmetry-representation-sensitive diagnostic. All outputs remain tentative; accepted-risk fields are null. Frames are correlated observations from one development sequence; there are no independent-frame confidence intervals or population claims.

Timing is instrumented. Initialization, tracking, child and parent clocks overlap and must not be summed. Model loading, synchronization and audit I/O are included according to their recorded boundaries. The legacy run completed its child after the parent interruption; original parent elapsed time and parent FPS stay unknown. No deployment-speed conclusion follows.

Independent review scope: clocks, 126 paired tables and 378 summary threshold rows recomputed directly from frozen per-frame errors without calling producer paired/count helpers. No GT geometry was recomputed. Source hashes identify the existing raw and metric evidence; this package contains only aggregate counts/errors/times and hashes, with no raw pose, GT matrix, trajectory, image, model, local path or weight.
