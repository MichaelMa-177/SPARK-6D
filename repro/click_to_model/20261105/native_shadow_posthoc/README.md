# Three-frame shadow pose accuracy (posthoc)

All six unpublished candidates pass the inherited thresholds. Mean ADD is nearly unchanged and slightly worse: main 2.007275 mm, shadow 2.007405 mm. Frame 6 improves and frame 7 worsens; frame 5 is identical. This does not establish identity recovery or general accuracy improvement.

The frozen 4-second shadow trial supplies frames 5–7 of one synthetic development stream. GT was opened only after runtime outputs were sealed; runtime used no GT. Both arms remain unpublished and identity-unverified. The earlier 1-second attempt timed out and remains preserved in the accompanying trial records.

Scoring reuses official BOP functions, all 386 original metre-scale vertices, unchanged camera calibration, identity-only declared symmetry, and all 21 prior thresholds. Saved original-coordinate returns are scored directly: no GT alignment, second centering transform, pose repair, selection, or visibility filter. Translation measures the original model origin. MSPD640 rescales actual-width projection error by 640/480.

The JSON preserves all three frames per arm, adverse changes, 42 threshold cells, definitions and source hashes. These adjacent frames are not independent trials. A synthetic gate-drop is not natural occlusion, and geometric agreement cannot establish physical identity.
