# Actual native depth dropout: complete 12-frame pair

Both real neural runs completed all 12 inputs. Control published 4 and abstained on 8; candidate published 11 and abstained on 1. At frame 4, only decoded depth was zeroed; original RGB and actual TAP RGB input remained unchanged. Both arms retained the actual FP return and performed no private arithmetic or update at that frame. Candidate returned to tentative publication at frames 5–11 using current depth and the original anchors; identity verification remained false.

Of the 7 additional post-gap publications, C/W/U counts are [7, 0, 0] at MSSD <0.05D, [7, 0, 0] at MSPD640 <5 px, and [7, 0, 0] at ADD <=0.1D. All 24 actual candidates, including frame 4, remain in the metrics. Full 21-threshold results and all five scopes are preserved without selecting successful frames.

GT was opened only after both runtime output trees, commands, source bindings and independent raw checks were sealed. Metrics reuse official BOP code, all 386 original metre-scale vertices, identity-only declared symmetry and original camera calibration. No pose alignment, repair, extra C transform, visibility filtering or threshold tuning occurred. Translation is at the original model origin; MSPD640 rescales actual-width480 pixels by 640/480.

This is one synthetic development stream with actual SAM2/TAPNext/FoundationPose, one fixed missing-depth frame, and one run per arm. More publication is assessed separately from accuracy. It does not establish physical identity after ambiguous occlusion, general robustness, or deployment speed. Native adaptation remains enabled; subsequent candidate changes include the extra updates permitted after current evidence recovers. Timing boundaries and all adverse errors are retained.

Two earlier CPU raw-audit attempts stopped on auditor assumptions about the RGB cue formula and storage of an empty initialization iteration slot. The corrected reviewer passed without rerunning either GPU arm. Both failed audit receipts remain preserved and hash-bound; these were not neural execution failures.
