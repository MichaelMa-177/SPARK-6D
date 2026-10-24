# Tracking and candidate appearance development experiments — 2026-10-24

These are measured development results, including failures. They do not establish
physical instance identity, reliable relocalization, generalization, or CVPR-level
performance. Only aggregate numeric tables and source-result hashes are included.

* `point_motion.csv`: six complete live 2D point/depth motion streams. Prediction
  is D(t,0) @ raw GT(0), evaluated with all original CAD vertices. This removes
  initialization error and is not absolute pose accuracy. Each backend has 1,294
  tracking frames across all three cases.
* `closed_loop.csv`: nine fresh registration and actual adaptive neural refinement
  streams. The primary metric is absolute ADD of the actual original-coordinate
  return. The no-point control, BootsTAPIR and TAPNext++ arms each retain 1,294
  tracking frames. Initial candidates, scores, transforms and selected indices
  are exactly paired within each case. The neural relative column separately
  evaluates P(t) @ inverse(P(0)) @ raw GT(0), without changing raw annotations.
* `candidate_appearance.csv`: eight fixed-pose requests, each with all 252
  candidates and 31,626 unordered pairs, under albedo and fixed illumination.
  Rows repeat the same pair set for two evidence channels and two GT labels;
  they are not additional independent runs. The informative denominator includes
  every pair with one correct and one incorrect label. Unknown, disagreement,
  tie and invalid outcomes stay in that denominator. No candidate was admitted.

Tracking failure is ADD > 0.1 original CAD diameter. Missing or invalid outputs
are infinite errors, never dropped. Empty mean fields mean the full trajectory
mean is undefined due to nonfinite failures; they are not finite-only means.
Frame zero is separate from tracking. Recovery is five consecutive correct
selected inputs after a fixed intervention, evaluated after inference; it is not
an online identity or correctness decision. Empty recovery delay means no such
window exists (or no intervention was defined).

All inference matrices finished before their GT evaluation. Raw output lineage
was independently audited. Candidate-appearance counts were independently
recomputed. These captures had already been used for development. The appearance
unknown results partly reflect the original whole-patch saturation exclusion;
they do not prove absence of texture information. This table retains that rule.

The CSV FPS boundary is completed tracking-slot wall time: decode, actual model
calls, depth fitting/refinement as applicable, and per-frame evidence writes.
Initialization, loading and whole-run verification are separate. The nine-stream
run shared the host with six TUD-L download workers and CPU review work. These
are artifact-producing development timings, not isolated production throughput.

Source result SHA256 values (paths identify local experiment records):

* `results/click_to_model/20261024_tentative_point_motion_posthoc_v1/evaluation_v1/comparison.json`: `284b2cb8a19bbc99e7889d33d0fd348e4b949b93b7e7ef7b2b4b5cbc4b748cc6`
* `results/click_to_model/20261024_initial_candidate_appearance_posthoc_v1/evaluation_v1/comparison.json`: `dc510db229beb11220c3fbb5767386e52d210524a7fd705308a60f06fb43a875`
* `results/click_to_model/20261024_relative_motion_refinement_posthoc_v1/evaluation_v1/comparison.json`: `828ca051366ec2b6af95ebedb915bbc344ffa9fae3b4b23fe37722cf4a84dc16`
* `results/click_to_model/20261024_relative_motion_refinement_v1/runtime_statistics_stress_v1/report.json`: `b9a0536622e96e6d865c2def2b2a85adb93a89e9c04dde1ee5252ddfd6d25403`
