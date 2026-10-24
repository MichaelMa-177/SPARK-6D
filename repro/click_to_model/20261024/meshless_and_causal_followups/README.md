# Generated-mesh full sequence and causal-query negative results

This independent numerical increment contains three completed experiments:
a new full 375-frame SAM2/SAM3D generated-mesh run, a four-cohort learned-query
replay, and a frozen 256-to-480-resolution comparison. It does not establish
CVPR-level performance, physical-instance identity or general recovery.

Nine exported array tables contain 1,845,886 numerical values. All values were
independently reconstructed from the written CSV fragments and checked against
their source numerical arrays; non-NaN floating-point bits match. Schema files
record array shape, dtype and CSV layout. Fragment headers repeat and fragments
must be read in order. String arrays, images, meshes, weights, source code,
execution logs, local paths and provenance digests are excluded.

## Full generated-mesh run

SAM2 segmentation, SAM3D mesh generation and adaptive tracking were rerun from
the first RGB-D input. The previous prefix mesh was not reused. Runtime did not
receive an original CAD model, supplied mask or GT. The development click
(167,392) came from an earlier mask-derived location. Automatic whole-object
mask selection remains ambiguous; this is not an independent human-click test.

The complete cracker-box sequence has 375 inputs and 374 tracking inputs.
Posthoc evaluation fixes one gauge from input zero:
`C = inverse(P_generated_0) * GT_original_0`, followed by `P_generated_t * C`.
All original CAD vertices are used to measure motion ADD. There is no scale fit,
trajectory or per-frame alignment, and raw GT rotations are not repaired.
The initial error is cancelled by construction. This measures relative motion,
not absolute pose, initial registration, generated shape quality or identity.

Motion ADD mean is 8.008 mm, median 7.379 mm, P95 12.827 mm and maximum
18.140 mm. All 374 predictions are retained, with zero missing and zero failures
above 0.1 original-CAD diameter. The controller itself reports 208 tracking and
166 uncertain-or-lost inputs. Thus zero motion-ADD failures does not mean every
pose was committed as trusted memory. Tracking throughput is 43.09 FPS under
the recorded runner boundary. Raw neural iterations, predicted and raw-GT poses,
the fixed gauge, verified flags, source indices and latency values are included.

## Multi-cohort replay

The mustard pressure stream has 737 inputs, with an existing synthetic
foreground-plane intervention on inputs 100 through 129. Learned query state
consumes each RGB input once, including those inputs. Queries are born at inputs
0, 30, 60 and 90 from independently recorded runtime poses at that same time.
Only reviewer poses from inputs 0 through 99 are accessed. These are frozen
external reviewers, so this is a controlled replay rather than the final online
controller. GT is used only later for independent scoring and diagnosis.

There are 123 query IDs and 105 become active after independent multi-frame
promotion. Every candidate pose is deliberately rejected by the controller to
test query-memory isolation. Fixed query descriptors and initial coordinates
remain unchanged. Candidate throughput is 13.84 FPS, excluding decoding,
hashing and model loading. Separate cohorts currently encode each input again.

The primary denominator retains all 736 GT-defined tracking inputs. Of these,
637 fail: 631 have no candidate and six have an incorrect finite candidate.
After reappearance all 607 inputs fail: 601 missing and six incorrect. The
six incorrect candidates occur at 133, 135, 136, 604, 614 and 615, with original
CAD ADD of 60.23, 56.02, 54.09, 102.54, 98.31 and 97.26 mm. None constitutes
recovery. Raw original-CAD ADD is coordinate-bound by the frozen first runtime
registration; optional first-frame motion diagnostics are labelled separately.

The original replay omitted a separately serialized CAD anchor table. The
included anchor table was reconstructed later from the frozen birth inputs and
reviewer poses. Every birth/promotion event and all 737 selected-correspondence
lists reproduced exactly. This limitation is retained explicitly.

The query observation table contains all 84,831 observations of query IDs that
had actually been born. Unused capacity columns and pre-birth placeholders are
omitted, with birth times and immutable IDs retained in the anchor table.
No meaningful query observation or failed frame is dropped. Query positions,
probabilities, active flags, depth validity, fixed-target pixel/3D residuals and
visibility proxies remain available. Raw pose predictions are in the per-frame
metric table. Object-frame anchors are sparse query points, not a mesh export.

After occlusion, the four cohorts have median valid-depth visible-query errors
of 28.8, 22.5, 31.6 and 20.5 mm, and median pixel errors of 11.9, 10.3, 15.1 and
9.9. Only 233/4627, 334/3993, 131/6271 and 534/8448 observations meet both
3 pixels and 8 mm. This is a descriptive geometric correspondence proxy,
not independent point-identity annotation. GT-projected surfaces often still
have measured depth support, so persistent object absence alone does not
explain the failure. The surface proxy combines front-facing sampled normals
with depth agreement at raw-GT projection; it is not a rendered visibility GT.

Noise statistics use only pre-occlusion trusted runtime poses, without GT:
inputs 1–69 for calibration and 70–99 for a holdout. MAD scales in u, v and z
are 1.15 pixels, 1.23 pixels and 3.12 mm. This is exploratory calibration on an
already examined sequence, not a retrospectively preregistered result.
The per-observation residuals are included so other calibration choices remain
auditable. Visibility probabilities are not metric uncertainty estimates.

## Frozen 480-resolution comparison

Before inference, the single-cohort comparison fixed the same complete mustard
stream, checkpoint, first mask, first runtime registration and all 30 native
query coordinates and 3D anchors used by the earlier 256-resolution probe.
Elementwise assertions verify equality of queries, IDs, anchors and initial pose.
Only the learned backend input size changes to 480 by 480. The rigid gate remains
8 mm, minimum eight points and 60% consensus. No future image or runtime GT is
used, and all candidates remain deliberately rejected by the controller.

Of 736 tracking inputs, 637 fail, all because candidates are missing. The 99
correct candidates occur before occlusion. All 607 post-occlusion inputs lack
a candidate. Increasing resolution recovers two pre-occlusion candidates
relative to the 256 probe, but does not solve this recovery failure. Throughput
falls to 22.04 FPS. This is an existing-sequence resolution ablation, not a full
Point2Pose reproduction. Point2Pose already combines causal learned queries,
depth, multiple rigid hypotheses and map validation; those components remain
an existing baseline, not a new contribution claimed here.

Primary denominators include hidden, unavailable and invalid candidate inputs.
No finite-only mean replaces an all-frame score with missing predictions.
Five consecutive correct candidates would be required to report candidate
reappearance; neither new mustard experiment has such a sequence. Candidate
availability, learned visibility, and endpoint ADD success are not equivalent to
verified online recovery or correct physical-instance identity.
