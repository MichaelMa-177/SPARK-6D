# Complete-hypothesis initialization: development results (2026-10-24)

These are aggregate measurements from eight completed runs of SPARK-6D, comparing legacy initialization with complete-hypothesis admission. Both arms use the same adaptive FoundationPose implementation (at most three tracking refinements), with the trusted-view / constant-velocity configuration available only after admission. All four complete-hypothesis initializations remained tentative; their subsequent path therefore did not maintain trusted point, view or velocity memory.

Each method processed all 8,389 inputs from four previously evaluated YCB-Video captures. Evaluation excludes the four initialization frames, retains all uncertain outputs, and uses all 8,385 tracking frames per method. There were no missing predictions. First-frame original target masks were used; this matrix is not a user-click segmentation benchmark or unseen holdout.

Both methods failed MSSD on 1,916/8,385 tracking frames (22.8503%) and 75/298 fixed BOP19 target frames (25.1678%). Complete-hypothesis admission reduced wrong poses reported as accepted from 1,914 to zero, but left all 6,469 MSSD-correct poses unconfirmed. This does not establish improved pose accuracy or useful positive confirmation. The mustard bottle remained incorrectly oriented throughout both runs. Its ADI passed despite ADD/MSSD failure.

MSSD minimizes maximum corresponding-vertex error over declared model symmetries, using every original CAD vertex and a fixed 315-yaw finite approximation for the bowl's continuous symmetry. Failure threshold is strictly greater than 0.1 times original CAD diameter. Raw model coordinates and raw GT rotations are retained; no rigid alignment, scale fitting or SVD repair is applied. Ordinary ADD and nearest-neighbor ADI are reported separately; ordinary ADD is orientation sensitive for the symmetric bowl.

The CSV contains one row per complete run. Confidence categories are relative to MSSD and sum to every tracking frame. Initialization time is separate. Tracking FPS is a single-run measurement including actual network-return copying/compression audit costs; workloads differ between methods. It is not a same-function production throughput comparison. Initialization admission for the bowl exhausted its 30-second pairwise-check budget and used the bound original top-1 as an untrusted working pose.

All eight GPU attempts reached a terminal state before this evaluation opened GT. Independent output verification checked 30,431 actual network returns and 91,253 numeric arrays. Independent metric verification recomputed aggregates from 16,770 tracking / 596 fixed-target records and recomputed ADD/ADI/MSSD at 24 predetermined first/middle/last positions using original GT and all original CAD vertices. That audit recorded three dependency versions but did not enforce them as a pre-run package guard; no claim of a hermetic environment is made.

This publication contains aggregate metrics only. Full local trajectories, per-frame poses, raw network arrays, images, models, weights, logs, machine paths and source snapshots are not included. Their absence limits reproduction from this small publication alone. Complete trajectory publication remains pending separate scope confirmation; this aggregate update does not imply that those packages have been published.

Provenance SHA256 (hashes only):

- Frozen eight-run plan: `cfae197d733c73c1ce7059f25342986d03b145dfe11d21661cf427b90dbc5334`
- Completed evaluation: `806ca01cf4e1d82f3cea2367fd74dd514b61b58f844b2b648175329679cd97b1`
- Independent metric audit receipt: `42190c358314760130deff33a2d046d67874ed0112d745e4dc7dda44ae6da7a3`
- Independent raw output audit: `221d373cf34c49994ea4c40692b4df6c644735050aba5ad76f05ccd75840b7ac`
