# Legacy recovery controller: CPU numerical evidence

The combined caller and terminal checks stop two reproduced promotion errors.
This is a CPU fixture result. The native recovery path is not connected, and
these data do not establish real-scene recovery accuracy or physical identity.

All 14 scenario arms and all 42 planned input slots are included. No rejected
observation was dropped. The four-arm counterexample uses unchanged default
8 mm point support and 15 mm equivalence thresholds: normal and recovery stub
returns lie 7.8 mm on opposite sides of the current point fit, so their final
separation is 15.6 mm. Original, terminal-only and caller-only arms each still
promote on the third frame. The combined arm rejects all three slots with
ambiguous_refined_recovery_hypotheses and does not update trusted memory.

In the source-ambiguity example, two actual supported rigid hypotheses are
produced on frame three. Original and terminal-only arms still promote;
caller-only and combined arms cancel. Ordinary missing source points without
known competing hypotheses still permit the existing isolated chain to recover.
Both positive controls also preserve three-frame promotion: a stale normal pose
with CAD support but no current point support, and equivalent current supported
normal/recovery returns. Missing arms in these controls were not run and are not
imputed as results.

The normal competitor is the final current normal output, not earlier iterations.
It must pass the original commit predicate using the same source configuration
object, an independent copy of the current pending observations, and the current
proposal's symmetry gauge. A temporary base tracker is committed; source.commit
is never called for this probe. Source point/pose memory, registration and view
bank do not change on the tested rejection paths. Propose/reject bookkeeping can
still change, so this is not a claim that every object attribute is immutable.
Recognized ambiguity/unresolved diagnostics cancel an active quarantine; ordinary
source point loss does not. New alternatives still count against the unchanged
configured hypothesis budget. Terminal checks are bounded by max_hypotheses-1
(seven under the default eight); alternative_index is an index in the supported
sublist, not the original supplied list.

The public slot fields preserve status, exact rejection reason, local verified
flag, trusted-memory revision delta, callback/return counts, normal support and
terminal alternative distances. normal_output_selected copies the source
receipt's raw_normal_output_preserved field: it means the final selected output
is the normal output. It does NOT mean that a promoted frame lost the normal
return. All normal returns remain in the in-memory returned raw_iterations
array. selected_output_matches_in_memory_raw_iterations was asserted for every
completed slot. The CPU harness did not persist full pose matrices to disk;
this is not a complete on-disk raw-output archive. Pose matrices and pose
coordinates are intentionally absent from this package.

Sixteen helper contracts passed. A separate fresh process imported the complete
three-file candidate and passed all 85 tests: 11 controller integration, 30
quarantine, 24 RGB-D seed, 17 trusted-view, and three new controller contracts.
The new tests check exception cleanup, disabled-path behavior and configured
hypothesis-budget refusal. Test names and all denominators are included.

The neural callbacks are CPU stubs. The ambiguity fixture also supplies fixed
ORB descriptors/locations; matching, rigid consensus and ambiguity production
are actual code. LK, RGB-D lifting, CAD checks, transactions and promotion run
normally. There is no GPU, GT selection, learned-model accuracy evaluation or
held-out dataset. The dependent normal/refiner estimates can agree on a wrong
instance; unenumerated or observationally indistinguishable competitors remain
outside this test. These fixes do not clear the native identity-loss latch.

Helper timing includes assertions/mocks on a 128x128 fixture, not deployment
performance: four calls took about 0.482–0.534 ms. Snapshot-array ownership was
19,400–20,168 bytes; a successful trial retained 18,272–18,752 bytes, overlapping
the snapshot gray array, so those counts must not be summed. Counts exclude
NumPy arithmetic temporaries/Python objects/process peak memory and scale with
input image size. No temporal cache is added. The 85-test overlay took 1.327 s.

Both CPU executions were first attempts and passed. Intentional rejected slots
and injected-exception contracts are retained. The previous eight low-level
scenarios were not repeated or counted as new evidence. Source files, diffs,
images, raw matrices, GT, absolute/relative filesystem paths and model weights
are excluded. Only their hashes and artifact labels are published. manifest.json
binds the other three payload files; its own digest is supplied separately to the
publisher. This package was prepared locally and has not been uploaded by its
authoring agent.

Version 2 keeps every numerical slot/count/distance unchanged and preserves the
previous package. It corrects in-memory versus disk-retention wording and adds
two audit limits. The helper-exception contract verifies source cleanup and
exception propagation only. The existing controller rethrows and does not attach
the already returned normal raw pose to that exception or save it to disk.
Persistence under this exception is unverified and is not claimed.

The four-arm CPU runner imported the previously reviewed low-level terminal
candidate, but did not list that executed file in its own pre/post source
inventory. Its hash was bound at preparation and rechecked now, and the _observe
AST matches the complete candidate source. These checks cannot retroactively
prove a missing runtime pre/post binding. The separate 85-test complete-source
overlay has its own explicit three-file source binding. No new execution was
performed to conceal or repair that earlier inventory gap.
