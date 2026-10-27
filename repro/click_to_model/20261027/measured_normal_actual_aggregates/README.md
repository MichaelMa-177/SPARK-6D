# Actual measured-normal control: rank changes, no accuracy gain, higher cost

Both actual arms completed the same two synthetic streams: **360 inputs and 358 tracking frames per arm**. The ambiguous stream has two alternative persistent-ID worlds, evaluated using the same one output per arm. They are not independent videos or additional samples.

Direct dtype/shape/content comparison confirms **all 360 paired actual saved poses are byte-identical**. All 358 per-frame neural iteration counts also match. Every one of the 126 fixed target-metric paired tables is diagonal. **There is no pose or accuracy gain.** No offline normal replay or substituted trajectory is used.

The observed-depth normal diagnostic changes conditional local full-rank counts from **155 to 98 /179** on `continuous_visible`, and **147 to 131 /179** on `hidden_identity_ambiguous`. These are actual final-iteration diagnostics, not proof of physical identity or an accurate pose. All seven trust flags remain false; unverified is not successful abstention.

Instrumented cost worsens. Continuous initialization increases **16.102 → 36.778 s**, tracking **15.504 → 18.167 s**, and measured tracking FPS decreases **11.546 → 9.853**. Ambiguous initialization increases **12.786 → 35.216 s**, tracking **13.573 → 15.682 s**, and measured tracking FPS decreases **13.188 → 11.414**. Registration/admission are inside initialization; parent, child and stage scopes overlap. Do not add them or call these measurements deployment FPS.

Unchanged identity failure is retained: in the ambiguous exchange world, all 70 post-opening tracking frames follow the distractor. The restricted adjacent-switch count remains zero because neither-associated frames interrupt the counting chain. Zero does not mean preserved identity.

Files contain 22 association migration tables, 126 fixed metric migration tables, 28 rank phase rows, four actual timing rows, and source hashes. Initialization and all 179 tracking frames are separate; hidden and unknown slots are retained. Two-world counts must not be pooled as independent observations.

The first posthoc attempt failed only while locating the already-bound child timing receipt. Its complete numerical prefix and failure were retained. The narrow v2 lookup fix uses the exact path in raw `files_read`; all three recomputed per-frame files match the retained prefix byte-for-byte. No GPU run was repeated for that repair.

This package covers the full-image measured-normal implementation only. Sparse v3 is not yet included; a kernel speed ratio cannot be presented as whole-system FPS. The bounded aggregate check used separately written formulas over saved errors and actual diagnostics, with no new GT or geometry evaluation. It is author-side recomputation, not external replication. No raw GT matrices, poses, point trajectories, images or local machine paths are published.
