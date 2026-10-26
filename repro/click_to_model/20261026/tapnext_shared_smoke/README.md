# TAPNext++ shared-model runtime check — 2026-10-26

One actual model processed two independent causal state streams. All 42 planned calls completed; all 21 interleaved outputs and recurrent-state hashes exactly matched each stream's independent rollout. Inactive stream state and registered model parameters/buffers were unchanged.

The fixed synthetic fixture has 12 RGB frames, two black frames, eight original queries and thirteen secondary queries born at frame 3. There are no real scene images, CAD models, depth, GT, pose estimates or identity labels. No query insertion or recurrence merging was used.

The complete scope is 24.666 s including strict load, source/weight checks, state hashing and evidence persistence; it is not tracking FPS. This finite runtime result does not establish occlusion recovery or pose accuracy. The attached JSON retains the precise scope, counts, memory and provenance hashes.

This public package contains two aggregate files only. It excludes raw neural outputs, recurrent tensors, images, checkpoints, source code and individual trajectories. The original evidence and independent saved-output review remain local.
