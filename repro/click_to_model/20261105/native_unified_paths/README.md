# Unified existing-mesh and generated-mesh native tracking

Both known development prefixes completed 30 input frames and 29 tracking frames, with 29 private updates and 30 tentative publications each. Existing mesh used 756 initialization candidates; generated mesh used 252.

Existing mesh matched the prior native entry on 13 visible numerical files and four delivered masks: 383 numerical comparisons and nine structural comparisons had zero differences. Generated mesh passed actual mesh/mask consumption and the original metric ICP acceptance checks. These are functional and saved-output checks, not proof of generated shape, absolute orientation, identity or 6D accuracy. No GT was read.

`summary.json` contains counts and limits; `timing.json` separates parent preflight/child/postflight from nested pipeline and tracking timings; `sources.json` contains only source/evidence digests and byte counts. Missing CAD stage wall time is null. Timing is one run per case and is not a matched speed comparison.

The first CAD CPU verifier failed because it looked for delivered masks in the wrong directory. The original failure is preserved and hashed. A one-expression path correction passed a check of the actual locations, and the corrected verifier succeeded. No GPU rerun or acceptance-rule change followed from this error.

This package contains aggregate numbers and hashes only: no source code, model geometry, GT, images, trajectories, original filesystem paths, weights or logs. Native adaptation remains experimental.
