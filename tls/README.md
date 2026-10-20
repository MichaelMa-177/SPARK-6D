# Server project snapshot

`clm/` contains the 2026-10-20 source snapshot of `/home/mmw/tls/clm`.
The existing SPARK-6D repository remains at the repository root.

The snapshot includes project scripts, experiment code, small historical training checkpoints, saved human first-frame prompts and inputs, reference reports, and vendored source with its original licenses. `SNAPSHOT_MANIFEST.json` records file hashes and original symbolic-link targets. Existing licenses in each source subtree continue to apply; inclusion does not relicense third-party code.

Machine-specific compiled extensions, SQLite files, Python caches and nested Git metadata are excluded. The surrounding `/home/mmw/tls` proxy/service configuration, external datasets, virtual environment, large SAM3D/DINO weights and linked evaluation outputs are outside this snapshot. Existing absolute symlinks and hardcoded paths describe the original server installation; review them before running legacy scripts.

The original experiment results are historical baselines, not validation of the new latent-memory model.
