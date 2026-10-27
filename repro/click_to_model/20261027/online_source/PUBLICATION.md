# Source snapshot accompanying the online continuity experiment

This directory is a self-contained source root for the mesh-conditioned online
entry reviewed against the 180-frame experiment. Run its commands from this
directory after obtaining the declared external dependencies and weights.
`README_DELIVERY.md` is the preserved preparation report: its "not published"
statement describes preparation time, before this snapshot was added to GitHub.

The package preserves 98 source files from the actual runtime checkout and adds
five byte-exact official SAM2 build/license files. Three dependency/source
reports document the scope. Original licenses remain in force.

The corresponding numeric experiment is in `../continuity_online_aggregates/`.
The actual candidates were identical to its original baseline; selective output
reduced observed published errors while also withholding correct poses. This
source snapshot is not evidence of improved candidate accuracy or solved
physical-instance identity.

Validation: the original checkout passed 1658 CPU tests and the actual full
180-frame GPU run passed raw-transfer audit. The packaged entry/adapter passed
18 relocation CPU tests in the existing environment. The isolated SAM2 CPU wheel
built offline; 25 Python, 13 YAML and both licenses were exact, with all 44 wheel
RECORD entries checked independently again. That wheel, weights, binaries,
images, GT, experiment logs and private environment are not included.

A clean-machine CUDA build and complete end-to-end installation have not been
validated. This snapshot excludes live-camera continuity, the SAM3D runtime,
neural-adaptation research prototypes and newer birth-policy experiments.
See `EXTERNAL_DEPENDENCIES.json` for exact source and required weight hashes.
