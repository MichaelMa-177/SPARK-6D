# Review candidate: online mesh-conditioned SPARK-6D source

This is an exact-source review snapshot for the real `tentative-preloss`,
`sam-score`, `continuity` entry. It contains no experiment-results code, data,
GT, images, model weights, compiled extensions, Git database, credentials or
local environment. Original source bytes and notices are preserved. It has not
been published or tested as a clean-machine installation.

`SOURCE_MANIFEST.json` records the real online module imports, the entry's
explicit source inventory and a recursive static local-import closure. Two
CPU test files exercise the online adapter and production entry; their two
imported fixture files and three unchanged meshless import helpers are included.
Those helper imports do not constitute a SAM3D installation or execution claim. The
full 1658-test regression applied to the original checkout; the complete test
suite is not claimed to be included here. No source file has been edited to
hide a portability issue. Datareader retains three unused upstream LINEMOD
absolute-path literals; this explicit-sequence entry does not use those readers.

## License and external dependencies

The project retains its NVIDIA license: research/evaluation use only. Preserve
this complete LICENSE and all file-level attribution. This package does not
relicense upstream code or authorize commercial use. SAM2's separate Apache-2.0
LICENSE is retained with its source snapshot. TAPNext is fetched separately from
its official Apache-2.0 repository at the exact required commit; keeping a clean
Git checkout is part of the runtime source check. No new license is invented for
new files beyond the containing repository's existing terms.

`EXTERNAL_DEPENDENCIES.json` pins source and required model/configuration bytes,
with observed package versions and official source links. Weight download and
use remain subject to their original terms. The project build recipes describe
Python 3.11, CMake, Eigen, Boost, pybind11 and OpenMP. Install a matching CUDA
PyTorch stack plus PyTorch3D and nvdiffrast, then requirements.txt and build mycpp
locally. This path also requires hydra-core, einops, iopath and plyfile; their
observed versions are in EXTERNAL_DEPENDENCIES.json because the old generic
requirements.txt is not a complete pinned installer for this new entry.
No local binary is portable by assumption. The CUDA BundleSDF/NeRF
extension is optional and outside this mesh-conditioned validation.

The SAM2 runtime snapshot is now content-bound to official commit
[`2b90b9f5ceec907a1c18123530e92e794ad901a4`](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4).
All original 40 paths match: 36 regular files and four configuration symlinks
whose target contents were already materialized in the original source package.
This correspondence does not establish the historical local checkout's commit;
the parent SPARK commit is not a SAM2 commit. All 98 original source files retain
their exact bytes. Version 2 adds the official `setup.py`, `pyproject.toml`,
`MANIFEST.in`, `README.md` and referenced `LICENSE_cctorch` unchanged. The package
remains a runtime subset; links in the upstream README may refer to notebooks,
training scripts and assets available only in the full official repository.

An isolated copy built `sam_2-1.0-py3-none-any.whl` offline in the existing research
environment, with `SAM2_BUILD_CUDA=0`, `SAM2_BUILD_ALLOW_ERRORS=0`, `PIP_NO_INDEX=1`,
`--no-deps` and `--no-build-isolation`. Nothing was installed. All 25 Python files,
13 YAML files and both license files in that wheel match their packaged sources;
all 44 wheel RECORD entries were checked. The wheel is a private verification
artifact and is not included in this source delivery. Its hashes and exact scope
are recorded in EXTERNAL_DEPENDENCIES.json and SOURCE_MANIFEST.json.

This CPU packaging check does not test a clean machine, resolve dependencies,
load a model, build `_C`, or establish equivalence to the actual GPU experiment.
The CUDA source and official build metadata are present for a separate build with
a matching CUDA/PyTorch toolchain. Missing `_C` can affect SAM2 post-processing;
no inference-equivalence claim is made for the extension-free wheel.

## Explicit launch contract

Use registered RGB/depth PNGs with exactly matching sorted basenames and a
`cam_K.txt` 3x3 camera matrix. Mesh coordinates and camera-Z depth must be metres;
set the raw-depth multiplier explicitly. Click coordinates are on the first
RGB frame, with no GT mask, GT pose or future-frame query. Runtime source frame
IDs are the zero-based input order; no capture timestamp is inferred.

After dependencies, matching native extensions and the separately provided
weight/configuration files are installed:

```bash
python tools/run_click_to_model.py \
  --sequence /path/to/registered_rgbd --mesh /path/to/object.ply \
  --click 172 160 --depth-scale 0.0001 --max-frames 0 --seed 0 \
  --register-iterations 5 --max-refine-iterations 3 --mode adaptive \
  --tracking-policy tentative-preloss --working-mask-selection sam-score \
  --output-policy continuity --initialization-admission all-hypotheses \
  --tapnext-source-root /path/to/clean/tapnet \
  --tapnext-checkpoint /path/to/tapnextpp.pt \
  --tapnext-checkpoint-sha256 cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da \
  --sam2-checkpoint /path/to/sam2.1_hiera_small.pt \
  --output-dir /path/to/new_empty_output
```

The sample click/depth scale describe the known synthetic integration case;
replace them for your actual sensor/object without claiming the same experiment.
FoundationPose weight directories retain their required relative locations in
EXTERNAL_DEPENDENCIES.json. Symmetry metadata, when supplied, uses the existing
BOP object-entry format with metre translations; none is silently inferred.

`poses.npy` is raw candidate evidence. Consumers must use the per-frame
availability and published arrays in `output_policy.json` and
`published_poses.npz`: an abstention has no published pose. All identity and
trust flags remain false; a tentative publication is not physical identity
certification. Exceptions close the output epoch. Reclick means a new process/run.

This snapshot does not include SAM3D runtime dependencies, live-camera continuity,
research weight-adaptation prototype, synthetic generator, GPU raw observer or
experiment supervisors. It therefore does not claim to reproduce every research
artifact or the entire project's optional modes. The actual 180-frame online
run was separately audited; its results do not replace clean-install validation.
