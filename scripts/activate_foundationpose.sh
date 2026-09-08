#!/usr/bin/env bash
# Source this file before running or building FoundationPose:
#   source scripts/activate_foundationpose.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced: source scripts/activate_foundationpose.sh" >&2
  exit 2
fi

FP_DEFAULT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FP_REPO_ROOT="${FOUNDATIONPOSE_REPO_ROOT:-${FP_DEFAULT_ROOT}}"
FP_ENV_PREFIX="${FOUNDATIONPOSE_ENV_PREFIX:-${CONDA_PREFIX:-}}"

if [[ -z "${FP_ENV_PREFIX}" || ! -x "${FP_ENV_PREFIX}/bin/python" ]]; then
  echo "Activate the FoundationPose Conda environment or set FOUNDATIONPOSE_ENV_PREFIX." >&2
  return 1
fi
if [[ ! -f "${FP_REPO_ROOT}/run_demo.py" ]]; then
  echo "FoundationPose repository not found: ${FP_REPO_ROOT}" >&2
  return 1
fi

export CONDA_PREFIX="${FP_ENV_PREFIX}"
export PATH="${FP_ENV_PREFIX}/bin:${PATH}"
export CUDA_HOME="${FP_ENV_PREFIX}"
export CUDACXX="${FP_ENV_PREFIX}/bin/nvcc"

# PyTorch 2.6/cu124 extensions are built with the host GCC 11 toolchain.
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
export CUDAHOSTCXX="/usr/bin/g++"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"

FP_PY_SITE="$("${FP_ENV_PREFIX}/bin/python" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
FP_NVIDIA_SITE="${FP_PY_SITE}/nvidia"
export CPATH="${FP_NVIDIA_SITE}/cuda_runtime/include:${FP_NVIDIA_SITE}/cublas/include:${FP_NVIDIA_SITE}/cusparse/include:${FP_NVIDIA_SITE}/cusolver/include:${FP_NVIDIA_SITE}/cufft/include:${FP_NVIDIA_SITE}/curand/include:${FP_NVIDIA_SITE}/cudnn/include:${FP_ENV_PREFIX}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${FP_ENV_PREFIX}/lib:${FP_NVIDIA_SITE}/cuda_runtime/lib:${FP_NVIDIA_SITE}/cublas/lib:${FP_NVIDIA_SITE}/cusparse/lib:${FP_NVIDIA_SITE}/cusolver/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${FP_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FP_REPO_ROOT}:${FP_REPO_ROOT}/mycpp/build${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export QT_QPA_FONTDIR="${QT_QPA_FONTDIR:-/usr/share/fonts/truetype}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"

cd "${FP_REPO_ROOT}"
echo "FoundationPose environment active: ${FP_ENV_PREFIX}"
echo "Repository: ${FP_REPO_ROOT}"
echo "CUDA arch: ${TORCH_CUDA_ARCH_LIST}"
