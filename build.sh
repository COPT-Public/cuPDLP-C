#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CM="${CMAKE:-cmake}"
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODE="${MODE:-cpu}"
case "$MODE" in cpu|gpu) modes="$MODE";; all) modes="cpu gpu";; *) echo "MODE must be cpu, gpu or all" >&2; exit 2;; esac
if [ -z "${HIGHS_HOME:-}" ]; then
  "$CM" -S ThirdParty/HiGHS -B .coinor-deps/highs-build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PWD/.coinor-deps/highs" -DBUILD_SHARED_LIBS=ON -DFAST_BUILD=ON
  "$CM" --build .coinor-deps/highs-build -j4
  "$CM" --install .coinor-deps/highs-build
  export HIGHS_HOME="$PWD/.coinor-deps/highs"
fi
"$PY" tests/generate_problem.py
for mode in $modes; do
  flag=OFF; args=()
  if [ "$mode" = gpu ]; then
    flag=ON
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    args+=("-DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCH:-native}")
  fi
  prefix="$PWD/.coinor-deps/cupdlp-$mode-install"
  "$CM" -S . -B "build-coinor-$mode" -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA="$flag" \
    -DBUILD_PYTHON=OFF -DBUILD_APPS=OFF -DBUILD_TESTING=ON -DCMAKE_INSTALL_PREFIX="$prefix" "${args[@]}"
  "$CM" --build "build-coinor-$mode" -j4
  "$CM" --build "build-coinor-$mode" --target test
  "$CM" --install "build-coinor-$mode"
  "$PY" "$prefix/share/cupdlp/tests/run_tests.py" --solver "$prefix/bin/plc"
  "$PY" tests/run.py --project cuPDLP-C --mode "$mode" --binary "$prefix/bin/plc" --output "tests/logs/$mode"
done
"$PY" tests/check_checker.py --project cuPDLP-C
