#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${MECH_BENCH_SOLVER_IMAGE:-mech-bench-solver:local}"
PLATFORM="${MECH_BENCH_SOLVER_PLATFORM:-}"

BUILD_PLATFORM_ARGS=("")
RUN_PLATFORM_ARGS=("")
if [[ -n "${PLATFORM}" ]]; then
  BUILD_PLATFORM_ARGS=(--platform "${PLATFORM}")
  RUN_PLATFORM_ARGS=(--platform "${PLATFORM}")
fi

if [[ -z "${DOCKER_BUILDKIT:-}" ]]; then
  if docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=1
  else
    export DOCKER_BUILDKIT=0
  fi
fi

docker build \
  ${BUILD_PLATFORM_ARGS[@]} \
  -f "${ROOT}/docker/solver/Dockerfile" \
  -t "${IMAGE}" \
  "${ROOT}"

docker run --rm ${RUN_PLATFORM_ARGS[@]} "${IMAGE}" \
  oracle-smoke --require-real
