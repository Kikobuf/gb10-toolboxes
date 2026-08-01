#!/usr/bin/env bash
#
# refresh-toolboxes.sh — pull the latest image for one or all gb10 toolbox
# tags and recreate the corresponding toolbox/distrobox container.
#
# Usage:
#   ./refresh-toolboxes.sh all
#   ./refresh-toolboxes.sh llama-cpp-cuda
#   ./refresh-toolboxes.sh vllm-gb10

set -euo pipefail

REGISTRY="${GB10_TOOLBOXES_REGISTRY:-ghcr.io/kikobuf/gb10-toolboxes}"

if command -v toolbox >/dev/null 2>&1; then
  TOOLCMD="toolbox"
elif command -v distrobox >/dev/null 2>&1; then
  TOOLCMD="distrobox"
else
  echo "Neither 'toolbox' nor 'distrobox' found on PATH. Install one first." >&2
  exit 1
fi

refresh_one() {
  local tag="$1"
  local container_name="gb10-${tag}"
  local image="${REGISTRY}:${tag}"
  local extra_flags="--gpus all"

  if [[ "$tag" == "vllm-gb10" ]]; then
    extra_flags="--gpus all --ipc host"
  fi

  echo "==> Refreshing ${container_name} (${image})"
  docker pull "${image}" || podman pull "${image}"

  if ${TOOLCMD} list 2>/dev/null | grep -q "${container_name}"; then
    echo "    Removing existing container ${container_name}"
    ${TOOLCMD} rm -f "${container_name}" || true
  fi

  echo "    Creating ${container_name}"
  ${TOOLCMD} create "${container_name}" \
    --image "${image}" \
    -- ${extra_flags}

  echo "    Done. Enter with: ${TOOLCMD} enter ${container_name}"
}

TARGET="${1:-all}"

case "$TARGET" in
  all)
    refresh_one "llama-cpp-cuda"
    refresh_one "vllm-gb10"
    ;;
  llama-cpp-cuda|vllm-gb10)
    refresh_one "$TARGET"
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    echo "Usage: $0 [all|llama-cpp-cuda|vllm-gb10]" >&2
    exit 1
    ;;
esac
