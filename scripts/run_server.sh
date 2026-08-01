#!/usr/bin/env bash
#
# run_server.sh — launch vLLM inside the gb10-vllm toolbox with sensible
# GB10-specific defaults (host networking, FLASH_ATTN backend).
#
# Usage:
#   ./run_server.sh --model <hf-model-id-or-path> [--served-model-name <name>] [--port <port>]

set -euo pipefail

MODEL=""
SERVED_NAME=""
PORT="8000"

usage() {
  echo "Usage: $0 --model <hf-model-id-or-path> [--served-model-name <name>] [--port <port>]"
  echo ""
  echo "  --model              HF model ID or local path"
  echo "  --served-model-name  Name exposed via the OpenAI-compatible API (default: derived from --model)"
  echo "  --port               Port to serve on (default: 8000; ignored if using host networking)"
  echo ""
  echo "See docs/model-guidance.md before picking a model/quantization combo."
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --served-model-name) SERVED_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required." >&2
  usage
fi

if [[ -z "$SERVED_NAME" ]]; then
  SERVED_NAME=$(basename "$MODEL")
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN is not set. Gated models will fail to download." >&2
fi

echo "==> Launching vLLM: model=${MODEL} served-as=${SERVED_NAME}"
echo "    Using network_mode=host (see docs/vllm-notes.md for why)."

docker run \
  --rm \
  --gpus all \
  --ipc host \
  --network host \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "VLLM_ATTENTION_BACKEND=FLASH_ATTN" \
  gb10-toolboxes:vllm-gb10 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}"
