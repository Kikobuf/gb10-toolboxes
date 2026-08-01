# AGENTS.md

Context for AI coding agents (Claude Code, etc.) working on this repo.

## What this repo is

Containerized `toolbox`/`distrobox` images for running llama.cpp and vLLM on the NVIDIA
GB10 Grace Blackwell Superchip (DGX Spark / ASUS Ascent GX10). Sibling repos in this
family target Intel/AMD GPUs, Tenstorrent accelerators, Rockchip NPUs, and AWS
Trainium/Inferentia.

## Key facts to keep in mind when editing

- GB10 is **`sm_121a`** — a consumer/edge Blackwell variant, distinct from datacenter
  Blackwell (`sm_100`/`sm_101`, B100/B200). Never conflate these; a build targeting one
  will not work correctly on the other. `CUDA_ARCH`/`TORCH_CUDA_ARCH_LIST` build args
  must stay pinned to 121/12.1a specifically.
- GB10 is **ARM64**, unlike most other NVIDIA-targeted tooling which assumes x86_64.
  Dockerfiles and CI must target `linux/arm64`; note in docs that emulated builds (QEMU)
  are slow and less reliable than building natively on the hardware.
- GB10 has **unified memory** (128GB shared CPU+GPU, no separate VRAM pool). Don't add
  VRAM-sizing logic that assumes a fixed GPU memory number — `nvidia-smi` won't report
  one. Memory planning here is closer to desktop RAM budgeting.
- Dense models are **bandwidth-bound** (LPDDR5X ~273 GB/s), not compute-bound. Any
  guidance encouraging a model choice must account for this — MoE architectures and
  aggressive quantization (AWQ 4-bit, NVFP4) are meaningfully better fits than
  unquantized dense models. Don't lose this framing when editing `docs/model-guidance.md`.
- Official NVIDIA images for GB10 have historically lagged behind current vLLM releases
  — this is the entire reason `vllm-gb10` builds from source with a pinned `VLLM_REF`
  rather than wrapping an official image the way `tt-vllm-toolboxes` or
  `neuron-vllm-toolboxes` do. Don't "simplify" this repo by switching to wrapping an
  official NVIDIA image without re-verifying the lag issue is resolved.
- `FLASHINFER` attention backend has a known warmup bug in community GB10 builds as of
  this writing — `FLASH_ATTN` is the documented default. Don't switch the default backend
  without checking current upstream status first.

## Things NOT to do

- Don't merge this repo's scope with generic/desktop NVIDIA GPU tooling — GB10's
  ARM64+unified-memory+new-compute-capability combination is what justifies this repo
  existing at all; a generic "NVIDIA toolbox" would mostly duplicate Ollama/NGC/vLLM's
  own official support, which is already excellent for typical discrete GPUs.
- Don't assume x86_64 CI/build patterns from sibling repos transfer directly — always
  double-check ARM64-specific assumptions (runner type, buildx platform flags, build
  time expectations).
