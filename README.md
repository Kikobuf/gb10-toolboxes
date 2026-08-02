# NVIDIA GB10 Toolboxes

Pre-built `toolbox`/`distrobox` containers for running LLMs on the **NVIDIA GB10 Grace
Blackwell Superchip** — the ARM64 CPU+GPU unified-memory SoC inside the **NVIDIA DGX
Spark** and **ASUS Ascent GX10**. Covers both llama.cpp (CUDA backend, built for
`sm_121a`) and vLLM, since official prebuilt images have historically lagged behind
current releases on this specific architecture.

Companion projects:
- [`amd-strix-halo-toolboxes`](https://github.com/kyuz0/amd-strix-halo-toolboxes) — closest analog: AMD's unified-memory iGPU
- `intel-igpu-toolboxes` — Intel Arc / integrated GPUs
- [`tt-metal-toolboxes`](https://github.com/Kikobuf/tt-metal-toolboxes) / [`tt-vllm-toolboxes`](https://github.com/Kikobuf/tt-vllm-toolboxes) — Tenstorrent accelerators
- `rk3588-toolboxes` — Rockchip RK3588/RK3576 NPUs
- `neuron-vllm-toolboxes` — AWS Trainium/Inferentia

---

## Table of Contents

- [Why GB10 Needs Its Own Toolbox](#why-gb10-needs-its-own-toolbox)
- [Supported Hardware](#supported-hardware)
- [Supported Toolboxes](#supported-toolboxes)
- [Quick Start](#quick-start)
- [Unified Memory & Quantization Guidance](#unified-memory--quantization-guidance)
- [Host Configuration](#host-configuration)
- [Running llama.cpp](#running-llamacpp)
- [Running vLLM](#running-vllm)
- [Building Locally](#building-locally)
- [Keeping Updated](#keeping-updated)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Why GB10 Needs Its Own Toolbox

Unlike most NVIDIA hardware — which is extremely well served by existing tooling
(Ollama, NGC containers, official vLLM/llama.cpp releases) — the GB10 sits in an
awkward gap:

- **New compute capability.** GB10 is `sm_121a` (consumer/edge Blackwell), distinct from
  datacenter Blackwell (`sm_100`/`sm_101` — B100/B200). Generic "Blackwell" CUDA images
  built for datacenter cards do not automatically work here.
- **ARM64 + unified memory, not a typical discrete-GPU x86 host.** Most vLLM/llama.cpp
  container tutorials assume an x86 host with a dedicated-VRAM GPU. GB10 breaks both
  assumptions: it's Grace (ARM) CPU fused with Blackwell GPU sharing one 128GB pool over
  NVLink-C2C, with no separate VRAM number to check via `nvidia-smi`.
  Only 30 GB visible in `nvidia-smi` if this happens — check driver.
- **Official images lag.** Community reports of NVIDIA's own vLLM image being over a
  month behind current releases by the time it's needed are common — hence a small
  ecosystem of community-maintained GB10 vLLM images has sprung up already.

This repo exists to package the "we figured this out the hard way" builds so you don't
have to re-derive them from forum threads every time.

## Supported Hardware

| Device                    | SoC | Memory                  | Notes                          |
| ---------------------------- | ----- | -------------------------- | ---------------------------------|
| NVIDIA DGX Spark             | GB10 | 128GB unified (LPDDR5X)   | Reference platform                |
| ASUS Ascent GX10             | GB10 | 128GB unified (LPDDR5X)   | OEM variant, same SoC             |
| Other GB10-based mini workstations | GB10 | 128GB unified (LPDDR5X) | Should behave identically         |

Compute capability: `sm_121a`. Memory bandwidth: ~273 GB/s (LPDDR5X), which — unlike
discrete GPU VRAM bandwidth — is the binding constraint for dense model throughput far
more often than raw compute.

## Supported Toolboxes

| Container Tag       | What it wraps                                              | Notes                                          |
| ---------------------- | -------------------------------------------------------------| --------------------------------------------------|
| `llama-cpp-cuda`      | llama.cpp built from source with `CMAKE_CUDA_ARCHITECTURES=121` | Lightweight, good for quantized dense models     |
| `vllm-gb10`           | vLLM built/patched for `sm_121a`, tracking current releases   | OpenAI-compatible server, better for MoE + batching |

---

## Quick Start

**Prerequisites:** DGX Spark / ASUS Ascent GX10 running its stock DGX OS (or Ubuntu with
NVIDIA's ARM64 Grace-Blackwell driver stack installed) — see
[Host Configuration](#host-configuration).

### llama.cpp (CUDA, sm_121a)

```bash
toolbox create gb10-llama-cpp \
  --image ghcr.io/kikobuf/gb10-toolboxes:llama-cpp-cuda \
  -- --gpus all

toolbox enter gb10-llama-cpp
```

### vLLM

```bash
toolbox create gb10-vllm \
  --image ghcr.io/kikobuf/gb10-toolboxes:vllm-gb10 \
  -- --gpus all --ipc host

toolbox enter gb10-vllm
```

*(Ubuntu users: use `distrobox create` / `distrobox enter` instead of `toolbox`. Both
containers assume the NVIDIA Container Toolkit is installed on the host — see
[Host Configuration](#host-configuration).)*

### Verify GPU visibility

```bash
nvidia-smi
```

You should see the GB10 listed. Note that `nvidia-smi` will **not** show a dedicated
VRAM number the way it would on a discrete GPU — the 128GB is shared system memory, not
a fixed GPU memory pool.

---

## Unified Memory & Quantization Guidance

This is the single most important thing to understand before picking a model on GB10.

- **Dense models are bandwidth-bound, not compute-bound.** A 7B dense model in BF16 tops
  out around ~19 tok/s theoretical, purely from LPDDR5X's ~273 GB/s bandwidth ceiling —
  more compute won't help past that wall.
- **Quantize aggressively for dense models.** AWQ 4-bit (or similar) meaningfully
  improves throughput by cutting the bytes-per-token moved from memory, which is the
  actual bottleneck here.
- **MoE models are a much better fit for GB10's unified memory.** Only the active
  expert(s) need to be "hot" in the compute path per token — e.g. a ~40GB MoE model might
  only move ~3B params worth of weights per token, letting a much larger total model run
  at speeds competitive with dense models many times smaller. If you have 128GB to work
  with, prefer a large MoE over a mid-size dense model.
- **Unified memory means no separate VRAM pool to manage.** You're not choosing "does
  this fit in VRAM" — you're choosing "does the OS + model + KV cache fit in the shared
  128GB," which behaves more like desktop RAM budgeting than typical GPU deployment.

See [`docs/model-guidance.md`](docs/model-guidance.md) for a fuller breakdown and example
model/quantization combinations that work well on GB10.

## Host Configuration

See [`docs/host-config.md`](docs/host-config.md) for the full walkthrough. Summary:

- Ship with DGX OS (NVIDIA's own Ubuntu-based image) when possible — it comes with the
  correct ARM64 Grace-Blackwell driver stack and CUDA 13 pre-configured
- NVIDIA Container Toolkit must be installed for `--gpus all` passthrough to work
- Confirm `nvcc --version` reports CUDA 13.x and `nvidia-smi` reports the GB10 with
  compute capability 12.1 before building or pulling containers

## Running llama.cpp

```bash
llama-cli \
  -m /models/your-model.Q4_K_M.gguf \
  -ngl 999 \
  -t 16 \
  -p "Your prompt here"
```

`-ngl 999` offloads all layers to GPU — with unified memory there's less reason to
partially offload the way you might on a VRAM-constrained discrete GPU setup. Monitor
with `nvtop` to confirm GPU utilization during inference.

## Running vLLM

```bash
./scripts/run_server.sh \
  --model Qwen/Qwen3-Coder-Next \
  --served-model-name coding-model
```

This wraps a `docker run`/`docker compose` invocation using `network_mode: host` (vLLM
binds directly to the machine's real IP — no port forwarding needed on GB10). See
[`scripts/run_server.sh`](scripts/run_server.sh) for the full flags, or
[`docs/vllm-notes.md`](docs/vllm-notes.md) for backend selection (FLASH_ATTN vs.
FLASHINFER — FLASHINFER has known issues in community builds as of this writing).

## Building Locally

Since official images lag, building your own pinned version is often the more reliable
path here:

```bash
# llama.cpp
docker build \
  --build-arg CUDA_ARCH=121 \
  -t gb10-toolboxes:llama-cpp-cuda \
  toolboxes/llama-cpp-cuda

# vLLM
docker build \
  --build-arg VLLM_REF=v0.20.1 \
  -t gb10-toolboxes:vllm-gb10 \
  toolboxes/vllm-gb10
```

See [`docs/building.md`](docs/building.md) for build arg details and expected build
times (vLLM from source on ARM64 is not fast). If you're wiring this up in CI, see
[`docs/self-hosted-runner.md`](docs/self-hosted-runner.md) — the vLLM build outgrows
GitHub's hosted ARM64 runners and needs a bigger self-hosted box.

## Keeping Updated

```bash
./refresh-toolboxes.sh all
```

---

## Troubleshooting

- **`nvidia-smi` doesn't show the GPU** → NVIDIA Container Toolkit not installed, or
  `--gpus all` not passed at container creation. See [Host Configuration](#host-configuration).
- **CUDA error: no kernel image available / unsupported arch** → you're running a build
  targeting the wrong compute capability. Confirm `sm_121a`/`121` was used, not a
  datacenter Blackwell (`sm_100`) or older-architecture build.
- **Model loads but throughput far below expectations** → check whether you're running a
  dense model that's bandwidth-bound (expected ceiling ~19 tok/s for a 7B BF16 dense
  model); consider AWQ 4-bit quantization or an MoE model instead.
- **FLASHINFER backend crashes on warmup** → known issue in community vLLM builds as of
  this writing (`non_blocking=None TypeError`); use `FLASH_ATTN` instead.
- **Official NVIDIA vLLM image doesn't support your model / is out of date** → this is
  exactly the gap this repo exists to paper over; check `docs/vllm-notes.md` for current
  community-maintained alternatives and their tracked vLLM version.

---

## References

- [vLLM on the DGX Spark — official vLLM blog](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark)
- [Arm Learning Paths: Build llama.cpp GPU on GB10](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/)
- [NVIDIA Developer Forums: DGX Spark / GB10 Projects](https://forums.developer.nvidia.com/c/nvidia-dgx-platforms/dgx-spark/)
- [shamily/vllm-gb10](https://github.com/shamily/vllm-gb10) — community vLLM setup reference
- [timothystewart6/ubuntu-gb10](https://github.com/timothystewart6/ubuntu-gb10) — server-first Ubuntu setup for DGX Spark

## Support

Hobby project, same spirit as the other toolbox repos in this family. Issues and PRs welcome.
