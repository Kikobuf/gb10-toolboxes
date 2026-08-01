# vLLM Notes for GB10

## Attention backend

- **`FLASH_ATTN`** — recommended default. vLLM auto-selects this on GB10 in recent
  versions (FlashAttention 2).
- **`FLASHINFER`** — listed as an available option but has a known bug in community
  builds: a `non_blocking=None TypeError` on warmup. Avoid until this is resolved
  upstream; explicitly set `VLLM_ATTENTION_BACKEND=FLASH_ATTN` if you're not sure which
  one got selected.

## Networking

Use `network_mode: host` in Docker Compose (or `--network host` with `docker run`) rather
than port-mapping. vLLM then binds directly to the machine's real IP with no forwarding
needed — this sidesteps a class of ARM64/container-networking quirks reported by
community GB10 vLLM deployments.

## Served model naming

If you're wrapping a coding assistant or other tool that expects a specific model name at
the API level, set it explicitly rather than relying on the HF repo ID:

```bash
--served-model-name my-model-name
```

`scripts/run_server.sh` exposes this as `--served-model-name` — override it to match
whatever your client tooling expects.

## Compilation / first-run behavior

Like other CUDA-based Blackwell deployments, expect a slower first request while kernels
warm up / any JIT compilation completes. This is normal and not a hang — subsequent
requests are much faster.

## Tracking upstream vLLM releases

Because official NVIDIA GB10 vLLM images have lagged behind current vLLM releases in the
past, this repo builds from source pinned to a specific `VLLM_REF` tag rather than
depending on NVIDIA's image cadence. Check
[vllm-project/vllm releases](https://github.com/vllm-project/vllm/releases) for the
current version before bumping `VLLM_REF` in `toolboxes/vllm-gb10/Dockerfile`, and check
the [NVIDIA Developer Forums DGX Spark category](https://forums.developer.nvidia.com/c/nvidia-dgx-platforms/dgx-spark/)
for any GB10-specific compatibility notes on that version before assuming a clean
upgrade.

## Multi-GPU / multi-Spark setups

Community reports of running vLLM + Ray across two DGX Spark units exist but are less
standardized than single-unit setups as of this writing. If you need this, search the
NVIDIA Developer Forums for current threads rather than assuming this repo's default
`run_server.sh` invocation extends automatically — it doesn't currently handle multi-node
orchestration.
