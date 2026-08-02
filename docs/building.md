# Building Locally

Both toolboxes here are source builds by design — official prebuilt images for GB10's
`sm_121a` architecture have historically lagged behind current releases, so pinning and
building your own is often more reliable than waiting on upstream image cadence.

## llama-cpp-cuda

```bash
docker build \
  --build-arg LLAMA_CPP_REF=master \
  --build-arg CUDA_ARCH=121 \
  -t gb10-toolboxes:llama-cpp-cuda \
  toolboxes/llama-cpp-cuda
```

Pin `LLAMA_CPP_REF` to a specific tag/commit if you want reproducibility rather than
always tracking `master`. Build time: roughly 10-20 minutes on GB10 hardware itself.

## vllm-gb10

```bash
docker build \
  --build-arg VLLM_REF=v0.16.0 \
  --build-arg TORCH_CUDA_ARCH_LIST="12.1a" \
  -t gb10-toolboxes:vllm-gb10 \
  toolboxes/vllm-gb10
```

**Expect this to take significantly longer** than the llama.cpp build — vLLM's build
process on ARM64 is not fast. Community reports suggest 45-90+ minutes depending on the
host and whether you're building natively on the Spark vs. cross-compiling.

Check [vllm-project/vllm releases](https://github.com/vllm-project/vllm/releases) for the
current stable tag before picking a `VLLM_REF` — this repo does not auto-track vLLM's
release cadence by default (see the GitHub Action for the opt-in dispatch-triggered
tracking behavior).

## Cross-building from an x86_64 dev machine

If you're not building directly on GB10 hardware, use `docker buildx` with QEMU
emulation:

```bash
docker buildx build --platform linux/arm64 -t gb10-toolboxes:llama-cpp-cuda toolboxes/llama-cpp-cuda
```

Note: emulated ARM64 builds are considerably slower than native builds, and CUDA-related
build steps in particular may behave unpredictably under emulation. Building directly on
the Spark/GX10 itself is the more reliable path for these images specifically.

## Pushing to your own registry

```bash
docker tag gb10-toolboxes:vllm-gb10 ghcr.io/<you>/gb10-toolboxes:vllm-gb10
docker push ghcr.io/<you>/gb10-toolboxes:vllm-gb10
```
