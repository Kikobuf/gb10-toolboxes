# vllm-gb10 toolbox (sm_121a)
#
# Builds vLLM from source pinned to a specific ref, targeting the GB10's
# compute capability. Built from source (rather than wrapping NVIDIA's
# official image) because official GB10 vLLM images have repeatedly lagged
# behind current vLLM releases — see README "Why GB10 Needs Its Own Toolbox".
#
# Must run on an ARM64 host with the NVIDIA Container Toolkit installed.

FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

LABEL org.opencontainers.image.source="https://github.com/Kikobuf/gb10-toolboxes"
LABEL org.opencontainers.image.description="vLLM built for GB10 (sm_121a), packaged for toolbox/distrobox"

ARG VLLM_REF=v0.16.0
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV DEBIAN_FRONTEND=noninteractive

# libnuma-dev: vLLM's CPU backend (csrc/cpu/utils.cpp) includes numa.h
# directly; without the header the build fails with "fatal error: numa.h:
# No such file or directory" regardless of CUDA architecture.
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        curl \
        git \
        vim \
        htop \
        nvtop \
        python3 \
        python3-dev \
        python3.12-dev \
        python3-pip \
        python3-venv \
        build-essential \
        cmake \
        ninja-build \
        ca-certificates \
        libnuma-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vllm

RUN git clone --depth 1 --branch ${VLLM_REF} https://github.com/vllm-project/vllm.git . \
    || (git clone https://github.com/vllm-project/vllm.git . && git checkout ${VLLM_REF})

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Cap parallel compile jobs (vLLM's setup.py reads MAX_JOBS). Left at its
# default, ninja spawns one compile job per core, and each CUDA/CUTLASS
# kernel translation unit here is large enough that 4 concurrent jobs can
# exceed a 16GB-RAM build host, silently OOM-killing the build with no
# error in the log. Trades build time for peak memory.
ENV MAX_JOBS=2

# GB10 is ARM64 — this build takes considerably longer than an x86_64 build.
# Expect 45-90+ minutes depending on host. See docs/building.md for details.
# No separate build-requirements install needed: pip install -e . resolves
# the build-time deps declared in pyproject.toml's [build-system] itself via
# PEP 517 build isolation.
# -v: without it, pip captures the build backend's cmake/ninja output and
# only shows it on failure, so a long build just prints an opaque "still
# running..." heartbeat with no way to tell real progress from a hang.
# -v streams that output live so we can see actual compile progress.
RUN pip install --no-cache-dir --break-system-packages -v -e .

# FLASH_ATTN is the recommended/stable backend on GB10 as of this writing.
# FLASHINFER has a known warmup bug in community builds — see docs/vllm-notes.md.
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN

EXPOSE 8000

WORKDIR /workspace
CMD ["/bin/bash"]
