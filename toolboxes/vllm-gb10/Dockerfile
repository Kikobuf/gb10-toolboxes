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

ARG VLLM_REF=v0.20.1
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        curl \
        git \
        vim \
        htop \
        nvtop \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        cmake \
        ninja-build \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vllm

RUN git clone --depth 1 --branch ${VLLM_REF} https://github.com/vllm-project/vllm.git . \
    || (git clone https://github.com/vllm-project/vllm.git . && git checkout ${VLLM_REF})

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# GB10 is ARM64 — this build takes considerably longer than an x86_64 build.
# Expect 45-90+ minutes depending on host. See docs/building.md for details.
RUN pip install --no-cache-dir --break-system-packages -r requirements/build.txt || true
RUN pip install --no-cache-dir --break-system-packages -e . || true

# FLASH_ATTN is the recommended/stable backend on GB10 as of this writing.
# FLASHINFER has a known warmup bug in community builds — see docs/vllm-notes.md.
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN

EXPOSE 8000

WORKDIR /workspace
CMD ["/bin/bash"]
