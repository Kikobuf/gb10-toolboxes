# llama-cpp-cuda toolbox for GB10 (sm_121a)
#
# Builds llama.cpp from source with CUDA support targeted at the GB10's
# compute capability. Must run on an ARM64 host with the NVIDIA Container
# Toolkit installed (see docs/host-config.md).

FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

LABEL org.opencontainers.image.source="https://github.com/Kikobuf/gb10-toolboxes"
LABEL org.opencontainers.image.description="llama.cpp built for GB10 (sm_121a), packaged for toolbox/distrobox"

ARG LLAMA_CPP_REF=master
ARG CUDA_ARCH=121
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        curl \
        git \
        vim \
        htop \
        nvtop \
        cmake \
        build-essential \
        ninja-build \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/llama.cpp

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp.git . \
    || (git clone --depth 1 https://github.com/ggml-org/llama.cpp.git . && git checkout ${LLAMA_CPP_REF})

# Build with CUDA support, explicitly targeting the GB10's compute capability.
# Do NOT rely on "auto" arch detection here — sm_121a is new enough that older
# CMake/CUDA toolkit combinations may not auto-detect it correctly.
RUN mkdir -p build && cd build && \
    cmake .. -G Ninja \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_BUILD_TYPE=Release && \
    ninja

ENV PATH="/opt/llama.cpp/build/bin:${PATH}"

WORKDIR /workspace
CMD ["/bin/bash"]
