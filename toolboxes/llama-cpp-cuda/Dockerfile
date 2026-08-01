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

# The build container has the CUDA toolkit but no physical GPU/driver, so the
# real libcuda.so isn't present. Link against the CUDA driver stub instead;
# the NVIDIA Container Toolkit / host driver provides the real libcuda.so.1
# at runtime.
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"

# Build with CUDA support, explicitly targeting the GB10's compute capability.
# Do NOT rely on "auto" arch detection here — sm_121a is new enough that older
# CMake/CUDA toolkit combinations may not auto-detect it correctly.
#
# The LIBRARY_PATH stub above is enough for ggml-cuda's own .so to link (a
# shared object is allowed unresolved symbols by default). But the test/CLI
# executables link against that .so by full path rather than "-lcuda", so ld
# tries to resolve its transitive NEEDED entry (libcuda.so.1) by exact
# filename — and the stub dir only ships an unversioned "libcuda.so", so that
# lookup fails and every driver-API symbol comes back as an undefined
# reference for the executable link. -Wl,--allow-shlib-undefined tells ld to
# trust that libcuda.so.1 (provided at runtime by the NVIDIA Container
# Toolkit) will satisfy those symbols instead of requiring proof now.
RUN mkdir -p build && cd build && \
    cmake .. -G Ninja \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    ninja

ENV PATH="/opt/llama.cpp/build/bin:${PATH}"

WORKDIR /workspace
CMD ["/bin/bash"]
