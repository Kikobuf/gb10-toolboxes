# Host Configuration

## 1. Use DGX OS if possible

The DGX Spark ships with **DGX OS**, NVIDIA's own Ubuntu-based image with the correct
ARM64 Grace-Blackwell driver stack and CUDA 13 pre-installed and pre-configured. This is
the path of least resistance — avoid reinstalling drivers manually unless you have a
specific reason to (e.g. wanting a server-first Ubuntu setup, see
[timothystewart6/ubuntu-gb10](https://github.com/timothystewart6/ubuntu-gb10) if so).

## 2. Verify the base stack before touching containers

```bash
nvidia-smi
nvcc --version
```

- `nvidia-smi` should show the GB10 with compute capability **12.1**
- `nvcc --version` should report **CUDA 13.x**

If either of these is wrong or missing, fix it at the host level first — no container
configuration will compensate for a broken base driver/CUDA install.

Note: `nvidia-smi` will **not** show a dedicated VRAM figure — GB10 has no separate VRAM
pool. The 128GB is shared system memory accessed by both CPU and GPU over NVLink-C2C.

## 3. Install the NVIDIA Container Toolkit

Required for `--gpus all` passthrough into `toolbox`/`distrobox` containers:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

## 4. Memory planning (no separate VRAM pool)

Since GB10 shares one 128GB pool between OS, CPU workloads, and GPU inference, plan
memory the way you would for a desktop machine's RAM, not the way you'd plan around a
discrete GPU's fixed VRAM:

- Leave headroom for the OS and any other running processes
- KV cache grows with context length and concurrent requests — factor this in alongside
  model weight size, not just the model's raw parameter count
- See [Unified Memory & Quantization Guidance](../README.md#unified-memory--quantization-guidance)
  in the main README for how this affects model/quantization choice

## 5. Networking note (vLLM specifically)

Community vLLM setups on GB10 commonly use `network_mode: host` in their compose files
(vLLM binds directly to the machine's real IP) rather than port-mapping — this avoids a
class of ARM64/container-networking quirks reported on this platform. See
[`scripts/run_server.sh`](../scripts/run_server.sh) for how this repo's wrapper handles it.

## 6. Multi-Spark / clustering

If you're connecting multiple DGX Spark / GX10 units for larger models (e.g. via Ray),
that's a more advanced, less standardized setup as of this writing — check current
NVIDIA Developer Forum threads on "DGX Spark multi-node" for the latest community
approach rather than relying on a static guide here, since this area is still evolving
quickly.
