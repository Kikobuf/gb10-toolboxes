# Self-Hosted ARM64 Runner for build-vllm-gb10

## Why

`build-vllm-gb10` on GitHub's hosted `ubuntu-24.04-arm` runners (4 vCPU / 16GB RAM) hit a
real ceiling twice over:

- At default parallelism, the build got silently killed partway through (no error, no
  logs archived — the signature of the runner itself being OOM-killed).
- Capped to `MAX_JOBS=2` it stopped crashing, but verbose build output showed genuine,
  steady compile progress through hundreds of CUTLASS-templated CUDA kernel files —
  just too slow to land inside any reasonable CI timeout (multiple hours).

The workload needs more cores *and* more RAM than the hosted runner offers, not a
trade-off between the two. A self-hosted runner with a bigger box fixes both at once,
and — just as importantly — lets you watch real memory/CPU usage live while it builds
(`htop`/`free -h` in a second SSH session) to actually tune `MAX_JOBS` empirically,
instead of guessing blind across hour-long CI runs.

The runner doesn't need to be GB10 hardware. The build never touches a real GPU — it
links against the CUDA driver *stub* the whole way through (see the linking discussion
in `toolboxes/llama-cpp-cuda/Dockerfile`); the real driver is only needed when the
built image actually *runs*, via the NVIDIA Container Toolkit on the target host. Any
plain ARM64 Linux box with Docker works as the build machine.

## 1. Create the box

[Oracle Cloud's Always Free tier](https://www.oracle.com/cloud/free/) includes an
Ampere A1 (ARM64) shape, up to **4 OCPU / 24GB RAM** total across your account, at no
cost indefinitely — a reasonable starting point (50% more RAM than the GitHub-hosted
runner at the same core count). If that's still not enough once you can watch real
usage, size up to a paid ARM64 instance (Oracle, AWS Graviton, etc.) instead of
guessing further.

1. Sign up / log in at [cloud.oracle.com](https://cloud.oracle.com).
2. **Compute → Instances → Create Instance**.
3. Image: **Ubuntu 24.04** (aarch64/ARM). Shape: **VM.Standard.A1.Flex**, set to the
   max free allowance (4 OCPU, 24GB memory).
4. Add your SSH public key during creation (or paste one in — see step 2 if you don't
   have a keypair handy).
5. Once running, note the instance's public IP. In the VCN's security list (or the
   instance's attached NSG), allow inbound SSH (port 22) from your IP if it isn't
   already open.

## 2. Connect from your phone

You only need the phone to do the one-time setup below — once the runner is registered
as a service it runs unattended on the box, independent of the phone.

- **iPhone**: an SSH client like [Termius](https://termius.com/) or
  [Blink Shell](https://blink.sh/), or Oracle's own browser-based Cloud Shell from the
  instance's console page (works fine from Safari).
- Generate a keypair in the app if you didn't provide one at instance-creation time,
  and add the public key to the instance (Oracle's console has an "Add SSH key" action
  on the instance page if you missed it at creation).

```bash
ssh ubuntu@<instance-public-ip>
```

## 3. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo usermod -aG docker $USER
newgrp docker
```

No NVIDIA Container Toolkit or GPU driver needed on this box — it's a build host, not a
runtime host (see "Why" above).

## 4. Register the runner

In the repo: **Settings → Actions → Runners → New self-hosted runner**, select Linux /
ARM64, and copy the generated registration token (it's short-lived, generate it fresh
when you get here).

```bash
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-arm64.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-arm64-<version>.tar.gz
tar xzf actions-runner-linux-arm64.tar.gz

./config.sh --url https://github.com/Kikobuf/gb10-toolboxes \
  --token <TOKEN_FROM_GITHUB> \
  --labels vllm-builder

sudo ./svc.sh install
sudo ./svc.sh start
```

The `--labels vllm-builder` tag is what `build.yml` targets — it keeps this workflow
pinned to this specific box rather than matching any self-hosted runner that might get
added later. `svc.sh install` sets it up as a systemd service, so it survives reboots
and keeps running without a terminal attached.

## 5. Point the workflow at it

`build-vllm-gb10` in `.github/workflows/build.yml` targets `runs-on: [self-hosted,
vllm-builder]`. `build-llama-cpp-cuda` stays on the hosted `ubuntu-24.04-arm` runner —
it already builds cleanly there in ~15 minutes, no need to move it.

## 6. Tune MAX_JOBS

`toolboxes/vllm-gb10/Dockerfile` ships with `MAX_JOBS=2` — the conservative value that
was proven stable (but slow) on the 16GB hosted runner. With more RAM available, SSH in
during a build and watch `free -h` / `htop` to see real headroom, then raise `MAX_JOBS`
(a value at or near the box's OCPU count is a reasonable next step) and re-run. Push a
Dockerfile change once you've found a value that's fast without swapping.

## Security note

This workflow only triggers on `push` to `main`, `schedule`, and `workflow_dispatch` —
never on `pull_request` — so a self-hosted runner here isn't exposed to arbitrary code
from external PRs (the usual reason self-hosted runners are risky on public repos). If
that ever changes, don't add `pull_request` as a trigger without also restricting who
can run it.
