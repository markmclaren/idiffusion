# isambard-diffusion (`idiffusion`)

A CLI tool for running Hugging Face **Diffusers** and **ComfyUI** on [Isambard AI](https://www.isambard.ac.uk/) HPC from your local machine.

Inspired by [`ivllm`](https://github.com/ai4ci/ivllm), `idiffusion` manages SLURM compute jobs on NVIDIA GH200 Grace Hopper nodes, downloads and caches models in shared project storage, establishes an SSH tunnel to compute nodes, and exposes both an **OpenAI-compatible Images REST API** (port 8000) and an **interactive ComfyUI Web Interface** (port 8188).

```
[Local Mac / PC]
   │
   ├── http://localhost:8000/v1  ←── SSH Tunnel ──▶  FastAPI + Diffusers (REST API)
   │
   └── http://localhost:8188/    ←── SSH Tunnel ──▶  ComfyUI (Interactive Web UI)
                                                     Compute Node (GH200 H100 96GB/144GB)
```

---

## Features

- **Dual-Engine Architecture**:
  - **`diffusers`**: Headless lightweight FastAPI server exposing an OpenAI-compatible `/v1/images/generations` API.
  - **`comfyui`**: Full interactive node graph Web UI with real-time WebSocket queueing running directly in your browser.
- **HPC Supercomputing Power**: Runs inference on NVIDIA GH200 Grace Hopper Superchips (96GB/144GB unified HBM3 memory).
- **Fast Generation**: Generate full-resolution 1024x1024 images in 1–2 seconds with FLUX.1-schnell.
- **Shared Multi-User Architecture**:
  - **Environment**: Shared Python venv built once at `$PROJECTDIR/idiffusion/env/venv` — all project members share it.
  - **Models**: Weights cached in shared `$PROJECTDIR/model/hf/` — downloads happen once per project. ComfyUI automatically accesses this via `extra_model_paths.yaml`.
  - **Jobs**: Anyone in the project can view (`idiffusion status`) and connect to running jobs.
- **Smart Idle Timeout**: Automatically terminates jobs after periods of inactivity to conserve GPU allocation hours.
- **Automatic Config Inheritance**: Automatically picks up connection settings from existing `~/.config/ivllm/config.json` if configured!

---

## Prerequisites

1. **Bun** (or Node.js):
   ```bash
   npm install -g bun
   # or curl -fsSL https://bun.sh/install | bash
   ```
2. **SSH Connection**: Working SSH key connection to the Isambard AI login node cached in `ssh-agent`.
3. **Hugging Face Token**: Needed for gated models (e.g. FLUX.1-dev, SD 3.5).

---

## Installation

```bash
cd /Users/ismjml/Experiments/idiffusion

# Install dependencies and link binary globally
bun install
bun link
```

Now `idiffusion` is available in your shell.

---

## Configuration

If you already use `ivllm`, `idiffusion` automatically inherits your settings! Otherwise, configure connection details:

```bash
idiffusion config --login-host b6ai.aip2.isambard
idiffusion config --username ismjml.b6ai
idiffusion config --project-dir /projects/b6ai
idiffusion config --hf-token hf_...
```

To view current settings:
```bash
idiffusion config
```

---

## One-Time Cluster Setup: `idiffusion setup`

Before launching your first job, run `setup` once per project to build the PyTorch + Diffusers + ComfyUI environment on a GH200 compute node:

```bash
idiffusion setup
```

Once installed, all members of your project share this environment.

---

## Workflows

### 🎨 Option A: Launch Interactive ComfyUI in Your Browser

Launch a ComfyUI session on a compute node and automatically open the web UI on your Mac:

```bash
idiffusion comfy
```

This starts ComfyUI on an allocated GH200 GPU, sets up the SSH tunnel, mounts the shared project model storage, and opens `http://localhost:8188/` in your browser!

```
========================================================================
🎨 ComfyUI Web Interface Ready!
========================================================================
  Web UI:         http://localhost:8188/
  WebSocket API:  ws://localhost:8188/ws
  Job Name:       comfy
  SLURM Job ID:   123456
  Shared Models:  Auto-mounted from /projects/b6ai/model

Opening http://localhost:8188/ in your browser...
[Press Ctrl+C at any time to disconnect the tunnel. The HPC session will remain active.]
========================================================================
```

---

### ⚡ Option B: Headless Diffusers (OpenAI REST API)

Launch a headless Diffusers server on port 8000:

```bash
idiffusion connect flux --config examples/flux-1-schnell.yaml
```

#### Generate Images from CLI:
```bash
idiffusion generate \
  --prompt "A cinematic shot of an astronaut riding a horse on Mars, high detail, 8k" \
  --output mars.png
```

#### Generate Images from Python (OpenAI SDK):
```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="placeholder")

response = client.images.generate(
    model="black-forest-labs/FLUX.1-schnell",
    prompt="A magical glowing forest at night, fantasy digital art",
    size="1024x1024",
    response_format="b64_json"
)

image_data = base64.b64decode(response.data[0].b64_json)
with open("forest.png", "wb") as f:
    f.write(image_data)
```

---

### 📊 Check Job Status

```bash
idiffusion status
```

Output:
```
JOB           ENGINE      STATUS        PORT      COMPUTE       SLURM       UNTIL         INFO            USER          MODEL
--------------------------------------------------------------------------------------------------------------------------------------
comfy         comfyui     running       51942     nid001234     123456      18:00 16/09   -               ismjml.b6ai   ComfyUI Workspace
flux          diffusers   running       52104     nid001235     123457      18:00 16/09   -               ismjml.b6ai   black-forest-labs/FLUX.1-schnell
```

---

### 🛑 Cancel a Job

```bash
# Clean graceful cancel
idiffusion cancel comfy

# Force kill via scancel
idiffusion cancel comfy --force
```

---

## Configuration Presets

Ready-to-use YAML configurations are available in [`examples/`](examples/):

| Preset | File | Engine | Description |
|---|---|---|---|
| **ComfyUI Session** | `examples/comfyui.yaml` | `comfyui` | Interactive node graph web interface on port 8188 |
| **FLUX.1 [schnell]** | `examples/flux-1-schnell.yaml` | `diffusers` | Ultra-fast 4-step generation (~1–2s per image) |
| **FLUX.1 [dev]** | `examples/flux-1-dev.yaml` | `diffusers` | 28-step 12B parameter high-fidelity model |
| **Stable Diffusion 3.5 Large** | `examples/sd-3.5-large.yaml` | `diffusers` | 8B parameter MMDiT from Stability AI |
| **SDXL 1.0** | `examples/sdxl.yaml` | `diffusers` | Classic Stable Diffusion XL |

---

## Commands Summary

| Command | Description |
|---|---|
| `idiffusion comfy [jobName] [--local-port <port>]` | Launch interactive ComfyUI in browser (`localhost:8188`) |
| `idiffusion connect <job> [--config <file>] [--comfy]` | Start or connect to an image server and forward local port |
| `idiffusion generate -p <prompt> [-o <path>]` | Generate an image via active local tunnel |
| `idiffusion status [job]` | Show status table of all jobs on the HPC |
| `idiffusion cancel <job> [--force]` | Cancel a running job |
| `idiffusion log <job>` | View or tail remote server log |
| `idiffusion config [options]` | View or update connection settings |
| `idiffusion setup [--force]` | Build the shared Python environment on HPC |