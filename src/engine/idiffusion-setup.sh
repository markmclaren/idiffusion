#!/bin/bash
# idiffusion-setup.sh — Sets up the shared Diffusers venv on Isambard AI

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

FORCE=0
USE_SLURM=0
OPTIND=1
while getopts "fsh" opt; do
    case $opt in
        f) FORCE=1 ;;
        s) USE_SLURM=1 ;;
        h) echo "Usage: $0 [-f (force reinstall)] [-s (submit via slurm)]"; exit 0 ;;
        *) echo "Usage: $0 [-f] [-s]"; exit 1 ;;
    esac
done

VENV_DIR="$(resolve_venv_dir)"
SETUP_LOG="$IDIFFUSION_PROJECTDIR/idiffusion/setup.log"

if [[ -f "$VENV_DIR/bin/activate" && $FORCE -eq 0 ]]; then
    echo "idiffusion environment is already installed at:"
    echo "  $VENV_DIR"
    echo "Use 'idiffusion setup --force' to reinstall."
    exit 0
fi

echo "=================================================="
echo "Installing idiffusion environment on Isambard AI"
echo "  Target: $VENV_DIR"
echo "  Log:    $SETUP_LOG"
echo "=================================================="

mkdir -p "$(resolve_env_dir)" "$(dirname "$SETUP_LOG")"

# If SLURM was explicitly requested with -s, submit via sbatch
if [[ $USE_SLURM -eq 1 ]]; then
    TEMP_SETUP_SCRIPT="$(resolve_env_dir)/run_setup_worker.sh"
    cat > "$TEMP_SETUP_SCRIPT" <<EOF
#!/bin/bash
set -eo pipefail
"$here/idiffusion-setup.sh" -f
EOF
    chmod +x "$TEMP_SETUP_SCRIPT"

    # Check if interactive reservation is available and accessible
    PARTITION_FLAGS=""
    if sbatch --test-only --partition=interactive --reservation=interactive --wrap="true" >/dev/null 2>&1; then
        PARTITION_FLAGS="--partition=interactive --reservation=interactive"
    fi

    echo "Submitting setup job to SLURM..."
    # shellcheck disable=SC2086
    sbatch \
        --wait \
        --job-name="idiff_setup" \
        --nodes=1 \
        --gpus-per-node=1 \
        --cpus-per-gpu=16 \
        --mem=64G \
        --time="01:00:00" \
        --output="$SETUP_LOG" \
        --error="$SETUP_LOG" \
        $PARTITION_FLAGS \
        "$TEMP_SETUP_SCRIPT"

    EXIT_CODE=$?
    rm -f "$TEMP_SETUP_SCRIPT"
    exit $EXIT_CODE
fi

# Direct setup on login host (Linux aarch64, shared Lustre filesystem)
exec > >(tee -a "$SETUP_LOG") 2>&1

echo "[setup] Host:         $(hostname)"
echo "[setup] Architecture: $(uname -m)"
echo "[setup] Date:         $(date -u)"

# Ensure uv is in PATH
if ! command -v uv &>/dev/null; then
    if [[ -f "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "[setup] Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# Clean existing venv if force reinstall requested
if [[ -d "$VENV_DIR" && $FORCE -eq 1 ]]; then
    echo "[setup] Cleaning old virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create virtualenv using uv with Python 3.12 (standard for modern PyTorch / diffusers)
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "[setup] Creating Python 3.12 virtual environment at $VENV_DIR..."
    if command -v uv &>/dev/null; then
        uv venv "$VENV_DIR" --python 3.12 || uv venv "$VENV_DIR" --python 3.11 || uv venv "$VENV_DIR"
    else
        python3.11 -m venv "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
    fi
fi

# Activate environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[setup] Installing PyTorch with CUDA 12.4 support..."
if command -v uv &>/dev/null; then
    uv pip install \
        --index-url "https://download.pytorch.org/whl/cu124" \
        torch torchvision torchaudio

    echo "[setup] Installing Diffusers, Transformers, and core dependencies..."
    uv pip install \
        "diffusers>=0.30.0" \
        transformers \
        accelerate \
        sentencepiece \
        protobuf \
        safetensors \
        fastapi \
        uvicorn \
        pydantic \
        huggingface_hub \
        pyyaml \
        torchsde \
        einops \
        scipy \
        aiohttp \
        tqdm \
        psutil \
        alembic \
        kornia \
        spandrel \
        soundfile
else
    pip install --upgrade pip setuptools wheel
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install "diffusers>=0.30.0" transformers accelerate sentencepiece protobuf safetensors fastapi uvicorn pydantic huggingface_hub pyyaml
    pip install torchsde einops scipy aiohttp tqdm psutil alembic kornia spandrel soundfile
fi

COMFY_DIR="$IDIFFUSION_PROJECTDIR/idiffusion/comfyui"
if [[ ! -d "$COMFY_DIR/.git" ]]; then
    echo "[setup] Cloning ComfyUI repository to $COMFY_DIR..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" || true
else
    echo "[setup] ComfyUI repository already present at $COMFY_DIR."
fi

if [[ -f "$COMFY_DIR/requirements.txt" ]]; then
    echo "[setup] Installing/updating ComfyUI requirements..."
    if command -v uv &>/dev/null; then
        uv pip install -r "$COMFY_DIR/requirements.txt" || true
    else
        python3 -m pip install -r "$COMFY_DIR/requirements.txt" || true
    fi
fi

echo "[setup] Verifying installation..."
python3 -c "
import sys
import torch
import diffusers

print('✓ Python:            ', sys.version.split()[0])
print('✓ PyTorch Version:   ', torch.__version__)
cuda_ver = getattr(torch.version, 'cuda', None)
print('✓ PyTorch CUDA Build:', cuda_ver if cuda_ver else 'CPU-only')
print('✓ Diffusers Version: ', diffusers.__version__)
if torch.cuda.is_available():
    print('✓ GPU Device:        ', torch.cuda.get_device_name(0))
else:
    print('✓ Host verification complete (GPU will be engaged when jobs run on compute nodes).')
"

echo "=================================================="
echo "✓ idiffusion environment setup complete!"
echo "=================================================="

