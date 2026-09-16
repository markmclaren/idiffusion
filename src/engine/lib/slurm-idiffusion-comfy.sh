#!/bin/bash
# slurm-idiffusion-comfy.sh — Runs ComfyUI on compute node under SLURM allocation

set -uo pipefail

JOB_NAME="${1:-comfy}"
IDLE_TIMEOUT="${2:-60}"

if [[ -z "$JOB_NAME" ]]; then
    echo "[comfy] ERROR: No job name passed" >&2
    exit 1
fi

# Determine engine directory:
# In SLURM batch execution, ${BASH_SOURCE[0]} points to /var/spool/slurmd/job<ID>/slurm_script.
# Resolve using IDIFFUSION_ENGINE_DIR, SLURM_SUBMIT_DIR, or standard fallback paths.
if [[ -n "${IDIFFUSION_ENGINE_DIR:-}" && -f "$IDIFFUSION_ENGINE_DIR/lib/utils.sh" ]]; then
    ENGINE_DIR="$IDIFFUSION_ENGINE_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/lib/utils.sh" ]]; then
    ENGINE_DIR="$SLURM_SUBMIT_DIR"
elif [[ -f "$HOME/.local/share/idiffusion/engine/lib/utils.sh" ]]; then
    ENGINE_DIR="$HOME/.local/share/idiffusion/engine"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$script_dir/utils.sh" ]]; then
        ENGINE_DIR="$(cd "$script_dir/.." && pwd)"
    elif [[ -f "$script_dir/lib/utils.sh" ]]; then
        ENGINE_DIR="$script_dir"
    else
        echo "[compute] ERROR: Cannot locate idiffusion engine directory" >&2
        exit 1
    fi
fi

unset IDIFFUSION_UTILS
source "$ENGINE_DIR/lib/utils.sh"

STATUS_FILE="$(resolve_job_status "$JOB_NAME")"
LOG_FILE="$(resolve_job_log "$JOB_NAME")"
VENV_DIR="$(resolve_venv_dir)"
COMFY_DIR="$IDIFFUSION_PROJECTDIR/idiffusion/comfyui"

# If running outside SLURM (where SLURM --output isn't already active), tee to log file
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=================================================="
echo "Starting ComfyUI compute server for job: $JOB_NAME"
echo "  Node: $(hostname)"
echo "  Date: $(date -u)"
echo "  Idle timeout: $IDLE_TIMEOUT mins"
echo "=================================================="

# Check virtual environment
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    echo "[comfy] Activating virtualenv: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "[comfy] ERROR: Virtual environment not found at $VENV_DIR" >&2
    echo "[comfy] Run 'idiffusion setup' first to build the environment." >&2
    update_status_failed "$JOB_NAME" "virtualenv missing"
    exit 1
fi

# Ensure ComfyUI repo is cloned
if [[ ! -d "$COMFY_DIR/.git" ]]; then
    echo "[comfy] ComfyUI not found at $COMFY_DIR, cloning..."
    mkdir -p "$IDIFFUSION_PROJECTDIR/idiffusion"
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi

# Set HuggingFace cache to shared project directory
export HF_HOME="$(resolve_models_dir)"
export TRANSFORMERS_CACHE="$HF_HOME"
export DIFFUSERS_CACHE="$HF_HOME"

# Generate extra_model_paths.yaml to share models with /projects/b6ai/model
EXTRA_MODELS_PATH="$(resolve_job_dir "$JOB_NAME")/extra_model_paths.yaml"
cat > "$EXTRA_MODELS_PATH" <<EOF
isambard:
    base_path: $IDIFFUSION_PROJECTDIR/model
    checkpoints: hf/hub:checkpoints
    unet: hf/hub:unet
    clip: hf/hub:clip
    vae: hf/hub:vae
    loras: loras
    controlnet: controlnet
    upscale_models: upscale_models
EOF

# Find an available high ephemeral port
COMPUTE_HOST="$(hostname -s)"
PORT=$(python3 -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()" 2>/dev/null || shuf -i 49152-65535 -n 1)

echo "[comfy] Bound to host: $COMPUTE_HOST, port: $PORT"

# Update lockfile to running
python3 -c "
import json
from datetime import datetime, timezone
try:
    with open('$STATUS_FILE', 'r+') as f:
        d = json.load(f)
        d['status'] = 'running'
        d['engine'] = 'comfyui'
        d['model'] = 'ComfyUI Workspace'
        d['computeHostname'] = '$COMPUTE_HOST'
        d['serverPort'] = int('$PORT')
        d['startTime'] = datetime.now(timezone.utc).isoformat()
        f.seek(0)
        json.dump(d, f, indent=2)
        f.truncate()
except Exception as e:
    print('Failed to update status:', e)
" 2>/dev/null || true

# Start ComfyUI
pushd "$COMFY_DIR" > /dev/null

echo "[comfy] Launching ComfyUI on 0.0.0.0:$PORT..."
python3 main.py \
    --listen 0.0.0.0 \
    --port "$PORT" \
    --extra-model-paths-config "$EXTRA_MODELS_PATH" &

COMFY_PID=$!

# Idle monitor loop in background
(
    timeout_secs=$(( IDLE_TIMEOUT * 60 ))
    if (( timeout_secs > 0 )); then
        last_active=$(date +%s)
        while kill -0 "$COMFY_PID" 2>/dev/null; do
            sleep 30
            # Check if there are active network connections to the ComfyUI port
            active_conns=$(ss -tn "sport = :$PORT" 2>/dev/null | grep -v "State" | wc -l || echo 0)
            now=$(date +%s)
            if (( active_conns > 0 )); then
                last_active=$now
            else
                idle=$(( now - last_active ))
                if (( idle >= timeout_secs )); then
                    echo "[comfy] Idle timeout of $IDLE_TIMEOUT mins reached with no connections. Shutting down..."
                    kill -TERM "$COMFY_PID" 2>/dev/null || true
                    break
                fi
            fi
        done
    fi
) &
IDLE_PID=$!

# Wait for ComfyUI process
wait "$COMFY_PID"
EXIT_CODE=$?

kill -TERM "$IDLE_PID" 2>/dev/null || true
popd > /dev/null

echo "[comfy] ComfyUI exited with code $EXIT_CODE"

# Update lockfile to stopped
python3 -c "
import json
from datetime import datetime, timezone
try:
    with open('$STATUS_FILE', 'r+') as f:
        d = json.load(f)
        d['status'] = 'stopped'
        d['stopTime'] = datetime.now(timezone.utc).isoformat()
        if 'reason' not in d or not d['reason']:
            d['reason'] = 'server completed'
        f.seek(0)
        json.dump(d, f, indent=2)
        f.truncate()
except Exception:
    pass
" 2>/dev/null || true

exit $EXIT_CODE
