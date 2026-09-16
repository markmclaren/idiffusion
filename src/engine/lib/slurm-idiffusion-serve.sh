#!/bin/bash
# slurm-idiffusion-serve.sh — Runs on the compute node inside SLURM allocation

set -uo pipefail

JOB_NAME="${1:-}"
MODEL="${2:-black-forest-labs/FLUX.1-schnell}"
IDLE_TIMEOUT="${3:-30}"
TORCH_DTYPE="${4:-bfloat16}"

if [[ -z "$JOB_NAME" ]]; then
    echo "[compute] ERROR: No job name passed" >&2
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

# If running outside SLURM (where SLURM --output isn't already active), tee to log file
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=================================================="
echo "Starting idiffusion compute server for job: $JOB_NAME"
echo "  Node: $(hostname)"
echo "  Date: $(date -u)"
echo "  Model: $MODEL"
echo "=================================================="

# Check virtualenv
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    echo "[compute] Activating virtualenv: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "[compute] ERROR: Virtual environment not found at $VENV_DIR" >&2
    echo "[compute] Run 'idiffusion setup' first to build the environment." >&2
    update_status_failed "$JOB_NAME" "virtualenv missing"
    exit 1
fi

# Point HuggingFace cache to shared project directory
export HF_HOME="$(resolve_models_dir)"
export TRANSFORMERS_CACHE="$HF_HOME"
export DIFFUSERS_CACHE="$HF_HOME"
echo "[compute] HuggingFace cache: $HF_HOME"

# Find an available high port (ephemeral port range)
COMPUTE_HOST="$(hostname -s)"
PORT=$(python3 -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()" 2>/dev/null || shuf -i 49152-65535 -n 1)

echo "[compute] Bound to host: $COMPUTE_HOST, port: $PORT"

# Update lockfile to initialising
python3 -c "
import json
try:
    with open('$STATUS_FILE', 'r+') as f:
        d = json.load(f)
        d['status'] = 'initialising'
        d['computeHostname'] = '$COMPUTE_HOST'
        d['serverPort'] = int('$PORT')
        f.seek(0)
        json.dump(d, f, indent=2)
        f.truncate()
except Exception as e:
    print('Failed to update status:', e)
" 2>/dev/null || true

# Run diffusers server
SERVER_SCRIPT="$ENGINE_DIR/server/diffusers_server.py"

python3 "$SERVER_SCRIPT" \
    --host "0.0.0.0" \
    --port "$PORT" \
    --model "$MODEL" \
    --idle-timeout "$IDLE_TIMEOUT" \
    --torch-dtype "$TORCH_DTYPE" \
    --status-file "$STATUS_FILE"

EXIT_CODE=$?

echo "[compute] Server exited with code $EXIT_CODE"

# Ensure status is updated to stopped if not already
python3 -c "
import json
from datetime import datetime, timezone
try:
    with open('$STATUS_FILE', 'r+') as f:
        d = json.load(f)
        if d.get('status') != 'failed':
            d['status'] = 'stopped'
            d['stopTime'] = datetime.now(timezone.utc).isoformat()
            if 'reason' not in d:
                d['reason'] = 'server completed'
            f.seek(0)
            json.dump(d, f, indent=2)
            f.truncate()
except Exception:
    pass
" 2>/dev/null || true

exit $EXIT_CODE
