#!/bin/bash
# idiffusion-serve.sh — Launches a SLURM job for diffusers on compute node

set -uo pipefail

usage() {
    echo "Usage: $0 -j <jobName> [-t <time>] [-m <model>] [-e <diffusers|comfyui>] [-b]"
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

JOB=""
MAX_TIME="08:00:00"
PARTITION_FLAGS=""
MODEL_OVERRIDE=""
ENGINE_OVERRIDE=""
BATCH_MODE=0

OPTIND=1
while getopts "j:t:m:e:bh" opt; do
    case $opt in
        j) JOB="$OPTARG" ;;
        t) MAX_TIME="$OPTARG" ;;
        m) MODEL_OVERRIDE="$OPTARG" ;;
        e) ENGINE_OVERRIDE="$OPTARG" ;;
        b) BATCH_MODE=1 ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ $BATCH_MODE -eq 0 ]]; then
    # Auto-detect if interactive reservation is accessible for current user
    if sbatch --test-only --partition=interactive --reservation=interactive --wrap="true" >/dev/null 2>&1; then
        PARTITION_FLAGS="--partition=interactive --reservation=interactive"
    else
        PARTITION_FLAGS=""
    fi
fi

if [[ -z "$JOB" ]]; then
    echo "[serve] ERROR: No job parameter supplied (-j <job>)" >&2
    exit 1
fi

CONFIG_FILE="$(resolve_job_config "$JOB")"
MODEL="black-forest-labs/FLUX.1-schnell"
IDLE_TIMEOUT="30"
TORCH_DTYPE="bfloat16"
ENGINE="diffusers"

# Parse configuration if present
if [[ -f "$CONFIG_FILE" ]]; then
    parsed=$(python3 -c "
import yaml, sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        cfg = yaml.safe_load(f) or {}
    print(cfg.get('model', ''))
    print(cfg.get('idle-timeout', 30))
    print(cfg.get('torch-dtype', 'bfloat16'))
    print(cfg.get('engine', 'diffusers'))
except Exception:
    pass
" 2>/dev/null || true)
    cfg_model=$(echo "$parsed" | sed -n '1p')
    cfg_timeout=$(echo "$parsed" | sed -n '2p')
    cfg_dtype=$(echo "$parsed" | sed -n '3p')
    cfg_engine=$(echo "$parsed" | sed -n '4p')
    if [[ -n "$cfg_model" ]]; then MODEL="$cfg_model"; fi
    if [[ -n "$cfg_timeout" ]]; then IDLE_TIMEOUT="$cfg_timeout"; fi
    if [[ -n "$cfg_dtype" ]]; then TORCH_DTYPE="$cfg_dtype"; fi
    if [[ -n "$cfg_engine" ]]; then ENGINE="$cfg_engine"; fi
fi

if [[ -n "$MODEL_OVERRIDE" ]]; then
    MODEL="$MODEL_OVERRIDE"
fi
if [[ -n "$ENGINE_OVERRIDE" ]]; then
    ENGINE="$ENGINE_OVERRIDE"
fi

if [[ "$ENGINE" == "comfyui" ]]; then
    MODEL="ComfyUI Workspace"
    IDLE_TIMEOUT="${IDLE_TIMEOUT:-60}"
fi

STATUS_FILE="$(resolve_job_status "$JOB")"
LOG_FILE="$(resolve_job_log "$JOB")"

# Check if job is already active
if [[ -f "$STATUS_FILE" ]]; then
    current_status=$(python3 -c "import json; print(json.load(open('$STATUS_FILE')).get('status', ''))" 2>/dev/null || echo "")
    if [[ "$current_status" == "running" || "$current_status" == "initialising" || "$current_status" == "pending" ]]; then
        echo "[serve] Job '$JOB' is already in state [$current_status]"
        exit 0
    fi
fi

# Create pending lockfile
create_status_pending "$JOB" "$MODEL" "$IDLE_TIMEOUT"
if [[ "$ENGINE" == "comfyui" ]]; then
    python3 -c "import json; f=open('$STATUS_FILE', 'r+'); d=json.load(f); d['engine']='comfyui'; f.seek(0); json.dump(d, f, indent=2); f.truncate()" 2>/dev/null || true
fi

echo "=================================================="
echo "Submitting idiffusion job: $JOB"
echo "  Engine:       $ENGINE"
echo "  Model:        $MODEL"
echo "  Max runtime:  $MAX_TIME"
echo "  Idle timeout: $IDLE_TIMEOUT mins"
if [[ -n "$PARTITION_FLAGS" ]]; then
    echo "  Partition:    interactive (reservation)"
else
    echo "  Partition:    workq (batch queue)"
fi
echo "  Log:          $LOG_FILE"
echo "=================================================="

# Select SLURM script based on engine
if [[ "$ENGINE" == "comfyui" ]]; then
    RUNNER="$here/lib/slurm-idiffusion-comfy.sh"
    RUNNER_ARGS=("$JOB" "$IDLE_TIMEOUT")
else
    RUNNER="$here/lib/slurm-idiffusion-serve.sh"
    RUNNER_ARGS=("$JOB" "$MODEL" "$IDLE_TIMEOUT" "$TORCH_DTYPE")
fi

export IDIFFUSION_ENGINE_DIR="$here"

# Submit sbatch job
slurmJobId=$(sbatch \
    --parsable \
    --chdir="$here" \
    --export=ALL,IDIFFUSION_ENGINE_DIR="$here" \
    --job-name "idiff_$JOB" \
    --nodes=1 \
    --gpus-per-node=1 \
    --cpus-per-gpu=16 \
    --mem=115G \
    --time="$MAX_TIME" \
    --output="$LOG_FILE" \
    --error="$LOG_FILE" \
    --open-mode=truncate \
    $PARTITION_FLAGS \
    "$RUNNER" "${RUNNER_ARGS[@]}")

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 && -n "$slurmJobId" ]]; then
    echo "✓ SLURM job submitted: $slurmJobId"
    update_status_slurm_id "$JOB" "$slurmJobId"
else
    echo "ERROR: SLURM job submission failed" >&2
    update_status_failed "$JOB" "SLURM submission failed"
    exit 1
fi
