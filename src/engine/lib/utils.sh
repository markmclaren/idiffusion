#!/bin/bash
# utils.sh — Shared bash utilities for idiffusion

if [[ -n "${IDIFFUSION_UTILS:-}" ]] && declare -f resolve_job_status > /dev/null; then
    return 0
fi
IDIFFUSION_UTILS=1

if [[ -z ${IDIFFUSION_PROJECTDIR:-} ]]; then
    export IDIFFUSION_PROJECTDIR=${PROJECTDIR:-$HOME/idiffusion}
fi

export IDIFFUSION_GRP=$(stat "$IDIFFUSION_PROJECTDIR" -c %g 2>/dev/null || id -g)

# Ensure TMPDIR exists and is writable on compute nodes
if [[ -n "${TMPDIR:-}" ]]; then
    mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR="/tmp/idiffusion-${USER:-user}"
else
    export TMPDIR="/tmp/idiffusion-${USER:-user}"
fi
mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR="/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# Group writable permissions
umask 0002

mkdir -p "$IDIFFUSION_PROJECTDIR/idiffusion/jobs" "$IDIFFUSION_PROJECTDIR/idiffusion/env" "$IDIFFUSION_PROJECTDIR/model/hf" 2>/dev/null || true

resolve_job_dir() {
    local job="$1"
    echo "$IDIFFUSION_PROJECTDIR/idiffusion/jobs/$job"
}

resolve_job_status() {
    local job="$1"
    echo "$(resolve_job_dir "$job")/status.json"
}

resolve_job_log() {
    local job="$1"
    echo "$(resolve_job_dir "$job")/server.log"
}

resolve_job_config() {
    local job="$1"
    echo "$(resolve_job_dir "$job")/diffusion.yaml"
}

resolve_env_dir() {
    echo "$IDIFFUSION_PROJECTDIR/idiffusion/env"
}

resolve_venv_dir() {
    echo "$IDIFFUSION_PROJECTDIR/idiffusion/env/venv"
}

resolve_models_dir() {
    echo "$IDIFFUSION_PROJECTDIR/model/hf"
}

create_status_pending() {
    local job="$1"
    local model="$2"
    local idle_timeout="${3:-30}"
    local job_dir="$(resolve_job_dir "$job")"
    local status_file="$(resolve_job_status "$job")"

    mkdir -p "$job_dir"
    cat > "$status_file" <<EOF
{
  "status": "pending",
  "jobName": "$job",
  "model": "$model",
  "serverPort": 0,
  "user": "$USER",
  "requestedTime": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "idleTimeout": $idle_timeout,
  "resources": "1 GPUs"
}
EOF
}

update_status_slurm_id() {
    local job="$1"
    local slurm_id="$2"
    local status_file="$(resolve_job_status "$job")"
    if [[ -f "$status_file" ]]; then
        python3 -c "import json; f=open('$status_file', 'r+'); d=json.load(f); d['slurmJobId']='$slurm_id'; f.seek(0); json.dump(d, f, indent=2); f.truncate()" 2>/dev/null || true
    fi
}

update_status_failed() {
    local job="$1"
    local reason="$2"
    local status_file="$(resolve_job_status "$job")"
    if [[ -f "$status_file" ]]; then
        python3 -c "import json; f=open('$status_file', 'r+'); d=json.load(f); d['status']='failed'; d['reason']='$reason'; f.seek(0); json.dump(d, f, indent=2); f.truncate()" 2>/dev/null || true
    fi
}
