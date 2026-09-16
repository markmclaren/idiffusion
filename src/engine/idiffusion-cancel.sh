#!/bin/bash
# idiffusion-cancel.sh — Cancel a running idiffusion job

set -uo pipefail

usage() {
    echo "Usage: $0 -j <jobName> [-f]"
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

JOB=""
FORCE=0

OPTIND=1
while getopts "j:fh" opt; do
    case $opt in
        j) JOB="$OPTARG" ;;
        f) FORCE=1 ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "$JOB" ]]; then
    echo "ERROR: -j <jobName> is required" >&2
    exit 1
fi

STATUS_FILE="$(resolve_job_status "$JOB")"

if [[ ! -f "$STATUS_FILE" ]]; then
    echo "ERROR: Job '$JOB' not found" >&2
    exit 1
fi

slurmJobId=$(python3 -c "import json; print(json.load(open('$STATUS_FILE')).get('slurmJobId', ''))" 2>/dev/null || echo "")

if [[ -n "$slurmJobId" && "$slurmJobId" != "null" ]]; then
    echo "Cancelling SLURM job $slurmJobId for '$JOB'..."
    scancel "$slurmJobId" 2>/dev/null || true
fi

# Update status file
python3 -c "
import json
from datetime import datetime, timezone
try:
    with open('$STATUS_FILE', 'r+') as f:
        d = json.load(f)
        d['status'] = 'stopped'
        d['reason'] = 'cancelled by user'
        d['stopTime'] = datetime.now(timezone.utc).isoformat()
        f.seek(0)
        json.dump(d, f, indent=2)
        f.truncate()
except Exception as e:
    print('Failed to update status file:', e)
" 2>/dev/null || true

echo "✓ Job '$JOB' cancelled."
