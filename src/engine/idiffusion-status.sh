#!/bin/bash
# idiffusion-status.sh — Output status of all idiffusion jobs as JSON

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

JOBS_DIR="$IDIFFUSION_PROJECTDIR/idiffusion/jobs"

if [[ ! -d "$JOBS_DIR" ]]; then
    echo "[]"
    exit 0
fi

# Get list of currently running slurm jobs for the user across all partitions
active_slurm_jobs=$(squeue -u "$USER" --all -h -o "%i" 2>/dev/null | tr '\n' ' ' || true)

python3 -c "
import glob
import json
import os
import sys

jobs_dir = '$JOBS_DIR'
active_slurm = set('$active_slurm_jobs'.split())
results = []

for status_path in glob.glob(os.path.join(jobs_dir, '*', 'status.json')):
    try:
        with open(status_path, 'r') as f:
            data = json.load(f)
        
        slurm_id = str(data.get('slurmJobId', '')).strip()
        current_status = data.get('status', '')
        
        # Check if slurm job ended while lockfile still thought it was running/pending
        if slurm_id and slurm_id not in active_slurm and current_status in ('running', 'initialising', 'pending'):
            # Double check with specific job query to prevent false-positives
            ret = os.system(f'squeue -j {slurm_id} -h -o \"%i\" >/dev/null 2>&1')
            if ret != 0:
                data['status'] = 'stopped'
                data['reason'] = 'SLURM job ended'
                try:
                    with open(status_path, 'w') as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass

        # Clear stale reason if job is running
        if data.get('status') == 'running' and data.get('reason') == 'SLURM job ended':
            data.pop('reason', None)
            try:
                with open(status_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

        results.append(data)
    except Exception:
        pass

# Sort by requestedTime descending
results.sort(key=lambda x: x.get('requestedTime', ''), reverse=True)
print(json.dumps(results))
"
