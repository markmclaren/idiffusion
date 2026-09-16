import type { DiffusionLockfile } from './types';
import { format, isPast } from 'date-fns';

/**
 * Format a single job's lockfile status as a single-row string.
 *
 * Columns: JOB, STATUS, PORT, COMPUTE, SLURM, UNTIL, INFO, USER, MODEL.
 */
export function formatJobRow(job: DiffusionLockfile): string {
    const stop =
        job.stopTime && !isPast(new Date(job.stopTime))
            ? format(new Date(job.stopTime), 'HH:mm dd/MM')
            : undefined;

    const parts: string[] = [
        job.jobName.padEnd(12),
        (job.engine ?? 'diffusers').padEnd(10).slice(0, 10),
        job.status.padEnd(12).slice(0, 12),
        (job.serverPort ? String(job.serverPort) : '-').padEnd(8).slice(0, 8),
        (job.computeHostname ?? '-').padEnd(12).slice(0, 12),
        (job.slurmJobId ?? '-').padEnd(10).slice(0, 10),
        (stop ?? '-').padEnd(12),
        (job.reason ?? '-').padEnd(14).slice(0, 14),
        (job.user ?? 'unknown').padEnd(12).slice(0, 12),
        job.model ?? 'unknown',
    ];
    return parts.join('  ').trimEnd();
}

/**
 * Format a list of jobs as a table with header, separator, and rows.
 */
export function formatJobTable(jobs: DiffusionLockfile[]): string {
    if (jobs.length === 0) return 'No idiffusion jobs found.';
    const header =
        'JOB'.padEnd(12) +
        '  ' +
        'ENGINE'.padEnd(10) +
        '  ' +
        'STATUS'.padEnd(12) +
        '  ' +
        'PORT'.padEnd(8) +
        '  ' +
        'COMPUTE'.padEnd(12) +
        '  ' +
        'SLURM'.padEnd(10) +
        '  ' +
        'UNTIL'.padEnd(12) +
        '  ' +
        'INFO'.padEnd(14) +
        '  ' +
        'USER'.padEnd(12) +
        '  ' +
        'MODEL';

    const separator = '-'.repeat(header.length);
    const rows = jobs.map(formatJobRow);
    return [header, separator, ...rows].join('\n');
}
