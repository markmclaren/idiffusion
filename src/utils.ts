import type { DiffusionLockfile } from './types';
import type { LocalSession } from './session-manager';
import { format, isPast } from 'date-fns';

/**
 * Format a single job's lockfile status as a single-row string.
 *
 * Columns: JOB, ENGINE, STATUS, LOCAL TUNNEL, PORT, COMPUTE, SLURM, UNTIL, INFO, USER, MODEL.
 */
export function formatJobRow(job: DiffusionLockfile, localSession?: LocalSession): string {
    const stop =
        job.stopTime && !isPast(new Date(job.stopTime))
            ? format(new Date(job.stopTime), 'HH:mm dd/MM')
            : undefined;

    const localTunnelStr = localSession ? `:${localSession.localPort} [PID ${localSession.pid}]` : '-';

    const parts: string[] = [
        job.jobName.padEnd(12),
        (job.engine ?? 'diffusers').padEnd(10).slice(0, 10),
        job.status.padEnd(12).slice(0, 12),
        localTunnelStr.padEnd(18).slice(0, 18),
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
export function formatJobTable(jobs: DiffusionLockfile[], localSessions?: LocalSession[]): string {
    if (jobs.length === 0) return 'No idiffusion jobs found.';
    const sessionMap = new Map<string, LocalSession>();
    if (localSessions) {
        for (const s of localSessions) {
            sessionMap.set(s.jobName, s);
        }
    }

    const header =
        'JOB'.padEnd(12) +
        '  ' +
        'ENGINE'.padEnd(10) +
        '  ' +
        'STATUS'.padEnd(12) +
        '  ' +
        'LOCAL TUNNEL'.padEnd(18) +
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
    const rows = jobs.map((job) => formatJobRow(job, sessionMap.get(job.jobName)));
    return [header, separator, ...rows].join('\n');
}

