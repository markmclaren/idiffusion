import { existsSync, mkdirSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';

export interface LocalSession {
    jobName: string;
    localPort: number;
    pid: number;
    engine?: string;
    slurmJobId?: string;
    model?: string;
    startTime: string;
}

const SESSIONS_DIR = join(homedir(), '.config', 'idiffusion', 'sessions');

function ensureSessionsDir(): void {
    if (!existsSync(SESSIONS_DIR)) {
        mkdirSync(SESSIONS_DIR, { recursive: true });
    }
}

function getSessionFilePath(jobName: string): string {
    const safeName = jobName.replace(/[^a-zA-Z0-9_-]/g, '_');
    return join(SESSIONS_DIR, `${safeName}.json`);
}

function isProcessAlive(pid: number): boolean {
    try {
        return process.kill(pid, 0);
    } catch (err: any) {
        return err.code === 'EPERM';
    }
}

export function saveSession(session: LocalSession): void {
    ensureSessionsDir();
    const filePath = getSessionFilePath(session.jobName);
    writeFileSync(filePath, JSON.stringify(session, null, 2) + '\n', 'utf-8');
}

export function removeSession(jobName: string): void {
    const filePath = getSessionFilePath(jobName);
    if (existsSync(filePath)) {
        try {
            unlinkSync(filePath);
        } catch {
            // ignore cleanup error
        }
    }
}

export function getActiveSessions(): LocalSession[] {
    ensureSessionsDir();
    const active: LocalSession[] = [];
    try {
        const files = readdirSync(SESSIONS_DIR);
        for (const file of files) {
            if (!file.endsWith('.json')) continue;
            const fullPath = join(SESSIONS_DIR, file);
            try {
                const content = readFileSync(fullPath, 'utf-8');
                const session = JSON.parse(content) as LocalSession;
                if (session && session.pid && isProcessAlive(session.pid)) {
                    active.push(session);
                } else {
                    // Stale file
                    unlinkSync(fullPath);
                }
            } catch {
                // ignore corrupt file
            }
        }
    } catch {
        // ignore dir read error
    }
    return active;
}

export function getSessionByJob(jobName: string): LocalSession | null {
    const sessions = getActiveSessions();
    return sessions.find((s) => s.jobName === jobName) || null;
}

export function getSessionByPort(port: number): LocalSession | null {
    const sessions = getActiveSessions();
    return sessions.find((s) => s.localPort === port) || null;
}

export async function stopLocalSession(session: LocalSession): Promise<boolean> {
    if (session.pid && isProcessAlive(session.pid)) {
        try {
            process.kill(session.pid, 'SIGTERM');
            let checks = 0;
            while (isProcessAlive(session.pid) && checks < 10) {
                await new Promise((res) => setTimeout(res, 200));
                checks++;
            }
            if (isProcessAlive(session.pid)) {
                process.kill(session.pid, 'SIGKILL');
            }
        } catch {
            // ignore kill error
        }
    }
    removeSession(session.jobName);
    return true;
}
