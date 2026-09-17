import type {
    CloseableEventEmitter,
    DiffusionLockfile,
    Credentials,
} from '../types';

export abstract class Backend {
    protected creds: Credentials;

    constructor(creds: Credentials) {
        this.creds = creds;
    }

    abstract bootstrap(): Promise<void>;

    abstract setup(force?: boolean): Promise<void>;

    abstract connect(
        job: string,
        localPort: number,
    ): Promise<CloseableEventEmitter>;

    abstract requestCancel(
        job: string,
        force: boolean,
    ): Promise<void>;

    abstract requestStart(
        job: string,
        maxTime: string,
        batch: boolean,
        config?: string,
        model?: string,
        engine?: string,
    ): Promise<void>;

    abstract getAllJobStatus(): Promise<DiffusionLockfile[]>;

    abstract watchLog(
        job: string,
    ): Promise<CloseableEventEmitter>;

    abstract clearLog(job: string): Promise<void>;

    abstract clearAllLogs(): Promise<void>;

    async getJobStatus(job: string): Promise<DiffusionLockfile> {
        const statuses = await this.getAllJobStatus();
        const matched = statuses.find((s) => s.jobName === job);
        if (!matched) {
            throw new Error(`Job status for '${job}' not found.`);
        }
        return matched;
    }

    async getStatusFlag(job: string): Promise<string> {
        try {
            const status = await this.getJobStatus(job);
            return status.status;
        } catch {
            return '';
        }
    }

    async isStarting(job: string): Promise<boolean> {
        const s = await this.getStatusFlag(job);
        return s === 'pending' || s === 'initialising';
    }

    async isStartable(job: string): Promise<boolean> {
        const s = await this.getStatusFlag(job);
        return s === '' || s === 'stopped' || s === 'failed';
    }

    async isStopped(job: string): Promise<boolean> {
        const s = await this.getStatusFlag(job);
        return s === 'stopped' || s === 'failed' || s === '';
    }

    protected parseLockfile(raw: string): DiffusionLockfile | null {
        if (!raw.trim()) return null;
        try {
            const obj = JSON.parse(raw);
            if (typeof obj.status !== 'string' || typeof obj.jobName !== 'string') {
                return null;
            }
            return obj as DiffusionLockfile;
        } catch {
            return null;
        }
    }
}
