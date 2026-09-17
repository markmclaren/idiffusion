import type {
    CloseableEventEmitter,
    DiffusionLockfile,
    Credentials,
    EnvVarEntry,
} from '../types';
import { Backend } from './Backend';
import { SshRemoteOps } from '../ops/SshRemoteOps';
import path from 'path';
import fs from 'fs';
import { isLocalPortInUse } from '../local-ops';

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export class IsambardDiffusionBackend extends Backend {
    ops: SshRemoteOps;
    envs: EnvVarEntry[];
    bootstrapped = false;
    remoteHome?: string;

    constructor(creds: Credentials) {
        super(creds);
        this.ops = new SshRemoteOps(creds);
        this.envs = [
            { key: 'IDIFFUSION_PROJECTDIR', value: creds.projectDir },
        ];
        if (creds.hfToken) {
            this.envs.push({ key: 'HF_TOKEN', value: creds.hfToken });
        }
    }

    async bootstrap(): Promise<void> {
        await this.ops.checkSSH();
        if (this.bootstrapped) return;

        const currentDir = import.meta.dir;
        const enginePath = path.resolve(currentDir, '../engine');
        const remoteEngine = await this.getRemoteEngine();

        await this.ops.runRemote(`mkdir -p "${remoteEngine}"`, {
            env: this.envs,
            silent: true,
        });

        await this.ops.copyDirectory(
            `${enginePath}/`,
            `${remoteEngine}/`,
            'up',
        );

        // Make scripts executable
        await this.ops.runRemote(`chmod +x "${remoteEngine}"/*.sh "${remoteEngine}"/lib/*.sh`, {
            env: this.envs,
            silent: true,
        });

        this.bootstrapped = true;
    }

    async setup(force?: boolean): Promise<void> {
        await this.bootstrap();
        const remoteEngine = await this.getRemoteEngine();

        console.log('Setting up idiffusion environment on Isambard AI...');
        const { stdout, exitCode } = await this.ops.runRemote(
            `${remoteEngine}/idiffusion-setup.sh${force ? ' -f' : ''}`,
            { env: this.envs, silent: false },
        );

        if (exitCode !== 0) {
            throw new Error(`setup request failed (exit ${exitCode}): ${stdout}`);
        }
    }

    async connect(
        job: string,
        localPort: number,
    ): Promise<CloseableEventEmitter> {
        await this.bootstrap();

        if (await isLocalPortInUse(localPort)) {
            throw new Error(`Local port ${localPort} is already in use.`);
        }

        let jobStatus: DiffusionLockfile;

        while (true) {
            jobStatus = await this.getJobStatus(job);
            const s = jobStatus.status;
            if (s === 'failed' || s === 'stopped' || s === 'cancel') {
                throw new Error(
                    `Could not connect to job '${job}', which is in state '${s}'${jobStatus.reason ? ` (${jobStatus.reason})` : ''}.`,
                );
            }
            if (s === 'running') break;
            if (s === 'pending') console.log(`Waiting for SLURM allocation for '${job}'...`);
            if (s === 'initialising') console.log(`Diffusers server is initialising & loading model weights...`);
            await sleep(5000);
        }

        if (jobStatus.computeHostname && jobStatus.serverPort) {
            return this.ops.spawnTunnel(
                localPort,
                jobStatus.computeHostname,
                jobStatus.serverPort,
            );
        } else {
            throw new Error(
                `Could not connect to '${job}': missing computeHostname or serverPort in status.json`,
            );
        }
    }

    async requestStart(
        job: string,
        maxTime: string = '08:00:00',
        batch: boolean = false,
        config?: string,
        model?: string,
        engine?: string,
    ): Promise<void> {
        await this.bootstrap();
        const remoteEngine = await this.getRemoteEngine();

        if (config) {
            if (fs.existsSync(config)) {
                const remoteConfig = `${this.creds.projectDir}/idiffusion/jobs/${job}/diffusion.yaml`;
                console.log(`Uploading config ${config} to ${remoteConfig}...`);
                await this.ops.copyFile(config, remoteConfig);
            } else {
                throw new Error(`No configuration file found at: ${config}`);
            }
        }

        const modelArg = model ? ` -m "${model}"` : '';
        const engineArg = engine ? ` -e "${engine}"` : '';
        const { stdout, exitCode } = await this.ops.runRemote(
            `${remoteEngine}/idiffusion-serve.sh -j "${job}" -t "${maxTime}"${batch ? ' -b' : ''}${modelArg}${engineArg}`,
            { env: this.envs, silent: false },
        );

        if (exitCode !== 0) {
            throw new Error(
                `Startup request failed for job '${job}' (exit ${exitCode}): ${stdout}`,
            );
        }
    }

    async getAllJobStatus(): Promise<DiffusionLockfile[]> {
        await this.bootstrap();
        const remoteEngine = await this.getRemoteEngine();

        const { stdout, exitCode } = await this.ops.runRemote(
            `${remoteEngine}/idiffusion-status.sh`,
            { env: this.envs, silent: true },
        );

        if (exitCode !== 0) {
            throw new Error(
                `Could not access idiffusion job status (exit ${exitCode}): ${stdout}`,
            );
        }

        let jobs: DiffusionLockfile[] = [];
        try {
            const parsed = JSON.parse(stdout || '[]');
            if (Array.isArray(parsed)) {
                jobs = parsed
                    .map((j) => this.parseLockfile(JSON.stringify(j)))
                    .filter((j): j is DiffusionLockfile => j !== null);
            }
        } catch {
            // ignore parse errors
        }

        return jobs;
    }

    async requestCancel(job: string, force: boolean): Promise<void> {
        await this.bootstrap();
        const remoteEngine = await this.getRemoteEngine();

        const { stdout, exitCode } = await this.ops.runRemote(
            `${remoteEngine}/idiffusion-cancel.sh -j "${job}"${force ? ' -f' : ''}`,
            { env: this.envs, silent: false },
        );

        if (exitCode !== 0) {
            throw new Error(
                `Cancel request failed for job '${job}' (exit ${exitCode}): ${stdout}`,
            );
        }
    }

    async watchLog(job: string): Promise<CloseableEventEmitter> {
        await this.bootstrap();
        const logPath = `${this.creds.projectDir}/idiffusion/jobs/${job}/server.log`;
        return this.ops.runRemoteSync(
            `mkdir -p "$(dirname "${logPath}")" && touch "${logPath}" && tail -n 100 -f "${logPath}"`,
            this.envs,
        );
    }

    async clearLog(job: string): Promise<void> {
        await this.bootstrap();
        const logPath = `${this.creds.projectDir}/idiffusion/jobs/${job}/server.log`;
        await this.ops.runRemote(`mkdir -p "$(dirname "${logPath}")" && > "${logPath}"`, {
            env: this.envs,
            silent: true,
        });
    }

    async clearAllLogs(): Promise<void> {
        await this.bootstrap();
        const jobsDir = `${this.creds.projectDir}/idiffusion/jobs`;
        await this.ops.runRemote(`find "${jobsDir}" -name "server.log" -exec truncate -s 0 {} + 2>/dev/null || true`, {
            env: this.envs,
            silent: true,
        });
    }

    private async getRemoteEngine(): Promise<string> {
        const home = await this.getRemoteHome();
        return `${home}/.local/share/idiffusion/engine`;
    }

    private async getRemoteHome(): Promise<string> {
        if (!this.remoteHome) {
            const { stdout } = await this.ops.runRemote('echo $HOME', {
                env: this.envs,
                silent: true,
            });
            this.remoteHome = stdout.trim();
        }
        return this.remoteHome;
    }
}
