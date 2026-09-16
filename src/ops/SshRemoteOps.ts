import type {
    Credentials,
    RunRemoteOptions,
    RunRemoteResult,
    CloseableEventEmitter,
    EnvVarEntry,
} from '../types';
import { RemoteOps } from './RemoteOps';
import { spawn } from 'child_process';
import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import net from 'net';

const SSH_MUX_OPTS = [
    '-o',
    'ControlMaster=auto',
    '-o',
    'ControlPersist=600',
    '-o',
    'ControlPath=/tmp/idiffusion-ssh-%r@%h:%p',
    '-o',
    'ServerAliveInterval=15',
    '-o',
    'ServerAliveCountMax=3',
] as const;

export class SshRemoteOps extends RemoteOps {
    private config: Credentials;
    private lastCheckTime = 0;
    private readonly CACHE_TTL_MS = 5000;

    constructor(config: Credentials) {
        super();
        this.config = config;
    }

    async runRemote(
        command: string,
        options: RunRemoteOptions = { env: [], silent: true },
    ): Promise<RunRemoteResult> {
        return new Promise((resolve, reject) => {
            const target = `${this.config.username}@${this.config.loginHost}`;
            const fullCommand = this.makeFullCommand(command, options.env || []);

            const proc = spawn(
                'ssh',
                [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, fullCommand],
                {
                    stdio: ['ignore', 'pipe', 'inherit'],
                },
            );

            let stdout = '';
            let lineBuffer = '';

            proc.stdout?.on('data', (chunk: Buffer) => {
                const text = chunk.toString();
                stdout += text;

                if (options.silent) return;

                lineBuffer += text;
                const lines = lineBuffer.split('\n');
                lineBuffer = lines.pop() ?? '';
                for (const line of lines) console.log(line);
            });

            proc.on('error', reject);
            proc.on('close', (code) => {
                if (!options.silent && lineBuffer.length > 0) {
                    console.log(lineBuffer);
                }
                resolve({ exitCode: code ?? 0, stdout: stdout.trim() });
            });
        });
    }

    async copyFile(localPath: string, remotePath: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const target = `${this.config.username}@${this.config.loginHost}`;
            const remoteDir = path.dirname(remotePath);
            const remoteCommand = `umask 002 && mkdir -p "${remoteDir}" && cat > "${remotePath}"`;
            const proc = spawn(
                'ssh',
                [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, remoteCommand],
                { stdio: ['pipe', 'inherit', 'inherit'] },
            );
            fs.createReadStream(localPath).pipe(proc.stdin);
            proc.on('error', reject);
            proc.on('close', (code) => {
                if (code === 0) resolve();
                else reject(new Error(`copyFile exited with code ${code}`));
            });
        });
    }

    async copyDirectory(
        localPath: string,
        remotePath: string,
        direction: 'up' | 'down',
    ): Promise<void> {
        return new Promise((resolve, reject) => {
            const target = `${this.config.username}@${this.config.loginHost}`;
            const sshCmd = `ssh ${SSH_MUX_OPTS.join(' ')} -o BatchMode=yes`;

            let src: string;
            let dst: string;

            if (direction === 'up') {
                src = localPath.endsWith('/') ? localPath : `${localPath}/`;
                dst = `${target}:${remotePath}`;
            } else {
                src = `${target}:${remotePath.endsWith('/') ? remotePath : `${remotePath}/`}`;
                dst = localPath;
            }

            const proc = spawn(
                'rsync',
                ['-az', '--chmod=D775,F664', '-e', sshCmd, src, dst],
                { stdio: 'inherit' },
            );

            proc.on('error', reject);
            proc.on('close', (code) => {
                if (code === 0) resolve();
                else reject(new Error(`rsync exited with code ${code}`));
            });
        });
    }

    runRemoteSync(command: string, env: EnvVarEntry[]): CloseableEventEmitter {
        const target = `${this.config.username}@${this.config.loginHost}`;
        const fullCommand = this.makeFullCommand(command, env);

        const proc = spawn(
            'ssh',
            [
                ...SSH_MUX_OPTS,
                '-tt',
                '-o',
                'BatchMode=yes',
                target,
                fullCommand,
            ],
            {
                stdio: ['ignore', 'inherit', 'inherit'],
                detached: false,
            },
        );

        return Object.assign(proc, {
            isAlive: async () => proc.exitCode === null && !proc.killed,
            close: () =>
                new Promise<void>((resolve) => {
                    if (proc.exitCode !== null) {
                        resolve();
                        return;
                    }
                    proc.once('close', () => resolve());
                    proc.kill();
                }),
        });
    }

    spawnTunnel(
        localPort: number,
        remoteHost: string,
        remotePort: number,
    ): CloseableEventEmitter {
        const target = `${this.config.username}@${this.config.loginHost}`;
        const forwardSpec = `${localPort}:${remoteHost}:${remotePort}`;
        const emitter = new EventEmitter() as unknown as CloseableEventEmitter;

        let closed = false;
        let pollTimer: ReturnType<typeof setInterval> | undefined;

        const runControl = (args: string[]) =>
            new Promise<number>((resolve) => {
                const p = spawn(
                    'ssh',
                    [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', ...args, target],
                    { stdio: 'ignore' },
                );
                p.on('close', (code) => resolve(code ?? 1));
                p.on('error', () => resolve(1));
            });

        const registered = runControl(['-O', 'forward', '-L', forwardSpec]);

        const checkPortListening = () =>
            new Promise<boolean>((resolve) => {
                const sock = net.createConnection({
                    port: localPort,
                    host: '127.0.0.1',
                });
                sock.once('connect', () => {
                    sock.destroy();
                    resolve(true);
                });
                sock.once('error', () => resolve(false));
                sock.setTimeout(2000, () => {
                    sock.destroy();
                    resolve(false);
                });
            });

        const markClosed = () => {
            if (closed) return;
            closed = true;
            if (pollTimer) clearInterval(pollTimer);
            emitter.emit('close');
        };

        emitter.isAlive = async () => {
            if ((await registered) !== 0) return false;
            return checkPortListening();
        };

        emitter.close = async () => {
            if (closed) return;
            await registered;
            await runControl(['-O', 'cancel', '-L', forwardSpec]);
            markClosed();
        };

        registered.then((code) => {
            if (code !== 0) {
                emitter.emit(
                    'error',
                    new Error(`Failed to register SSH forward (exit ${code})`),
                );
                markClosed();
                return;
            }

            pollTimer = setInterval(async () => {
                if (closed) return;
                if (!(await checkPortListening())) markClosed();
            }, 5000);
        });

        return emitter;
    }

    async checkSSH(): Promise<boolean> {
        const now = Date.now();
        const target = `${this.config.username}@${this.config.loginHost}`;

        if (now - this.lastCheckTime < this.CACHE_TTL_MS) {
            return true;
        }

        const isMuxAlive = await new Promise<boolean>((resolve) => {
            const checkProc = spawn(
                'ssh',
                [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', '-O', 'check', target],
                { stdio: 'ignore', detached: false },
            );
            checkProc.on('close', (code) => resolve(code === 0));
            checkProc.on('error', () => resolve(false));
        });

        if (isMuxAlive) {
            console.log('✓ SSH multiplexer is alive and healthy.');
            this.lastCheckTime = Date.now();
            return true;
        }

        console.log('Initialising master SSH connection...');

        return new Promise<boolean>((resolve, reject) => {
            const connectArgs = [
                ...SSH_MUX_OPTS,
                '-o',
                'BatchMode=yes',
                '-o',
                'ConnectTimeout=10',
                '-N',
                '-f',
                target,
            ];

            const errorChunks: string[] = [];
            const connProc = spawn('ssh', connectArgs, {
                stdio: ['ignore', 'ignore', 'pipe'],
                detached: false,
            });

            connProc.stderr?.on('data', (chunk) => {
                errorChunks.push(chunk.toString());
            });

            connProc.on('close', (code) => {
                if (code === 0) {
                    console.log('✓ Successfully established new SSH master connection.');
                    this.lastCheckTime = Date.now();
                    resolve(true);
                    return;
                }

                const rawError = errorChunks.join('').trim();
                console.error('\n❌ CRITICAL: Failed to establish SSH connection to the login node.');
                if (rawError) {
                    console.error(`System Error Message:\n--> ${rawError}`);
                }
                reject(new Error(`Failed to connect to ${this.config.loginHost} (SSH exit code ${code})`));
            });

            connProc.on('error', reject);
        });
    }
}
