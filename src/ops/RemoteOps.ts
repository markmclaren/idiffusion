import type {
    RunRemoteOptions,
    RunRemoteResult,
    CloseableEventEmitter,
    EnvVarEntry,
} from '../types';

/**
 * Abstract interface for executing remote operations on the HPC login node.
 */
export abstract class RemoteOps {
    abstract runRemote(
        command: string,
        options?: RunRemoteOptions,
    ): Promise<RunRemoteResult>;

    abstract copyFile(localPath: string, remotePath: string): Promise<void>;

    abstract copyDirectory(
        localPath: string,
        remotePath: string,
        direction: 'up' | 'down',
    ): Promise<void>;

    abstract runRemoteSync(
        command: string,
        env: EnvVarEntry[],
    ): CloseableEventEmitter;

    abstract spawnTunnel(
        localPort: number,
        remoteHost: string,
        remotePort: number,
    ): CloseableEventEmitter;

    abstract checkSSH(): Promise<boolean>;

    protected makeFullCommand(command: string, env: EnvVarEntry[]): string {
        const envPrefix =
            env.map((v) => `${v.key}="${v.value}"`).join(' ') + ' ';
        const fullCommand = (envPrefix + command).trim();
        return fullCommand;
    }
}
