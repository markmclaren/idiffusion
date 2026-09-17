import type EventEmitter from 'events';

// =================================
// CONFIG AND CREDENTIALS
// =================================

export interface Credentials {
    loginHost: string;
    username: string;
    projectDir: string;
    defaultLocalPort?: number;
    hfToken?: string;
}

export interface EnvVarEntry {
    key: string;
    value: string;
}

// =================================
// REMOTE OPERATIONS INTERFACE
// =================================

export type RunRemoteOptions = {
    env?: EnvVarEntry[];
    silent?: boolean;
};

export type RunRemoteResult = {
    exitCode: number;
    stdout: string;
};

export interface CloseableEventEmitter extends EventEmitter {
    isAlive(): Promise<boolean>;
    close(): Promise<void>;
}

// =================================
// LOCKFILE & STATUS
// =================================

export type LockfileState =
    | 'pending'
    | 'initialising'
    | 'running'
    | 'failed'
    | 'stopped'
    | 'cancel';

export interface DiffusionLockfile {
    status: LockfileState;
    jobName: string;
    model: string;
    engine?: 'diffusers' | 'comfyui';
    serverPort: number;
    user: string;
    requestedTime: string;
    idleTimeout: number;
    slurmJobId?: string;
    computeHostname?: string;
    startTime?: string;
    stopTime?: string;
    reason?: string;
    exitCode?: number;
    resources?: string;
}

// =================================
// JOB CONFIGURATION
// =================================

export interface DiffusionConfig {
    name?: string;
    model: string;
    engine?: 'diffusers' | 'comfyui';
    torchDtype?: string;
    idleTimeout?: number;
    defaultSteps?: number;
    defaultSize?: string;
    gpus?: number;
    time?: string;
    env?: Record<string, string>;
}
