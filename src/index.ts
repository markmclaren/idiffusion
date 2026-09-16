#!/usr/bin/env bun
import { program } from 'commander';
import { getBackend } from './backends/backend-factory';
import {
    loadCredentials,
    assertConfigured,
    saveCredentials,
} from './config';
import { formatJobTable } from './utils';
import { isLocalPortInUse, isHealthy, generateImage } from './local-ops';
import type { CloseableEventEmitter } from './types';
import { execFile } from 'child_process';

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function main() {
    program
        .name('idiffusion')
        .version('1.0.0')
        .description('Manage Diffusers image generation jobs on Isambard AI HPC');

    // CONFIG COMMAND
    program
        .command('config')
        .description('Configure user credentials and connection settings for HPC')
        .option('--login-host <host>', 'SSH login host (e.g. b6ai.aip2.isambard)')
        .option('--username <user>', 'HPC username (e.g. ismjml.b6ai)')
        .option('--project-dir <path>', 'Shared project directory (e.g. /projects/b6ai)')
        .option('--hf-token <token>', 'HuggingFace access token')
        .option('--default-port <port>', 'Default local port for API tunnel (e.g. 8000)')
        .action(cmdConfig);

    // SETUP COMMAND
    program
        .command('setup')
        .description('Install Python Diffusers environment on HPC compute node (one-off per project)')
        .option('--force', 'Force reinstallation even if already present', false)
        .action(cmdSetup);

    // COMFY COMMAND
    program
        .command('comfy')
        .description('Start or connect to an interactive ComfyUI session on the cluster')
        .argument('[jobName]', 'Short name for the ComfyUI job', 'comfy')
        .option('--local-port <port>', 'Local port for ComfyUI web interface', '8188')
        .option('--batch', 'Submit to batch partition instead of interactive', false)
        .option('--time <duration>', 'SLURM time limit (e.g. 08:00:00)', '08:00:00')
        .option('--no-open', 'Do not automatically open browser upon connection', false)
        .action(async (jobName, opts) => {
            await cmdConnect(jobName || 'comfy', {
                ...opts,
                comfy: true,
                localPort: opts.localPort || '8188',
            });
        });

    // CONNECT COMMAND
    program
        .command('connect')
        .description('Start or connect to an image generation session on the cluster')
        .argument('<jobName>', 'Short name for the job (e.g. flux-quick, sd35, etc.)')
        .option('--config <configFile>', 'Path to YAML configuration file')
        .option('--model <modelId>', 'HuggingFace model ID (e.g. black-forest-labs/FLUX.1-schnell)')
        .option('--comfy', 'Launch ComfyUI instead of headless Diffusers', false)
        .option('--local-port <port>', 'Local port to expose the API on')
        .option('--batch', 'Submit to batch partition instead of interactive', false)
        .option('--time <duration>', 'SLURM time limit (e.g. 08:00:00)', '08:00:00')
        .option('--no-open', 'Do not automatically open browser (when using --comfy)', false)
        .action(cmdConnect);

    // STATUS COMMAND
    program
        .command('status')
        .description('Show status of image generation jobs on the HPC')
        .argument('[jobName]', 'Show specific job status')
        .action(cmdStatus);

    // CANCEL COMMAND
    program
        .command('cancel')
        .description('Cancel a running image generation job')
        .argument('<jobName>', 'Short name of the job')
        .option('--force', 'Force cancellation using scancel directly', false)
        .action(cmdCancel);

    // LOG COMMAND
    program
        .command('log')
        .description('Tail remote log for an image generation job')
        .argument('<jobName>', 'Short name of the job')
        .action(cmdLog);

    // GENERATE COMMAND
    program
        .command('generate')
        .description('Generate an image via the active local tunnel endpoint')
        .requiredOption('-p, --prompt <text>', 'Text prompt describing the desired image')
        .option('-o, --output <path>', 'Output file path', 'output.png')
        .option('-s, --size <WxH>', 'Image dimensions (e.g. 1024x1024, 768x768)', '1024x1024')
        .option('--steps <number>', 'Number of inference steps (e.g. 4 for FLUX schnell, 28 for dev)', (val) => parseInt(val, 10))
        .option('--guidance-scale <number>', 'Guidance scale (CFG)', (val) => parseFloat(val))
        .option('--seed <number>', 'Random seed for deterministic generation', (val) => parseInt(val, 10))
        .option('--negative-prompt <text>', 'Negative prompt')
        .option('--port <number>', 'Local port of the running tunnel', (val) => parseInt(val, 10))
        .option('--model <modelId>', 'Model ID to pass in request')
        .action(cmdGenerate);

    await program.parseAsync(process.argv);
}

// =================================
// COMMAND HANDLERS
// =================================

async function cmdConfig(options: {
    loginHost?: string;
    username?: string;
    projectDir?: string;
    hfToken?: string;
    defaultPort?: string;
}): Promise<void> {
    const config = loadCredentials();
    let changed = false;

    if (options.loginHost) {
        config.loginHost = options.loginHost;
        changed = true;
    }
    if (options.username) {
        config.username = options.username;
        changed = true;
    }
    if (options.projectDir) {
        config.projectDir = options.projectDir;
        changed = true;
    }
    if (options.hfToken) {
        config.hfToken = options.hfToken;
        changed = true;
    }
    if (options.defaultPort) {
        config.defaultLocalPort = parseInt(options.defaultPort, 10);
        changed = true;
    }

    if (changed) {
        saveCredentials(config);
        console.log('✓ Configuration updated.');
    } else {
        console.log(JSON.stringify(config, null, 2));
    }
}

async function cmdSetup(options: { force: boolean }): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);
    await backend.setup(options.force);
}

async function cmdConnect(
    jobName: string,
    options: {
        config?: string;
        model?: string;
        comfy?: boolean;
        localPort?: string;
        batch: boolean;
        time: string;
        open?: boolean;
    },
): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);

    const isComfy = options.comfy === true;
    const defaultPort = isComfy ? 8188 : (config.defaultLocalPort || 8000);
    const localPort = options.localPort ? parseInt(options.localPort, 10) : defaultPort;

    const portUsage = await isLocalPortInUse(localPort);
    if (portUsage) {
        console.error(`Port ${localPort} is already in use by process ${portUsage.process} (PID ${portUsage.pid}).`);
        console.error(`Specify a different port with --local-port <port> or terminate the existing process.`);
        process.exit(1);
    }

    const backend = getBackend(config);

    let logWatcher: CloseableEventEmitter | null = null;
    let tunnel: CloseableEventEmitter | null = null;

    const cleanupAndExit = async () => {
        console.log('\n\n[Ctrl+C] Disconnecting local session...');
        try {
            if (logWatcher) await logWatcher.close();
            if (tunnel) await tunnel.close();
        } catch (err) {
            console.error('Error during cleanup:', err);
        } finally {
            console.log('Done.');
            process.exit(0);
        }
    };

    process.once('SIGINT', cleanupAndExit);
    process.once('SIGTERM', cleanupAndExit);

    try {
        if (await backend.isStartable(jobName)) {
            console.log(`[connect] Job '${jobName}' is not currently active. Requesting start...`);
            await backend.requestStart(
                jobName,
                options.time,
                options.batch,
                options.config,
                options.model,
                isComfy ? 'comfyui' : undefined,
            );
        }

        if (await backend.isStarting(jobName)) {
            console.log(`[connect] Job '${jobName}' is starting. Attaching log stream...`);
            logWatcher = await backend.watchLog(jobName);

            while (await backend.isStarting(jobName)) {
                await sleep(2000);
            }

            await logWatcher.close();
            logWatcher = null;
        }

        const status = await backend.getStatusFlag(jobName);
        if (status !== 'running') {
            const lockfile = await backend.getJobStatus(jobName).catch(() => null);
            throw new Error(
                `Job failed to reach running state. Current status: [${status}]${lockfile?.reason ? ` - ${lockfile.reason}` : ''}.`,
            );
        }

        console.log(`[connect] Job '${jobName}' is running! Establishing SSH tunnel to compute node...`);
        tunnel = await backend.connect(jobName, localPort);

        // Wait a moment for port to be bound
        await sleep(1500);

        const lockfile = await backend.getJobStatus(jobName).catch(() => null);
        const slurmId = lockfile?.slurmJobId || '-';
        const engine = lockfile?.engine || (isComfy ? 'comfyui' : 'diffusers');

        if (engine === 'comfyui') {
            console.log(`
========================================================================
🎨 ComfyUI Web Interface Ready!
========================================================================
  Web UI:         http://localhost:${localPort}/
  WebSocket API:  ws://localhost:${localPort}/ws
  Job Name:       ${jobName}
  SLURM Job ID:   ${slurmId}
  Shared Models:  Auto-mounted from /projects/b6ai/model

[Press Ctrl+C at any time to disconnect the tunnel. The HPC session will remain active.]
========================================================================
`);
            if (process.platform === 'darwin' && options.open !== false) {
                console.log(`Opening http://localhost:${localPort}/ in your browser...`);
                execFile('open', [`http://localhost:${localPort}/`]);
            }
        } else {
            const modelName = lockfile?.model || options.model || 'Diffusers Model';
            console.log(`
========================================================================
🎨 idiffusion Server Connected
========================================================================
  Endpoint:       http://localhost:${localPort}/v1
  Images API:     http://localhost:${localPort}/v1/images/generations
  Health:         http://localhost:${localPort}/health
  Model:          ${modelName}
  Job Name:       ${jobName}
  SLURM Job ID:   ${slurmId}

Quick generation with idiffusion:
  idiffusion generate --prompt "A serene lake in the mountains at sunrise" --port ${localPort}

Example OpenAI Python SDK client:
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:${localPort}/v1", api_key="placeholder")
  res = client.images.generate(model="${modelName}", prompt="...", response_format="b64_json")

Example curl request:
  curl -X POST http://localhost:${localPort}/v1/images/generations \\
       -H "Content-Type: application/json" \\
       -d '{"prompt": "A futuristic city in cyberpunk style", "size": "1024x1024"}' \\
       | jq -r '.data[0].b64_json' | base64 --decode > output.png

[Press Ctrl+C at any time to disconnect the tunnel. The HPC job will remain alive.]
========================================================================
`);
        }

        // Keep tunnel alive until signal or tunnel close
        while (await tunnel.isAlive()) {
            await sleep(5000);
        }

        console.log('[connect] Tunnel closed.');
    } catch (err: any) {
        console.error(`\n[connect] ERROR: ${err.message || err}`);
        await cleanupAndExit();
    }
}

async function cmdStatus(jobName?: string): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);

    if (jobName) {
        try {
            const status = await backend.getJobStatus(jobName);
            console.log(formatJobTable([status]));
        } catch (err: any) {
            console.error(err.message);
        }
    } else {
        const statuses = await backend.getAllJobStatus();
        console.log(formatJobTable(statuses));
    }
}

async function cmdCancel(
    jobName: string,
    options: { force: boolean },
): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);
    console.log(`Cancelling job '${jobName}'${options.force ? ' (force)' : ''}...`);
    await backend.requestCancel(jobName, options.force);
    console.log(`✓ Job '${jobName}' cancelled.`);
}

async function cmdLog(jobName: string): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);

    console.log(`Tailing logs for job '${jobName}' (Ctrl+C to stop)...`);
    const watcher = await backend.watchLog(jobName);

    process.once('SIGINT', async () => {
        await watcher.close();
        process.exit(0);
    });

    while (await watcher.isAlive()) {
        await sleep(1000);
    }
}

async function cmdGenerate(options: {
    prompt: string;
    output: string;
    size: string;
    steps?: number;
    guidanceScale?: number;
    seed?: number;
    negativePrompt?: string;
    port?: number;
    model?: string;
}): Promise<void> {
    const config = loadCredentials();
    const port = options.port || config.defaultLocalPort || 8000;

    console.log(`Checking connection to http://localhost:${port}...`);
    const healthy = await isHealthy(port, 3000);
    if (!healthy) {
        console.error(`Cannot reach idiffusion server at http://localhost:${port}.`);
        console.error(`Ensure an active session is running with: idiffusion connect <jobName> --local-port ${port}`);
        process.exit(1);
    }

    console.log(`🎨 Generating image for prompt: "${options.prompt}"`);
    if (options.steps) console.log(`   Steps: ${options.steps}`);
    if (options.size) console.log(`   Size: ${options.size}`);
    if (options.seed !== undefined) console.log(`   Seed: ${options.seed}`);

    try {
        const result = await generateImage({
            port,
            prompt: options.prompt,
            model: options.model,
            size: options.size,
            steps: options.steps,
            guidanceScale: options.guidanceScale,
            seed: options.seed,
            negativePrompt: options.negativePrompt,
            outputPath: options.output,
        });

        console.log(`✓ Image generated successfully in ${(result.durationMs / 1000).toFixed(1)}s!`);
        console.log(`📁 Saved to: ${result.outputPath}`);
    } catch (err: any) {
        console.error(`\nGeneration error: ${err.message}`);
        process.exit(1);
    }
}

main().catch((err) => {
    console.error('Fatal error:', err);
    process.exit(1);
});
