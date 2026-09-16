import { execFile } from 'child_process';
import { writeFileSync } from 'fs';

/**
 * Detect whether a TCP listener is bound on `localhost` at the given port.
 */
export async function isLocalPortInUse(
    port: number,
): Promise<{ pid: string; process: string } | null> {
    return new Promise((resolve) => {
        execFile('lsof', ['-ti', `:${port}`, '-sTCP:LISTEN'], (err, stdout) => {
            if (err || !stdout.trim()) {
                resolve(null);
                return;
            }
            const pid = stdout.trim().split('\n')[0] as string;
            execFile('ps', ['-p', pid, '-o', 'comm='], (_err2, psOut) => {
                const process = psOut?.trim() || 'unknown';
                resolve({ pid, process });
            });
        });
    });
}

/**
 * Check health of diffusion server via GET /health.
 */
export async function isHealthy(
    localPort: number,
    timeoutMs: number = 3000,
): Promise<boolean> {
    const timeout = new Promise<boolean>((resolve) =>
        setTimeout(() => resolve(false), timeoutMs),
    );

    const networkRequest = fetch(`http://localhost:${localPort}/health`)
        .then((res) => res.ok)
        .catch(() => false);

    return Promise.race([networkRequest, timeout]);
}

/**
 * Generate an image using the local tunnel endpoint (POST /v1/images/generations).
 */
export async function generateImage(options: {
    port: number;
    prompt: string;
    model?: string;
    size?: string;
    steps?: number;
    guidanceScale?: number;
    seed?: number;
    negativePrompt?: string;
    outputPath: string;
}): Promise<{ outputPath: string; durationMs: number }> {
    const start = Date.now();
    const payload: Record<string, unknown> = {
        prompt: options.prompt,
        size: options.size || '1024x1024',
        response_format: 'b64_json',
    };
    if (options.model) payload.model = options.model;
    if (options.steps !== undefined) payload.num_inference_steps = options.steps;
    if (options.guidanceScale !== undefined) payload.guidance_scale = options.guidanceScale;
    if (options.seed !== undefined) payload.seed = options.seed;
    if (options.negativePrompt) payload.negative_prompt = options.negativePrompt;

    const res = await fetch(`http://localhost:${options.port}/v1/images/generations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        const errorText = await res.text().catch(() => res.statusText);
        throw new Error(`Generation failed (HTTP ${res.status}): ${errorText}`);
    }

    const data = await res.json() as any;
    if (!data.data || !data.data[0] || !data.data[0].b64_json) {
        throw new Error('Malformed response from image server (no b64_json found)');
    }

    const imageBuffer = Buffer.from(data.data[0].b64_json, 'base64');
    writeFileSync(options.outputPath, imageBuffer);
    const durationMs = Date.now() - start;

    return { outputPath: options.outputPath, durationMs };
}
