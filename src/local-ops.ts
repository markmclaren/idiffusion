import { execFile } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';

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
 * Generate an image using the local tunnel endpoint (POST /v1/images/generations or /v1/images/edits).
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
    imagePath?: string;
    maskPath?: string;
    strength?: number;
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
    if (options.strength !== undefined) payload.strength = options.strength;

    if (options.imagePath) {
        if (!existsSync(options.imagePath)) {
            throw new Error(`Input image file not found: ${options.imagePath}`);
        }
        const imgBuffer = readFileSync(options.imagePath);
        payload.image = imgBuffer.toString('base64');
    }

    if (options.maskPath) {
        if (!existsSync(options.maskPath)) {
            throw new Error(`Mask image file not found: ${options.maskPath}`);
        }
        const maskBuffer = readFileSync(options.maskPath);
        payload.mask = maskBuffer.toString('base64');
    }

    const endpoint = (options.imagePath || options.maskPath) ? '/v1/images/edits' : '/v1/images/generations';

    const res = await fetch(`http://localhost:${options.port}${endpoint}`, {
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
