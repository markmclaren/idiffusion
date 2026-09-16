import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';
import type { Credentials } from './types';

const IDIFFUSION_CONFIG_DIR = join(homedir(), '.config', 'idiffusion');
const IDIFFUSION_CONFIG_PATH = join(IDIFFUSION_CONFIG_DIR, 'config.json');
const IVLLM_CONFIG_PATH = join(homedir(), '.config', 'ivllm', 'config.json');

const DEFAULTS: Credentials = {
    loginHost: '',
    username: '',
    projectDir: '',
    defaultLocalPort: 8000,
};

/**
 * Loads credentials from ~/.config/idiffusion/config.json.
 * If not present, attempts to seed from ~/.config/ivllm/config.json.
 */
export function loadCredentials(): Credentials {
    if (existsSync(IDIFFUSION_CONFIG_PATH)) {
        try {
            const raw = readFileSync(IDIFFUSION_CONFIG_PATH, 'utf-8');
            return { ...DEFAULTS, ...JSON.parse(raw) } as Credentials;
        } catch {
            return { ...DEFAULTS };
        }
    }

    // Fallback: inherit from existing ivllm config if available
    if (existsSync(IVLLM_CONFIG_PATH)) {
        try {
            const raw = readFileSync(IVLLM_CONFIG_PATH, 'utf-8');
            const ivllmCreds = JSON.parse(raw);
            const inherited: Credentials = {
                loginHost: ivllmCreds.loginHost || '',
                username: ivllmCreds.username || '',
                projectDir: ivllmCreds.projectDir || '',
                defaultLocalPort: 8000,
                hfToken: ivllmCreds.hfToken,
            };
            return inherited;
        } catch {
            // fall through to defaults
        }
    }

    return { ...DEFAULTS };
}

/**
 * Saves credentials to ~/.config/idiffusion/config.json.
 */
export function saveCredentials(config: Credentials): void {
    if (!existsSync(IDIFFUSION_CONFIG_DIR)) {
        mkdirSync(IDIFFUSION_CONFIG_DIR, { recursive: true });
    }
    writeFileSync(
        IDIFFUSION_CONFIG_PATH,
        JSON.stringify(config, null, 2) + '\n',
        'utf-8',
    );
}

/**
 * Validates that essential connection parameters are configured.
 */
export function assertConfigured(config: Credentials): void {
    if (!config.loginHost) {
        throw new Error(
            'loginHost not configured. Run: idiffusion config --login-host <host>',
        );
    }
    if (!config.username) {
        throw new Error(
            'username not configured. Run: idiffusion config --username <user>',
        );
    }
    if (!config.projectDir) {
        throw new Error(
            'projectDir not configured. Run: idiffusion config --project-dir <project>',
        );
    }
}
