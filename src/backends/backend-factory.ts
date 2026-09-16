import type { Credentials } from '../types';
import { Backend } from './Backend';
import { IsambardDiffusionBackend } from './IsambardDiffusionBackend';

export function getBackend(creds: Credentials): Backend {
    return new IsambardDiffusionBackend(creds);
}
