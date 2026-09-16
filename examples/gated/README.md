# Gated Model Configs

The configs in this directory use **gated HuggingFace models** — models that require you to accept a licence agreement before the weights can be downloaded.

## Before using these configs

1. **Accept the model licence** on HuggingFace (links below)
2. **Set your HuggingFace token** on the compute cluster so the server can authenticate:

   ```bash
   # Option A — interactive login (stores token in ~/.cache/huggingface/token)
   huggingface-cli login

   # Option B — environment variable (add to ~/.bashrc or your job script)
   export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```

3. **Retry** your `idiffusion connect` command.

---

## Models

| Config | Model | Licence |
|--------|-------|---------|
| `flux-1-dev.yaml` | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | FLUX.1-dev Non-Commercial |
| `sd-3.5-large.yaml` | [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | Stability AI Community |

> **Note:** If you see a `GatedRepoError: 403 Client Error` in the server log, it almost always means either the token is missing on the cluster or you haven't accepted the licence yet.

---

## Open-weight alternatives

If you don't want to deal with gated access, the following configs in `examples/` work without any token:

| Config | Model |
|--------|-------|
| `flux-1-schnell.yaml` | FLUX.1-schnell (Apache 2.0) |
| `sdxl.yaml` | Stable Diffusion XL Base 1.0 (CreativeML RAIL) |
