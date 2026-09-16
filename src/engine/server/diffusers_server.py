#!/usr/bin/env python3
"""
idiffusion compute-side server.
Lightweight FastAPI server exposing an OpenAI-compatible /v1/images/generations endpoint
backed by Hugging Face Diffusers on NVIDIA GPUs.
"""

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
import json
import os
import socket
import sys
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import torch

# Global state
pipeline = None
loaded_model_id = ""
last_activity_time = time.time()
idle_timeout_seconds = 30 * 60
status_file_path: Optional[str] = None
server_hostname = socket.gethostname()
server_port = 8000


def get_current_time_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_status_file(updates: dict):
    if not status_file_path or not os.path.exists(status_file_path):
        return
    try:
        with open(status_file_path, "r") as f:
            data = json.load(f)
        data.update(updates)
        with open(status_file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[server] Warning: failed to update status file: {e}", file=sys.stderr)


async def idle_checker():
    """Background task to cleanly shut down after inactivity."""
    if idle_timeout_seconds <= 0:
        return
    while True:
        await asyncio.sleep(15)
        elapsed = time.time() - last_activity_time
        if elapsed > idle_timeout_seconds:
            print(f"[server] Idle timeout reached ({elapsed:.0f}s >= {idle_timeout_seconds}s). Shutting down...")
            update_status_file({
                "status": "stopped",
                "reason": "idle timeout",
                "stopTime": get_current_time_iso(),
            })
            os._exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    checker_task = asyncio.create_task(idle_checker())
    yield
    # Shutdown
    checker_task.cancel()
    update_status_file({
        "status": "stopped",
        "reason": "normal shutdown",
        "stopTime": get_current_time_iso(),
    })


app = FastAPI(title="idiffusion Image Generation Server", lifespan=lifespan)


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None


def load_pipeline(model_id: str, dtype: str = "bfloat16"):
    global pipeline, loaded_model_id
    print(f"[server] Loading model: {model_id} with dtype {dtype}...")

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else (torch.float16 if dtype == "float16" else torch.float32)
    hf_token = os.environ.get("HF_TOKEN")

    # Import specialized pipelines if available for maximum performance
    model_lower = model_id.lower()
    try:
        if "flux" in model_lower:
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
        elif "stable-diffusion-3" in model_lower or "sd3" in model_lower:
            from diffusers import StableDiffusion3Pipeline
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
        elif "xl" in model_lower:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
        else:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
    except Exception as e:
        print(f"[server] Specialized load failed, falling back to DiffusionPipeline: {e}")
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            token=hf_token,
        )

    if torch.cuda.is_available():
        pipe.to("cuda")
        print(f"[server] Pipeline moved to CUDA (device: {torch.cuda.get_device_name(0)})")
    else:
        print("[server] CUDA not available; running on CPU")

    pipeline = pipe
    loaded_model_id = model_id
    print(f"[server] Model {model_id} loaded successfully!")


@app.get("/health")
def health_check():
    global last_activity_time
    last_activity_time = time.time()

    cuda_info = {}
    if torch.cuda.is_available():
        cuda_info = {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 2),
        }

    return {
        "status": "ok",
        "model": loaded_model_id,
        "cuda": cuda_info,
        "hostname": server_hostname,
        "port": server_port,
    }


@app.get("/v1/models")
def list_models():
    global last_activity_time
    last_activity_time = time.time()
    return {
        "object": "list",
        "data": [
            {
                "id": loaded_model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "idiffusion",
            }
        ],
    }


@app.post("/v1/images/generations")
async def generate_images(request: ImageGenerationRequest):
    global last_activity_time, pipeline
    last_activity_time = time.time()

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not ready")

    # Parse dimensions
    try:
        parts = request.size.lower().split("x")
        width = int(parts[0])
        height = int(parts[1]) if len(parts) > 1 else width
    except Exception:
        width = 1024
        height = 1024

    # Setup generator for reproducibility
    generator = None
    if request.seed is not None and torch.cuda.is_available():
        generator = torch.Generator(device="cuda").manual_seed(request.seed)

    # Prepare call arguments
    call_kwargs = {
        "prompt": request.prompt,
        "width": width,
        "height": height,
        "num_images_per_prompt": request.n,
    }

    if generator is not None:
        call_kwargs["generator"] = generator

    if request.num_inference_steps is not None:
        call_kwargs["num_inference_steps"] = request.num_inference_steps
    elif "schnell" in loaded_model_id.lower():
        call_kwargs["num_inference_steps"] = 4

    if request.guidance_scale is not None:
        call_kwargs["guidance_scale"] = request.guidance_scale
    elif "schnell" in loaded_model_id.lower():
        call_kwargs["guidance_scale"] = 0.0

    if request.negative_prompt:
        call_kwargs["negative_prompt"] = request.negative_prompt

    print(f"[server] Generating image: prompt='{request.prompt}', size={width}x{height}, steps={call_kwargs.get('num_inference_steps')}")

    try:
        # Run inference in a thread pool to avoid blocking FastAPI's event loop
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(None, lambda: pipeline(**call_kwargs))
        images = output.images

        results = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            results.append({"b64_json": b64_str})

        return {
            "created": int(time.time()),
            "data": results,
        }
    except Exception as e:
        print(f"[server] Error during generation: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    global idle_timeout_seconds, status_file_path, server_hostname, server_port

    parser = argparse.ArgumentParser(description="Run idiffusion Diffusers Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell", help="Model ID")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", help="torch dtype")
    parser.add_argument("--idle-timeout", type=int, default=30, help="Idle timeout in minutes")
    parser.add_argument("--status-file", type=str, default="", help="Path to status.json lockfile")
    args = parser.parse_args()

    server_port = args.port
    idle_timeout_seconds = args.idle_timeout * 60 if args.idle_timeout > 0 else -1
    status_file_path = args.status_file if args.status_file else None

    # Load model weights
    load_pipeline(args.model, args.torch_dtype)

    # Update status to running
    update_status_file({
        "status": "running",
        "computeHostname": server_hostname,
        "serverPort": server_port,
        "startTime": get_current_time_iso(),
    })

    print(f"[server] Starting Uvicorn on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
