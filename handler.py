"""
RunPod Serverless Handler — SkyReels V1 Video Generation (I2V)
Uses the diffusers library with SkyReels V1 model from HuggingFace.
"""
import os
import runpod
import torch
import requests
import base64
import tempfile
from io import BytesIO
from PIL import Image

PIPE = None
CACHE_DIR = "/cache/skyreels"


def load_pipeline():
    global PIPE
    if PIPE is not None:
        return PIPE

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("[skyreels] Loading SkyReels V1 pipeline...")

    from diffusers import DiffusionPipeline

    PIPE = DiffusionPipeline.from_pretrained(
        "Skywork/SkyReels-V1-Hunyuan-I2V",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    ).to("cuda")

    print("[skyreels] Pipeline loaded.")
    return PIPE


def download_image(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def handler(event):
    inp = event.get("input", {})
    prompt = inp.get("prompt", "")
    image_url = inp.get("image_url")

    if not prompt:
        return {"error": "prompt is required"}
    if not image_url:
        return {"error": "image_url is required (SkyReels is I2V)"}

    duration = inp.get("duration", 5)
    width = inp.get("width", 768)
    height = inp.get("height", 1344)
    fps = inp.get("fps", 24)
    num_frames = fps * duration

    try:
        pipe = load_pipeline()
        image = download_image(image_url)
        image = image.resize((width, height))

        output = pipe(
            prompt=prompt,
            image=image,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=inp.get("steps", 30),
            guidance_scale=inp.get("guidance_scale", 7.0),
        )

        # Export video frames to mp4
        frames = output.frames[0]  # List of PIL Images

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        # Use imageio to write video
        import imageio
        import numpy as np
        writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264")
        for frame in frames:
            if hasattr(frame, "numpy"):
                writer.append_data(np.array(frame))
            else:
                writer.append_data(np.array(frame))
        writer.close()

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(tmp_path)

        return {
            "video_base64": video_b64,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
        }
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
