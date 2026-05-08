from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps

from src.inference import PRESETS, load_model_bundle, run_inference


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "pix2pix_best_epoch_263.pt"


def _get_checkpoint_path() -> Path:
    raw = os.getenv("FACE2COMIC_CHECKPOINT", "").strip()
    return Path(raw) if raw else DEFAULT_CHECKPOINT_PATH


@lru_cache(maxsize=1)
def _get_bundle():
    checkpoint = _get_checkpoint_path()
    device, gen = load_model_bundle(checkpoint)
    return device, gen


def _parse_origins(env: str) -> list[str]:
    items = [x.strip() for x in env.split(",")]
    return [x for x in items if x]


app = FastAPI(title="Face2Comic API")

cors_origins = _parse_origins(
    os.getenv(
        "FACE2COMIC_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/presets")
def presets() -> dict[str, list[str]]:
    return {"presets": list(PRESETS.keys())}


def _read_pil_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


@app.post("/api/generate")
async def generate(
    file: UploadFile = File(...),
    preset: str = Form("balanced"),
    enhance: str = Form("true"),
) -> Response:
    if preset not in PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid preset. Choose one of: {', '.join(PRESETS.keys())}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    pil_img = _read_pil_image(file_bytes)
    use_enhancement = enhance.lower() not in {"false", "0", "no", "off"}

    device, gen = _get_bundle()
    out = run_inference(
        pil_img,
        preset,
        gen=gen,
        device=device,
        use_enhancement=use_enhancement,
    )

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
