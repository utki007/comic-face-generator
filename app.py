from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from src.inference import PRESETS, PresetName, load_model_bundle, run_inference


REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "pix2pix_best_epoch_263.pt"


@lru_cache(maxsize=1)
def _get_bundle():
    device, gen = load_model_bundle(CHECKPOINT_PATH)
    return device, gen


def _on_upload(_: Optional[Image.Image]) -> str:
    return "Upload complete. Choose a preset and click Generate."  # lightweight status indicator


def _on_clear() -> str:
    return ""


def _upload_bar(percent: int) -> str:
    pct = max(0, min(100, int(percent)))
    return f"<progress value='{pct}' max='100' style='width:100%'></progress>"


def generate_comic(
    img: Optional[Image.Image],
    preset: PresetName,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> Image.Image:
    if img is None:
        raise gr.Error("Please upload an image first.")

    progress(0.05, desc="Loading model")
    device, gen = _get_bundle()

    progress(0.25, desc="Preprocessing")
    # (enhance_input + center-crop/resize/normalize happens inside run_inference)

    progress(0.65, desc="Generating")
    out = run_inference(img, preset, gen=gen, device=device)

    progress(0.90, desc="Postprocessing")
    return out


with gr.Blocks(title="Face2Comic", fill_height=True) as demo:
    gr.Markdown("# Face2Comic")
    gr.Markdown("Upload a face photo and generate a comic-style image using the trained pix2pix model.")

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            input_image = gr.Image(
                label="Input image",
                type="pil",
                sources=["upload", "webcam", "clipboard"],
            )
            upload_status = gr.Markdown("")
            upload_bar = gr.HTML(_upload_bar(0))

            preset = gr.Dropdown(
                label="Enhancement preset",
                choices=list(PRESETS.keys()),
                value="balanced",
            )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1, min_width=320):
            output_image = gr.Image(
                label="Comic (enhanced)",
                type="pil",
                interactive=False,
                buttons=["download"],
            )

    input_image.upload(
        fn=lambda img: (_on_upload(img), _upload_bar(100)),
        inputs=input_image,
        outputs=[upload_status, upload_bar],
        show_progress="minimal",
    )
    input_image.clear(
        fn=lambda: (_on_clear(), _upload_bar(0)),
        outputs=[upload_status, upload_bar],
    )

    generate_btn.click(
        fn=generate_comic,
        inputs=[input_image, preset],
        outputs=[output_image],
        show_progress="full",
    )


demo.queue()


if __name__ == "__main__":
    demo.launch()
