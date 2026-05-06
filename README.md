# Face2Comic — Image-to-Image Translation (pix2pix)

Transform real face photos into comic-style portraits using a paired pix2pix Conditional GAN (U-Net generator + PatchGAN discriminator).

## Project Overview

Face2Comic learns a supervised mapping (real → comic) that preserves identity, pose, and key facial features. The pipeline includes preprocessing, hyperparameter tuning, final training, and quantitative + qualitative evaluation on a held-out test set.

## Dataset

- Training (paired): https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic (~10k pairs)
- Testing / Inference (unseen real faces): https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset

Split: 80% train / 10% val / 10% test (≈8k / 1k / 1k).

## Objective

- Translate real face images → comic-style images
- Preserve facial structure and identity
- Evaluate generalization to unseen faces

## Model (brief)

- Generator: U-Net (encoder–decoder with skip connections). Output uses `tanh` to match normalized targets.
- Discriminator: PatchGAN (patch-based realism encourages local detail).

## Losses

- Adversarial (BCE) — encourages realism
- L1 (pixel) — enforces reconstruction
- Final generator objective: G_loss = BCE + λ_L1 × L1

## Preprocessing

- Resize images to 256×256
- Convert to channels-first (C, H, W)
- Normalize to [-1, 1]
- Save arrays as `data/npy/*.npy` for fast, memory-mapped training

## Training & Tuning (summary)

- Hyperparameter grid search over LR, batch size, and `λ_L1` (27 configs). Best found: LR = 5e-4, Batch = 32, λ_L1 = 150.
- Final training: up to 200 epochs with linear LR decay (checkpointing and CSV logging). Checkpoints saved to `checkpoints/` (e.g., `pix2pix_best_epoch_066.pt`).

## Evaluation (metrics)

- Reported metrics: L1, MSE, PSNR, SSIM, LPIPS, VGG feature distance.
- Findings: model preserves global layout and identity; test metrics show moderate degradation vs. train (expected), and fine details (hair, textures) are the main weakness.

## Results

- Strengths: consistent comic-style conversion; preserves facial structure and pose; reasonable generalization.
- Limitations: softer fine details, struggles with extreme poses/occlusions; room for perceptual/sharpness improvements.

## Artifacts & Paths

- Checkpoints: `checkpoints/`
- Logs: `logs/` (grid search + full training CSVs)
- Outputs: `outputs/` (grid samples, epoch snapshots, `outputs/eval/metrics_train_test_tuning_nb.csv`)

## Future Improvements

- Add attention modules, perceptual loss tuning, or progressive training.
- Consider diffusion-based or stronger perceptual objectives for higher-fidelity outputs.

## Run the App (Upload → Comic)

This repo includes a lightweight Gradio app that:

- Takes an uploaded image
- Applies the same preprocessing + enhancement presets used in the evaluation notebook
- Runs the trained pix2pix generator from `checkpoints/pix2pix_best_epoch_181.pt`
- Returns a single enhanced comic image (with download)

### 1) Install dependencies

```bash
pip install -r requirements-app.txt
```

### 2) Start the app

```bash
python app.py
```

Then open the local URL printed in the terminal.



