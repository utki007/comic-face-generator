# Face2Comic: Learning Comic Stylization of Faces using Paired GANs

## 1. Project Name & Overview

### a) Project Name

Face2Comic: Learning Comic Stylization of Faces using Paired GANs

### b) Overview

Face2Comic learns a supervised mapping (real → comic) that preserves identity, pose, and key facial features. The pipeline includes [preprocessing](notebooks/2.%20%20%20Preprocessing.ipynb), [hyperparameter tuning](notebooks/3.%20%20%20Hyperparameter%20Tuning.ipynb), [final training](notebooks/4.%20%20%20Model%20Training.ipynb), and [quantitative + qualitative evaluation](notebooks/5.%20%20%20Evaluation.ipynb) on a held-out test set.

The project also evaluates real world generalization and includes a lightweight web app for browser based comic generation from uploaded images.

#### i. Model

- Generator: U-Net encoder-decoder with skip connections for preserving spatial structure
- Discriminator: PatchGAN operating on image patches to enforce local realism

#### ii. Losses

- Adversarial (BCE): encourages realism
- L1 (pixel): enforces reconstruction
- Final generator objective: `G_loss = BCE + lambda_L1 x L1`

#### iii. Training & Tuning

- Preprocessing resizes to 256×256, creates paired 80/10/10 splits, normalizes to [-1, 1], and runs integrity checks.
- Conduct Hyperparemeter Tuning over 27 configuration
- Applied paired augmentations on the fly for training dataset
- Final training up to 300 epochs. [Logs](notebooks/Output/full_training_log.csv)
- Added linear learning-rate decay starting at epoch 150.
- Used label smoothing with real labels set to 0.9 for more stable GAN training.

# 2. Installation & run instructions.

## Model

- Download dataset from [Google Drive](https://drive.google.com/drive/folders/1JX1z6J0goF54lWUZCl-N77aN9lusY3ES?usp=sharing)
- Run [2.   Preprocessing.ipynb](notebooks/2.%20%20%20Preprocessing.ipynb), [3.   Hyperparameter Tuning.ipynb](notebooks/3.%20%20%20Hyperparameter%20Tuning.ipynb), [4.   Model Training.ipynb](notebooks/4.%20%20%20Model%20Training.ipynb) & [5.   Evaluation.ipynb](notebooks/5.%20%20%20Evaluation.ipynb) in order.

## Frontend

From the repository root (runs on epoch 263):

```bash
pip install -r requirements-app.txt
npm install
npm run dev
```

[Inference.py](src/inference.py) is being used to run my frontend model end-to-end.

# 3. Results of the Model

<img src="Report/Images/Picture9.png" width="500">

# 4. Extra Criteria Pursued

## a) Hyperparameter Tuning

Performed hyperparameter grid search for a subset of dataset (2000 train, 500 validation) over learning rate, batch size, and lambda_L1 across 27 configurations (10 epochs each). The best configuration used LR = 5e-4, Batch Size = 32, and lambda_L1 = 100, achieving a validation L1 loss of 0.2262. [Grid Search Samples](notebooks/Output/epoch_samples)

## b) Gallery GUI / Web Application

The React GUI lets users upload an image, auto crops and resizes it to 256×256, and sends it to a backend for instant comic generation with download, copy, and regenerate options. It also includes an optional enhancement toggle for improved preprocessing and postprocessing effects or a raw output mode.

<img src="Report/Images/Picture4.png" width="600">

## 5. Difficulties Faced & Solutions

A key challenge was poor generalization to real world images due to the synthetic training distribution. This was improved using stronger augmentations (noise, color jitter, blur), which significantly increased robustness. Training stability was further improved with label smoothing, linear learning rate decay, and longer training, resulting in more consistent and realistic outputs.
