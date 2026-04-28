1. Title
Face2Comic: Learning Comic Stylization of Faces using Paired GANs

2. Image Source
I will use a dual-dataset setup for training and evaluation:

Training Dataset (Paired):
https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic
~10,000 paired images (real face ↔ comic version)

Testing / Inference Dataset:
https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset
Unseen real face images for evaluating generalization

Goal:
Learn a mapping from real face images to comic-style representations while preserving identity (pose, structure, facial features), and evaluate how well the model generalizes to unseen real-world images.

3. Model Architecture
I will implement a Conditional GAN (cGAN) for paired image-to-image translation, trained from scratch.

Generator (G)
Architecture: U-Net (encoder–decoder with skip connections)

Purpose: Preserve spatial structure while applying comic-style transformation

Discriminator (D)
Architecture: PatchGAN

Purpose: Enforce local realism and sharp texture generation

Loss Functions
Adversarial Loss (BCE): Ensures generated images resemble comic domain

L1 Reconstruction Loss: Ensures output matches ground truth comic image

4. Extra Criteria
Hyperparameter Tuning
Tune:

Learning rates (G vs D)

L1 loss weight (lambda)

Batch size and training stability

Lightweight GUI (Node.js)
Build a simple UI using Node.js (Express + basic frontend):

Upload face image

Send image to backend for inference

Display generated comic-style output

Summary
This project focuses on implementing a paired image-to-image translation GAN from scratch to convert real faces into comic-style images. The emphasis is on correct architecture design, stable training, and evaluation of generalization, along with a simple Node.js-based interface for practical usability.