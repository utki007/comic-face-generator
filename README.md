

# 📘 Project Notes

## 📂 Dataset Links

* **Paired Training Dataset:**
  [https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic)

* **Real Faces Dataset (Testing / Inference):**
  [https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset](https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset)

## 🎯 Goal

Train a model to learn the mapping from **real face → comic-style face** while preserving identity (pose, facial structure, and key features).

Evaluate how well the model **generalizes to unseen real-world images** using a separate dataset.

## 📁 Data Setup

### Step 1: Download Datasets

Download both datasets from Kaggle and extract them locally.

### Step 2: Organize Directory Structure

```text
data/
  Input/
    real/        # Paired real face images (input)
    comic/       # Corresponding comic images (target)

  Inference/
    real/        # Unseen real images (from second dataset or user input)
    generated/   # Model-generated comic outputs
```

## ⚠️ Important Notes

* **Paired Alignment is Critical**
  Ensure that images in:

  ```
  real/0001.jpg ↔ comic/0001.jpg
  ```

  have matching filenames for correct training.

* **Do Not Mix Datasets**

  * `Input/` → training only
  * `Inference/` → testing and demo only

* **Keep Data Out of Git**
  Large datasets should not be committed. Use `.gitignore`.

## ✅ Summary

* Paired dataset → supervised training
* Unseen dataset → generalization testing
* Correct folder structure and alignment are essential for model performance

