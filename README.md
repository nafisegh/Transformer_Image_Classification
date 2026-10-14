# TensorFlow Vision Transformer Image Classifier

A Python command-line application for supervised, multi-class image classification using TensorFlow/Keras. It implements data preparation, transformer training, inference/testing, and accuracy reporting.

## Requirements and install

Python 3.10+. Install TensorFlow for your platform, then install this project:

```bash
python -m pip install -e .
```

## Input data layout

Put images in one directory per class. The directory names become class labels:

```text
images/
  cat/  (image files)
  dog/  (image files)
```

Use independent examples in each class. For repeated subjects, videos, patients, or near-duplicate images, split by subject/source before training to avoid leakage; the built-in random splitter assumes individual images are independent.

## Workflow

### 1. Prepare train, validation, and test data

```bash
vit-classifier prepare --source ./images --output ./data --val-fraction 0.15 --test-fraction 0.15
```

The command copies files into `data/train`, `data/val`, and `data/test`, retaining class subdirectories, and writes `classes.json`. It uses a fixed seed by default (`--seed 42`).

### 2. Train the transformer

```bash
vit-classifier train --data ./data --output ./artifacts/best_vit.keras --epochs 20 --batch-size 32
```

The software trains a Vision Transformer from scratch, prints training and validation loss/accuracy each epoch, and saves the checkpoint with the lowest validation loss as a Keras `.keras` model. A neighboring JSON file stores class names and model configuration. Configure model size with `--dim`, `--depth`, `--heads`, `--mlp-dim`, `--image-size`, and `--patch-size`.

### 3. Inference and test

Evaluate the held-out test set:

```bash
vit-classifier evaluate --data ./data --checkpoint ./artifacts/best_vit.keras --split test --report ./artifacts/test_metrics.json
```

Predict one image:

```bash
vit-classifier predict --checkpoint ./artifacts/best_vit.keras --image ./new_image.jpg
```

Inference returns the predicted label, confidence, and scores for each class. The test set is intended for final evaluation and should not be used for model selection.

### 4. Accuracy reporting

`evaluate` prints a JSON report containing overall accuracy, sample count, per-class precision/recall/F1, macro and weighted averages, and a confusion matrix. `--report` saves the same report to a JSON file. Accuracy is the fraction of correct predictions across evaluated samples.

## Notes

- This is a minimal educational baseline. The ViT is initialized randomly; it does not download or use pretrained weights. For small datasets, use transfer learning with a pretrained model.
- Training can require substantial compute. Start with a smaller model (for example `--dim 192 --depth 4 --heads 3 --mlp-dim 384`) and fewer epochs when exploring.
- The implementation expects RGB images and multi-class labels stored in class directories.
