"""Command line interface for the TensorFlow image-classification workflow."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data import make_dataset, prepare_data
from .model import build_model


def sidecar_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".json")


def prepare(args):
    prepare_data(args.source, args.output, args.val_fraction, args.test_fraction, args.seed)
    print(f"Prepared splits under {args.output}")


def train(args):
    tf.keras.utils.set_random_seed(args.seed)
    train_ds, classes = make_dataset(args.data, "train", args.image_size, args.batch_size, training=True)
    val_ds, val_classes = make_dataset(args.data, "val", args.image_size, args.batch_size)
    if classes != val_classes:
        raise ValueError("Train and validation class names/order differ")
    model_config = dict(num_classes=len(classes), image_size=args.image_size, patch_size=args.patch_size,
                        dim=args.dim, depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,
                        dropout=args.dropout)
    model = build_model(**model_config)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(args.output), monitor="val_loss", save_best_only=True,
                                           save_weights_only=False),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                         restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    sidecar_path(args.output).write_text(json.dumps({"classes": classes, "config": model_config}, indent=2) + "\n")
    print(f"Saved best model to {args.output}")


def load_model(checkpoint: Path):
    metadata = json.loads(sidecar_path(checkpoint).read_text())
    return tf.keras.models.load_model(checkpoint), metadata


def evaluate(args):
    model, metadata = load_model(args.checkpoint)
    dataset, classes = make_dataset(args.data, args.split, metadata["config"]["image_size"], args.batch_size)
    if classes != metadata["classes"]:
        raise ValueError("Dataset classes/order do not match the checkpoint")
    truth, predictions = [], []
    for images, labels in dataset:
        probabilities = model(images, training=False).numpy()
        truth.extend(labels.numpy().tolist())
        predictions.extend(probabilities.argmax(axis=1).tolist())
    report = {"split": args.split, "samples": len(truth),
              "accuracy": float(accuracy_score(truth, predictions)),
              "classification_report": classification_report(
                  truth, predictions, labels=list(range(len(classes))), target_names=classes,
                  output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(
                  truth, predictions, labels=list(range(len(classes)))).tolist()}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")


def predict(args):
    model, metadata = load_model(args.checkpoint)
    image_size = metadata["config"]["image_size"]
    image = tf.keras.utils.load_img(args.image, target_size=(image_size, image_size), color_mode="rgb")
    tensor = tf.keras.utils.img_to_array(image)[None, ...]
    probabilities = model.predict(tensor, verbose=0)[0]
    index = int(np.argmax(probabilities))
    print(json.dumps({"image": str(args.image), "label": metadata["classes"][index],
                      "confidence": float(probabilities[index]),
                      "scores": {name: float(score) for name, score in zip(metadata["classes"], probabilities)},
                      }, indent=2))


def model_options(parser):
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)


def main():
    parser = argparse.ArgumentParser(prog="vit-classifier")
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("prepare", help="Split class-folder images into train/val/test")
    p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.15); p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42); p.set_defaults(func=prepare)
    p = commands.add_parser("train", help="Train the Vision Transformer")
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5); p.add_argument("--seed", type=int, default=42)
    model_options(p); p.set_defaults(func=train)
    p = commands.add_parser("evaluate", help="Evaluate on a labeled split")
    p.add_argument("--data", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="test"); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--report", type=Path); p.set_defaults(func=evaluate)
    p = commands.add_parser("predict", help="Predict one image")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--image", type=Path, required=True)
    p.set_defaults(func=predict)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
