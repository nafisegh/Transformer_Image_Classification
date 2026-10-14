"""Dataset discovery, splitting, and TensorFlow input pipelines."""

import json
import random
import shutil
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def prepare_data(source: Path, output: Path, val_fraction: float, test_fraction: float, seed: int):
    """Split class-folder images into output/{train,val,test}/{class}/..."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be nonnegative and sum to less than 1")
    classes = sorted(p.name for p in source.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("Source must contain at least two class subdirectories")
    rng = random.Random(seed)
    for cls in classes:
        files = [p for p in (source / cls).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
        if len(files) < 3:
            raise ValueError(f"Class {cls!r} needs at least 3 images; found {len(files)}")
        rng.shuffle(files)
        n_test = max(1, round(len(files) * test_fraction)) if test_fraction else 0
        n_val = max(1, round(len(files) * val_fraction)) if val_fraction else 0
        while n_test + n_val >= len(files):
            if n_val >= n_test and n_val:
                n_val -= 1
            elif n_test:
                n_test -= 1
            else:
                break
        parts = {"test": files[:n_test], "val": files[n_test:n_test + n_val],
                 "train": files[n_test + n_val:]}
        for split, paths in parts.items():
            for i, path in enumerate(paths):
                target = output / split / cls / f"{i:06d}_{path.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    (output / "classes.json").write_text(json.dumps(classes, indent=2) + "\n")


def make_dataset(root: Path, split: str, image_size: int, batch_size: int, training: bool = False):
    import tensorflow as tf
    path = root / split
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {split} directory: {path}")
    dataset = tf.keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="int", image_size=(image_size, image_size),
        batch_size=batch_size, shuffle=training, seed=42,
    )
    class_names = dataset.class_names
    if training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
        ])
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE), class_names
