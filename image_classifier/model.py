"""A compact Vision Transformer built with TensorFlow/Keras."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="image_classifier")
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        return {**super().get_config(), "num_patches": self.num_patches,
                "projection_dim": self.projection.units}


def build_model(num_classes: int, image_size: int = 224, patch_size: int = 16,
                dim: int = 384, depth: int = 6, heads: int = 6,
                mlp_dim: int = 768, dropout: float = 0.1):
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by patch_size")
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    patches = layers.Conv2D(dim, kernel_size=patch_size, strides=patch_size, padding="valid")(x)
    num_patches = (image_size // patch_size) ** 2
    x = layers.Reshape((num_patches, dim))(patches)
    x = PatchEncoder(num_patches, dim)(x)
    for _ in range(depth):
        residual = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=dropout)(x, x)
        x = layers.Add()([x, residual])
        residual = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(dim)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Add()([x, residual])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="vision_transformer")
