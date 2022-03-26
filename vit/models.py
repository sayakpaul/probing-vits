from typing import List

import ml_collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .layers.mha import TFViTAttention
from .layers.pos_embed import PositionalEmbedding


def mlp(x: int, dropout_rate: float, hidden_units: List):
    """FFN for a Transformer block."""
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for (idx, units) in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=tf.nn.gelu if idx == 0 else None,
            kernel_initializer="glorot_uniform",
            bias_initializer=keras.initializers.RandomNormal(stddev=1e-6),
        )(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer(config: ml_collections.ConfigDict, name: str) -> keras.Model:
    """Transformer block with pre-norm."""
    num_patches = (
        config.num_patches + 1
        if config.classifier == "token"
        else config.num_patches + 0
    )
    encoded_patches = layers.Input((num_patches, config.projection_dim))

    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(
        encoded_patches
    )

    # Multi Head Self Attention layer 1.
    attention_output, attention_score = layers.MultiHeadAttention(
        num_heads=config.num_heads,
        key_dim=config.projection_dim,
        dropout=config.dropout_rate,
    )(x1, x1, return_attention_scores=True)
    attention_output = layers.Dropout(config.dropout_rate)(attention_output)

    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(x2)

    # MLP layer 1.
    x4 = mlp(
        x3, hidden_units=config.mlp_units, dropout_rate=config.dropout_rate
    )

    # Skip connection 2.
    outputs = layers.Add()([x2, x4])

    return keras.Model(encoded_patches, [outputs, attention_score], name=name)


def transformer_extended(
    config: ml_collections.ConfigDict, name: str
) -> keras.Model:
    """Transformer block with pre-norm. This layer is re-written to port the
    pre-trained JAX weights.
    """
    num_patches = (
        config.num_patches + 1
        if config.classifier == "token"
        else config.num_patches + 0
    )
    encoded_patches = layers.Input((num_patches, config.projection_dim))

    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(
        encoded_patches
    )

    # Multi Head Self Attention layer 1.
    attention_output, attention_score = TFViTAttention(config)(
        x1, output_attentions=True
    )

    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(x2)

    # MLP layer 1.
    x4 = mlp(
        x3, hidden_units=config.mlp_units, dropout_rate=config.dropout_rate
    )

    # Skip connection 2.
    outputs = layers.Add()([x2, x4])

    return keras.Model(encoded_patches, [outputs, attention_score], name=name)


class ViTClassifier(keras.Model):
    """Class that collates all the different elements for a Vision Transformer."""

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=config.projection_dim,
                    kernel_size=(config.patch_size, config.patch_size),
                    strides=(config.patch_size, config.patch_size),
                    padding="VALID",
                    name="conv_projection",
                ),
                layers.Reshape(
                    target_shape=(config.num_patches, config.projection_dim),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )

        self.positional_embedding = PositionalEmbedding(
            config, name="positional_embedding"
        )
        self.transformer_blocks = [
            transformer(config, name=f"transformer_block_{i}")
            for i in range(config.num_layers)
        ]

        if config.classifier == "token":
            initial_value = tf.zeros((1, 1, config.projection_dim))
            self.cls_token = tf.Variable(
                initial_value=initial_value, trainable=True, name="cls"
            )

        if config.classifier == "gap":
            self.gap_layer = layers.GlobalAvgPool1D()

        self.dropout = layers.Dropout(config.dropout_rate)
        self.layer_norm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps
        )
        self.classifier_head = layers.Dense(
            config.num_classes,
            kernel_initializer="zeros",
            dtype="float32",
            name="classifier",
        )

    def call(self, inputs, training=True, pre_logits=False):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.projection(inputs)

        # Append class token if needed.
        if self.config.classifier == "token":
            cls_token = tf.tile(self.cls_token, (n, 1, 1))
            if cls_token.dtype != projected_patches.dtype:
                cls_token = tf.cast(cls_token, projected_patches.dtype)
            projected_patches = tf.concat(
                [cls_token, projected_patches], axis=1
            )

        # Add positional embeddings to the projected patches.
        encoded_patches = self.positional_embedding(
            projected_patches
        )  # (B, number_patches, projection_dim)
        encoded_patches = self.dropout(encoded_patches)

        if not training:
            attention_scores = dict()

        # Iterate over the number of layers and stack up blocks of
        # Transformer.
        for transformer_module in self.transformer_blocks:
            # Add a Transformer block.
            encoded_patches, attention_score = transformer_module(
                encoded_patches
            )
            if not training:
                attention_scores[
                    f"{transformer_module.name}_att"
                ] = attention_score

        # Final layer normalization.
        representation = self.layer_norm(encoded_patches)

        # Pool representation.
        if self.config.classifier == "token":
            encoded_patches = representation[:, 0]
        elif self.config.classifier == "gap":
            encoded_patches = self.gap_layer(representation)

        if pre_logits:
            return encoded_patches

        else:
            # Classification head.
            output = self.classifier_head(encoded_patches)

            if not training:
                return output, attention_scores
            else:
                return output


class ViTClassifierExtended(keras.Model):
    """Class that collates all the different elements for a Vision Transformer.
    This class is for porting the original JAX weights to TF/Keras.
    """

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=config.projection_dim,
                    kernel_size=(config.patch_size, config.patch_size),
                    strides=(config.patch_size, config.patch_size),
                    padding="VALID",
                    name="conv_projection",
                ),
                layers.Reshape(
                    target_shape=(config.num_patches, config.projection_dim),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )

        init_value = tf.ones(
            (
                1,
                config.num_patches + 1
                if self.config.classifier == "token"
                else config.num_patches,
                config.projection_dim,
            )
        )
        self.positional_embedding = tf.Variable(
            init_value, name="pos_embedding"
        )  # This will be loaded with the pre-trained positional embeddings later.

        self.transformer_blocks = [
            transformer_extended(config, name=f"transformer_block_{i}")
            for i in range(config.num_layers)
        ]  # Extended transformer block to easily load the pre-train variables especially
        # in the attention layers.

        if config.classifier == "token":
            initial_value = tf.zeros((1, 1, config.projection_dim))
            self.cls_token = tf.Variable(
                initial_value=initial_value, trainable=True, name="cls"
            )

        if config.classifier == "gap":
            self.gap_layer = layers.GlobalAvgPool1D()

        self.dropout = layers.Dropout(config.dropout_rate)
        self.layer_norm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps
        )
        self.classifier_head = layers.Dense(
            config.num_classes,
            kernel_initializer="zeros",
            dtype="float32",
            name="classifier",
        )

    def call(self, inputs, training=True, pre_logits=False):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.projection(inputs)

        # Append class token if needed.
        if self.config.classifier == "token":
            cls_token = tf.tile(self.cls_token, (n, 1, 1))
            if cls_token.dtype != projected_patches.dtype:
                cls_token = tf.cast(cls_token, projected_patches.dtype)
            projected_patches = tf.concat(
                [cls_token, projected_patches], axis=1
            )

        # Add positional embeddings to the projected patches.
        encoded_patches = (
            self.positional_embedding + projected_patches
        )  # (B, number_patches, projection_dim)
        encoded_patches = self.dropout(encoded_patches)

        if not training:
            attention_scores = dict()

        # Iterate over the number of layers and stack up blocks of
        # Transformer.
        for transformer_module in self.transformer_blocks:
            # Add a Transformer block.
            encoded_patches, attention_score = transformer_module(
                encoded_patches
            )
            if not training:
                attention_scores[
                    f"{transformer_module.name}_att"
                ] = attention_score

        # Final layer normalization.
        representation = self.layer_norm(encoded_patches)

        # Pool representation.
        if self.config.classifier == "token":
            encoded_patches = representation[:, 0]
        elif self.config.classifier == "gap":
            encoded_patches = self.gap_layer(representation)

        if pre_logits:
            return encoded_patches
        
        else:
            # Classification head.
            output = self.classifier_head(encoded_patches)

            if not training:
                return output, attention_scores
            else:
                return output


def get_augmentation_model(config: ml_collections.ConfigDict, train=True):
    """Augmentation transformation models."""
    if train:
        data_augmentation = keras.Sequential(
            [
                layers.Resizing(
                    config.input_shape[0] + 20, config.input_shape[0] + 20
                ),
                layers.RandomCrop(config.image_size, config.image_size),
                layers.RandomFlip("horizontal"),
                layers.Rescaling(1 / 255.0),
            ],
            name="train_aug",
        )
    else:
        data_augmentation = keras.Sequential(
            [
                layers.Resizing(
                    config.input_shape[0] + 20, config.input_shape[0] + 20
                ),
                layers.CenterCrop(config.image_size, config.image_size),
                layers.Rescaling(1 / 255.0),
            ],
            name="test_aug",
        )
    return data_augmentation
