import ml_collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PositionalEmbedding(layers.Layer):
    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Compute the positions.
        positions = self.config.num_patches
        positions += 1 if self.config.classifier == "token" else 0

        # Build the sequence of positions in 1D.
        self.pos_flat_patches = tf.range(positions, dtype=tf.float32, delta=1)

        # Encode the positions with an Embedding layer.
        if self.config.pos_emb_mode == "learn":
            self.pos_embedding = layers.Embedding(
                input_dim=self.config.num_patches + 1
                if self.config.classifier == "token"
                else self.config.num_patches,
                output_dim=self.config.projection_dim,
                embeddings_initializer=keras.initializers.RandomNormal(
                    stddev=0.02
                ),
            )

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def get_1d_sincos_pos_embed(self):
        # Inspired from https://github.com/huggingface/transformers/blob/master/src/transformers/models/vit_mae/modeling_vit_mae.py#L184.
        # Build the sine-cosine positional embedding.
        omega = tf.range(self.config.projection_dim // 2, dtype=tf.float32)
        omega /= self.config.projection_dim / 2.0
        omega = 1.0 / 10000 ** omega  # (D/2,)

        out = tf.einsum(
            "m,d->md", self.pos_flat_patches, omega
        )  # (M, D/2), outer product

        emb_sin = tf.sin(out)  # (M, D/2)
        emb_cos = tf.cos(out)  # (M, D/2)

        emb = tf.concat([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def get_learnable_pos_embed(self):
        emb = self.pos_embedding(self.pos_flat_patches)
        return emb

    def call(self, inputs):
        if self.config.pos_emb_mode == "learn":
            pos_emb = self.get_learnable_pos_embed()
        else:
            pos_emb = self.get_1d_sincos_pos_embed()

        # Inject the positional embeddings with the tokens.
        if pos_emb.dtype != inputs.dtype:
            pos_emb = tf.cast(pos_emb, inputs.dtype)
        outputs = inputs + pos_emb
        return outputs
