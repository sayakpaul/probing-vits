import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.batch_size = 32
    config.buffer_size = config.batch_size * 2
    config.input_shape = (224, 224, 3)

    config.image_size = 224
    config.patch_size = 16
    config.num_patches = (config.image_size // config.patch_size) ** 2
    config.num_classes = 10

    config.pos_emb_mode = "sincos"

    config.initializer_range = 0.02
    config.layer_norm_eps = 1e-6
    config.projection_dim = 768
    config.num_heads = 12
    config.num_layers = 12
    config.mlp_units = [
        config.projection_dim * 4,
        config.projection_dim,
    ]
    config.dropout_rate = 0.0
    config.classifier = "token"

    return config.lock()
