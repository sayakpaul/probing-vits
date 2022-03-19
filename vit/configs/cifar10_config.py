import ml_collections


def get_cifar10_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.ds_name = "cifar10"

    config.batch_size = 256
    config.num_training_examples = 40000
    config.buffer_size = config.batch_size * 2
    config.input_shape = (32, 32, 3)

    config.image_size = 48
    config.patch_size = 4
    config.num_patches = (config.image_size // config.patch_size) ** 2
    config.num_classes = 10

    config.pos_emb_mode = "sincos"

    config.layer_norm_eps = 1e-6
    config.projection_dim = 128
    config.num_heads = 4
    config.num_layers = 6
    config.mlp_units = [
        config.projection_dim * 4,
        config.projection_dim,
    ]
    config.dropout_rate = 0.2
    config.classifier = "token"

    config.epochs = 100
    config.warmup_epoch_percentage = 0.15
    config.lr_start = 1e-5
    config.lr_max = 1e-3
    config.weight_decay = 1e-4
    config.patience = 5
    config.artifact_dir = "."

    return config.lock()
