# USAGE
# python train.py --classifier token --position learn
# python train.py --classifier token --position sincos
# python train.py --classifier gap --position learn
# python train.py --classifier gap --position sincos

# import the necessary packages
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import argparse
from vit import (
    get_config,
    get_cifar_dataset,
    ViTClassifier,
    WarmUpCosine,
)

import pprint
pp = pprint.PrettyPrinter(indent=4)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classifier", required=True,
    help="token or gap for the vit representation")
ap.add_argument("-p", "--position", required=True,
    help="learn or sincos for positional embedding")
args = vars(ap.parse_args())

# manipulating the configuration for CIFAR
cifar_config = get_config()
with cifar_config.unlocked():
    # Update the config according to the CLI arguments
    cifar_config.pos_emb_mode = args["position"]
    cifar_config.classifier = args["classifier"]

    # Update the config for the CIFAR dataset
    cifar_config.batch_size = 256
    cifar_config.buffer_size = cifar_config.batch_size * 2
    cifar_config.input_shape = (32, 32, 3)
    cifar_config.image_size = 48
    cifar_config.patch_size = 4
    cifar_config.num_patches = (cifar_config.image_size // cifar_config.patch_size) ** 2
    cifar_config.projection_dim = 128
    cifar_config.num_heads = 4
    cifar_config.num_layers = 6
    cifar_config.mlp_units = [
        cifar_config.projection_dim * 4,
        cifar_config.projection_dim,
    ]
    cifar_config.dropout_rate = 0.2
    cifar_config.epochs = 100
    cifar_config.warmup_epoch_percentage = 0.15
    cifar_config.lr_start = 1e-5
    cifar_config.lr_max = 1e-3
    cifar_config.weight_decay = 1e-4
    cifar_config.patience = 5

print("🚀 [INFO] Configuration...")
pp.pprint(cifar_config)

# load the dataset
print("🚀 [INFO] Loading the CIFAR10 dataset...")
(train_ds, val_ds, test_ds) = get_cifar_dataset(config=cifar_config)

# building the vit_classifier
print("🚀 [INFO] Building the ViT model...")
vit_classifier = ViTClassifier(cifar_config, name="vit")

# compiling the model
print("🚀 [INFO] Compiling the model...")
total_steps = int(train_ds.cardinality().numpy() * cifar_config.epochs)
warmup_steps = int(total_steps * cifar_config.warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    lr_start=cifar_config.lr_start,
    lr_max=cifar_config.lr_max,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
)
optimizer = tfa.optimizers.AdamW(
    learning_rate=scheduled_lrs,
    weight_decay=cifar_config.weight_decay,
)
vit_classifier.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

# training the model
print("🚀 [INFO] Training the model...")
history = vit_classifier.fit(
    train_ds,
    epochs=cifar_config.epochs,
    validation_data=val_ds,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cifar_config.patience,
            mode="auto",
        )
    ],
)

# testing the model
print("🚀 [INFO] Testing the model...")
loss, acc_top1, acc_top5 = vit_classifier.evaluate(test_ds)
print(f"Loss: {loss:0.2f}")
print(f"Top 1 test accuracy: {acc_top1*100:0.2f}%")
print(f"Top 5 test accuracy: {acc_top5*100:0.2f}%")
