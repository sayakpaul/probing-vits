# USAGE
# python train.py --classifier token --position learn
# python train.py --classifier token --position sincos
# python train.py --classifier gap --position learn
# python train.py --classifier gap --position sincos

# import the necessary packages
from vit import (
    get_cifar10_config,
    get_cifar_dataset,
    ViTClassifier,
    WarmUpCosine,
    get_augmentation_model,
)

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

import argparse
from pprint import pformat

import logging

_AUTO = tf.data.AUTOTUNE

logging.getLogger().setLevel(logging.INFO)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-c", "--classifier", required=True, help="token or gap for the vit representation"
)
ap.add_argument(
    "-p", "--position", required=True, help="learn or sincos for positional embedding"
)

args = vars(ap.parse_args())

# manipulating the configuration for CIFAR
cifar10_config = get_cifar10_config()
with cifar10_config.unlocked():
    # Update the config according to the CLI arguments
    cifar10_config.pos_emb_mode = args["position"]
    cifar10_config.classifier = args["classifier"]

logging.info(pformat(cifar10_config))

logging.info("Grabbing the CIFAR10 dataset...")
(train_ds, val_ds, test_ds) = get_cifar_dataset(config=cifar10_config)

# preprocess the training, validation and the testing dataset
train_augmentation_model = get_augmentation_model(config=cifar10_config, train=True)
test_augmentation_model = get_augmentation_model(config=cifar10_config, train=False)
train_ds = train_ds.map(
    lambda image, label: (train_augmentation_model(image), label)
).prefetch(_AUTO)
val_ds = val_ds.map(
    lambda image, label: (test_augmentation_model(image), label)
).prefetch(_AUTO)
test_ds = test_ds.map(
    lambda image, label: (test_augmentation_model(image), label)
).prefetch(_AUTO)

# building the vit_classifier
logging.info("Building the ViT model...")
vit_classifier = ViTClassifier(cifar10_config, name="vit")

# compiling the model
logging.info("Compiling the model...")
total_steps = int(train_ds.cardinality().numpy() * cifar10_config.epochs)
warmup_steps = int(total_steps * cifar10_config.warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    config=cifar10_config,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
)
optimizer = tfa.optimizers.AdamW(
    learning_rate=scheduled_lrs,
    weight_decay=cifar10_config.weight_decay,
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
logging.info("Training the model...")
history = vit_classifier.fit(
    train_ds,
    epochs=cifar10_config.epochs,
    validation_data=val_ds,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cifar10_config.patience,
            mode="auto",
        )
    ],
)

# testing the model
logging.info("Testing the model...")
loss, acc_top1, acc_top5 = vit_classifier.evaluate(test_ds)
logging.info(f"Loss: {loss:0.2f}")
logging.info(f"Top 1 test accuracy: {acc_top1*100:0.2f}%")
logging.info(f"Top 5 test accuracy: {acc_top5*100:0.2f}%")
