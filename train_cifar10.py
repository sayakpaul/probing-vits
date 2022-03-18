# USAGE
# python train_cifar10.py --classifier token --position learn
# python train_cifar10.py --classifier token --position sincos
# python train_cifar10.py --classifier gap --position learn
# python train_cifar10.py --classifier gap --position sincos


import argparse
import logging
import os
from datetime import datetime
from pprint import pformat

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from vit import (
    ViTClassifier,
    WarmUpCosine,
    get_augmentation_model,
    get_cifar10_config,
    get_cifar_dataset,
)
from vit.utils import logger

_AUTO = tf.data.AUTOTUNE

logging.getLogger().setLevel(logging.INFO)


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--classifier",
        required=True,
        help="token or gap for the vit representation",
    )
    ap.add_argument(
        "-p",
        "--position",
        required=True,
        help="learn or sincos for positional embedding",
    )

    args = vars(ap.parse_args())
    return args


def main(args):
    cifar10_config = get_cifar10_config()
    timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")

    with cifar10_config.unlocked():
        # Update the config according to the CLI arguments.
        cifar10_config.pos_emb_mode = args["position"]
        cifar10_config.classifier = args["classifier"]
        cifar10_config.artifact_dir = f"{cifar10_config.ds_name}-{args['position']}-{args['classifier']}-{timestamp}"

    if not os.path.exists(cifar10_config.artifact_dir):
        os.makedirs(cifar10_config.artifact_dir)

    logfile = os.path.join(cifar10_config.artifact_dir, f"logs.txt")

    with logger.Logger(logfile):
        logging.info(pformat(cifar10_config))

        (train_ds, val_ds, test_ds) = get_cifar_dataset(config=cifar10_config)

        train_augmentation_model = get_augmentation_model(
            config=cifar10_config, train=True
        )
        test_augmentation_model = get_augmentation_model(
            config=cifar10_config, train=False
        )
        train_ds = train_ds.map(
            lambda image, label: (train_augmentation_model(image), label),
            num_parallel_calls=_AUTO,
        ).prefetch(_AUTO)
        val_ds = val_ds.map(
            lambda image, label: (test_augmentation_model(image), label),
            num_parallel_calls=_AUTO,
        ).prefetch(_AUTO)
        test_ds = test_ds.map(
            lambda image, label: (test_augmentation_model(image), label),
            num_parallel_calls=_AUTO,
        ).prefetch(_AUTO)

        logging.info("Building the ViT model...")
        vit_classifier = ViTClassifier(cifar10_config, name="vit")

        logging.info("Compiling the model...")
        total_steps = int(
            train_ds.cardinality().numpy() * cifar10_config.epochs
        )
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
                keras.metrics.SparseTopKCategoricalAccuracy(
                    5, name="top-5-accuracy"
                ),
            ],
        )

        logging.info("Training the model...")
        ckpt_path = os.path.join(cifar10_config.artifact_dir, "checkpoints")
        _ = vit_classifier.fit(
            train_ds,
            epochs=cifar10_config.epochs,
            validation_data=val_ds,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath=ckpt_path,
                    monitor="val_accuracy",
                    mode="max",
                    save_best_only=True,
                )
            ],
        )

        logging.info("Testing the model...")
        loss, acc_top1, acc_top5 = vit_classifier.evaluate(test_ds)
        logging.info(f"Loss: {loss:0.2f}")
        logging.info(f"Top 1 test accuracy: {acc_top1*100:0.2f}%")
        logging.info(f"Top 5 test accuracy: {acc_top5*100:0.2f}%")

        logging.info("Serializing model with the best checkpoints.")
        vit_classifier.load_weights(ckpt_path)
        vit_classifier.save(
            os.path.join(cifar10_config.artifact_dir, "best_model")
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
