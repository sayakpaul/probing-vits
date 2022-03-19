from typing import Tuple

import ml_collections
import tensorflow as tf
from tensorflow import keras


def get_cifar_dataset(
    config: ml_collections.ConfigDict,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Loads the CIFAR-10 dataset and prepares tf.data.Dataset objects."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    num_training_examples = config.num_training_examples

    (x_train, y_train), (x_val, y_val) = (
        (x_train[:num_training_examples], y_train[:num_training_examples]),
        (x_train[num_training_examples:], y_train[num_training_examples:]),
    )
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(config.buffer_size).batch(config.batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(config.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(config.batch_size)

    return (train_ds, val_ds, test_ds)
