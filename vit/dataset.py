# import the necessary packages
import tensorflow as tf
from tensorflow import keras


def get_cifar_dataset(config):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_val, y_val) = (
        (x_train[:40000], y_train[:40000]),
        (x_train[40000:], y_train[40000:]),
    )
    AUTO = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = (
        train_ds.shuffle(config.buffer_size)
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(AUTO)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(config.batch_size, drop_remainder=True).prefetch(AUTO)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(config.batch_size, drop_remainder=True).prefetch(AUTO)

    return (train_ds, val_ds, test_ds)