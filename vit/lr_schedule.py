import numpy as np
import tensorflow as tf
from tensorflow import keras


# Reference: https://keras.io/examples/vision/shiftvit/#learning-rate-schedule
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, config, warmup_steps, total_steps):
        super().__init__()
        self.config = config
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.pi = tf.constant(np.pi)

    def get_config(self):
        config = dict()
        config.update(self.config)
        config.update({"warmup_steps": self.warmup_steps})
        config.update({"total_steps": self.total_steps})
        return config

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError(
                f"Total number of steps {self.total_steps} must be"
                + f"larger or equal to warmup steps {self.warmup_steps}."
            )
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        )
        learning_rate = 0.5 * self.config.lr_max * (1 + cos_annealed_lr)
        if self.warmup_steps > 0:
            if self.config.lr_max < self.config.lr_start:
                raise ValueError(
                    f"lr_start {self.config.lr_start} must be smaller or"
                    + f"equal to lr_max {self.config.lr_max}."
                )
            slope = (
                self.config.lr_max - self.config.lr_start
            ) / self.warmup_steps
            warmup_rate = (
                slope * tf.cast(step, tf.float32) + self.config.lr_start
            )
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
