from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs

from tensorflow.keras import layers

import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops

import util

class Generator():
    """
    Inspired by https://www.tensorflow.org/tutorials/generative/dcgan
    """
    def __init__(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, 512)))
        assert model.output_shape == (None, 8, 8, 512)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 64, 64, 3)

        self.model = model

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)

        return


    #@tf.function
    def __call__(self, image, training=False):
        logits = self.model(image, training=training)

        return logits

    @property
    def trainable_variables(self):
        vals = self.model.trainable_variables
        return vals

    def calc_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def optimizer(self):
        return self.optimizer



class Discriminator():
    """
    Inspired by https://www.tensorflow.org/tutorials/generative/dcgan
    """
    def __init__(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[64, 64, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        self.model = model

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)

        return

    def __call__(self, image, training=False):
        logits = self.model(image, training=training)

        return logits


    @property
    def trainable_variables(self):
        vals = self.model.trainable_variables
        return vals


    def calc_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def optimizer(self):
        return self.optimizer


