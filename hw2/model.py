from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops

import util

class Attention_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention_Layer, self).__init__()

        self.attention = tf.keras.layers.Attention()
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(128)
        self.add = tf.keras.layers.Add()

        #self.is_built = False
        return

    #def build(self):
        #self.dense = tf.keras.layers.dense(2)
        #self.is_built = True


    def call(self, x):
        #if not self.is_built:
            #self.build()
        query, value = x
        x = self.attention([query, value])
        z = self.add([x,value])
        z = self.norm1(x)
        z_shape = z.shape.as_list()
        x = tf.reshape(z, [-1,z_shape[1]])
        v = self.dense(x)
        v = tf.reshape(v, z_shape)
        v = self.add([v,z])
        v = self.norm2(v)

        return v


class Model1():
    def __init__(self, vectorize_layer, embedding_layer):
        self.query_layer = tf.keras.layers.Conv1D(
                    filters=100,
                    kernel_size=4,
                    padding='same')
        self.value_layer = tf.keras.layers.Conv1D(
                    filters=100,
                    kernel_size=4,
                    padding='same')

        self.concat = tf.keras.layers.Concatenate()

        self.att = [Attention_Layer() for i in range(5)]

        #self.output_layer1 = tf.keras.layers.Dense(256)
        #self.output_layer2 = tf.keras.layers.Dense(32)
        self.output_layer3 = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

        self.embedding_layer = embedding_layer
        self.vectorize_layer = vectorize_layer

        return


    #@tf.function
    def __call__(self, text):
        vec = self.vectorize_layer(text)
        embeddings = self.embedding_layer(vec)
        query = self.query_layer(embeddings)
        value = self.value_layer(embeddings)

        for cell in self.att:
            value = cell([query, value])

        attention_values = self.concat([query, value])
        shape = attention_values.shape.as_list()
        x = tf.reshape(attention_values, [shape[0], -1])
        #y = self.output_layer1(x)
        #z = self.output_layer2(y)
        logits = self.output_layer3(x)

        return logits

    def summary(self):
        #self.conv_classifier.summary()
        pass

    @property
    def layers(self):
        return self.conv_classifier.layers

    @property
    def trainable_variables(self):
        vals = self.concat.trainable_variables
        vals += self.vectorize_layer.trainable_variables
        vals += self.embedding_layer.trainable_variables
        vals += self.query_layer.trainable_variables
        vals += self.value_layer.trainable_variables
        #vals += self.output_layer1.trainable_variables
        #vals += self.output_layer2.trainable_variables
        vals += self.output_layer3.trainable_variables
        for cell in self.att:
            vals += cell.trainable_variables
        return vals

    def calc_loss(self, logits, labels):
        # convert labels to one-hot vectors
        labels = util.convert_labels_to_onehot(labels, 2) 

        return util.batch_cross_entropy(labels, logits)


class Model2():
    def __init__(self, vectorize_layer, embedding_layer):
        self.query_layer = tf.keras.layers.Conv1D(
                    filters=100,
                    kernel_size=4,
                    padding='same')
        self.value_layer = tf.keras.layers.Conv1D(
                    filters=100,
                    kernel_size=4,
                    padding='same')
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True, activation = tf.nn.tanh, kernel_initializer = 'random_normal'  )

        self.attention = tf.keras.layers.Attention()
        self.concat = tf.keras.layers.Concatenate()

        self.lstm2 = tf.keras.layers.LSTM(32, activation = tf.nn.relu, kernel_initializer = 'random_normal'  )

        self.output_layer = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

        self.embedding_layer = embedding_layer
        self.vectorize_layer = vectorize_layer

        return


    #@tf.function
    def __call__(self, text):
        vec = self.vectorize_layer(text)
        embeddings = self.embedding_layer(vec)
        query = self.query_layer(embeddings)
        value = self.value_layer(embeddings)

        value = self.attention([query, value])

        attention_values = self.concat([query, value])
    
        x = self.lstm(attention_values)
        x = self.lstm2(x)
        #shape = attention_values.shape.as_list()
        #x = tf.reshape(attention_values, [shape[0], -1])

        logits = self.output_layer(x)

        return logits

    def summary(self):
        #self.conv_classifier.summary()
        pass

    @property
    def layers(self):
        return self.conv_classifier.layers

    @property
    def trainable_variables(self):
        vals = self.concat.trainable_variables
        vals += self.vectorize_layer.trainable_variables
        vals += self.embedding_layer.trainable_variables
        vals += self.query_layer.trainable_variables
        vals += self.value_layer.trainable_variables
        vals += self.output_layer.trainable_variables
        vals += self.lstm.trainable_variables
        vals += self.lstm2.trainable_variables
        vals += self.attention.trainable_variables
        return vals

    def calc_loss(self, logits, labels):
        # convert labels to one-hot vectors
        labels = util.convert_labels_to_onehot(labels, 2) 

        return util.batch_cross_entropy(labels, logits)

