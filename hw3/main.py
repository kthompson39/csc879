from __future__ import print_function
import sys
print(sys.executable)

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
import datetime

from model import Model1
from model import Model2

import util

def main():
    # grab data
    train_ds, val_ds, test_ds = util.load_data()

    vectorize_layer = util.create_vectorize_layer(train_ds)
    embedding_layer = util.create_embedding_layer(vectorize_layer)
    
    # grab model
    model = Model1(vectorize_layer, embedding_layer)
    model2 = Model2(vectorize_layer, embedding_layer)
    

    early_stop = util.EarlyStopping(5)
    
    train_losses = []
    val_losses = []
    accuracy_values = []
    # train model2 completely
    for epoch in range(100):
        train_loss = train_model(train_ds, model2)
        validation_loss = validate_model(val_ds, model2)

        print(f'Epoch {epoch} loss: {validation_loss}')# (train loss: {tf.reduce_mean(train_loss)})')

        # test model 2
        test_accuracy = test_model(test_ds, model2)
        print("accuracy:", test_accuracy)

        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        accuracy_values.append(test_accuracy)

        # check for early stopping
        if early_stop.check(validation_loss):
            break
    # test model 2
    test_accuracy = test_model(test_ds, model2)
    print("Overall model2 accuracy on test dataset:", test_accuracy)
    #print(model.summary())

    util.graph_info('model' + str(datetime.datetime.now()).replace(' ', '-').replace(':','_'), train_losses, val_losses, accuracy_values)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.000005)

def train_model(train_ds, model) -> float:
    loss_values = []
    for batch in tqdm(train_ds):
        with tf.GradientTape() as tape:
            # run network
            loss = calc_batch_loss(batch, model)
        loss_values.append(loss)

        # gradient update
        train_vars = model.trainable_variables
        grads = tape.gradient(loss, train_vars)
        optimizer.apply_gradients(zip(grads, train_vars))

    return tf.math.reduce_mean(loss_values).numpy()


def validate_model(val_ds, model):
    loss = []
    for batch in val_ds:
        # calculate accuracy with validation set
        batch_loss = calc_batch_loss(batch, model)
        loss.append(tf.math.reduce_mean(batch_loss).numpy())

    return tf.math.reduce_mean(loss).numpy()


def test_model(test_ds, model) -> float:
    loss = []
    for batch in tqdm(test_ds):
        x = batch['text'] 
        logits = model(x)
        labels = batch['label']
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        loss.append(accuracy)
    
    return tf.math.reduce_mean(loss).numpy()


def calc_batch_loss(batch_data, model):
    x = batch_data['text'].numpy() # get data

    labels = batch_data['label']
    logits = model(x)
    # calculate loss
    loss = model.calc_loss(logits, labels)

    return loss



if __name__ == '__main__':
    main()
