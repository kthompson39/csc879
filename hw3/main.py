from __future__ import print_function
import sys
#print(sys.executable)

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
import datetime

import model 

import util

def main():
    # grab data
    original_train_ds, val_ds, test_ds = util.load_data()

    # grab model
    gen = model.Generator()
    disc = model.Discriminator()

    # visualize model
    # util.model_visual(gen,disc)

    early_stop = util.EarlyStopping(5)
    
    gen_losses = []
    disc_losses = []
    input_noise = tf.random.normal([16, 100])

    # train model completely
    for epoch in range(10):
        train_ds = util.augment_data(original_train_ds)

        gen_loss, disc_loss = train_gan(train_ds, gen, disc)
        #validation_loss = validate_gan(val_ds, model)

        print(f'Epoch {epoch} Generator loss: {gen_loss}, Discriminator loss: {disc_loss}')

        # test model
        #test_accuracy = test_model(test_ds, model)
        #print("accuracy:", test_accuracy)

        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)


        util.generate_and_save_images(gen, epoch, input_noise)

        # check for early stopping
        if early_stop.check(gen_loss):
            break
        
    # test model
    #test_accuracy = test_model(test_ds, model)
    #print("Overall model accuracy on test dataset:", test_accuracy)

    file_name = 'model' + str(datetime.datetime.now()).replace(' ', '-').replace(':','_')

    util.graph_info(file_name, gen_losses, disc_losses)




def train_gan(train_ds, gen, disc):
    gen_loss_values = []
    disc_loss_values = []
    
    for batch in tqdm(train_ds):
        noise = tf.random.normal([batch.shape[0], 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # run network
            generated_images = gen(noise, training=True)

            real_output = disc(batch, training=True)
            fake_output = disc(generated_images, training=True)

            gen_loss = gen.calc_loss(fake_output)
            disc_loss = disc.calc_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

            gen.optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
            disc.optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
        
        gen_loss_values.append(gen_loss)
        disc_loss_values.append(disc_loss)

    gen_loss = tf.math.reduce_mean(gen_loss_values).numpy()
    disc_loss = tf.math.reduce_mean(disc_loss_values).numpy()

    return gen_loss, disc_loss


def validate_gan(val_ds, gan, disc):
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
