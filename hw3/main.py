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
    original_train_ds, test_ds = util.load_data()

    # grab model
    gen = model.Generator()
    disc = model.Discriminator()

    # visualize model
    # util.model_visual(gen,disc)

    early_stop = util.EarlyStopping(5)
    
    gen_losses = []
    disc_losses = []
    fid_losses = []
    input_noise = tf.random.normal([16, 100])

    # train model completely
    for epoch in range(100):
        train_ds = util.augment_data(original_train_ds)

        training_gen_loss, training_disc_loss, _ = run_gan(train_ds, gen, disc, training=True)

        val_gen_loss, val_disc_loss, val_fid_loss = run_gan(test_ds, gen, disc, training=False)

        print(f'Epoch {epoch} Generator loss: {val_gen_loss}, Discriminator loss: {val_disc_loss}, FID loss: {val_fid_loss}')

        # test model
        #test_accuracy = test_model(test_ds, model)
        #print("accuracy:", test_accuracy)

        gen_losses.append(val_gen_loss)
        disc_losses.append(val_disc_loss)
        fid_losses.append(val_fid_loss)

        util.generate_and_save_images(gen, epoch, input_noise)

        # check for early stopping
        if early_stop.check(val_fid_loss):
            pass
            #break
        
    # test model
    #test_accuracy = test_model(test_ds, model)
    #print("Overall model accuracy on test dataset:", test_accuracy)

    file_name = 'model' + str(datetime.datetime.now()).replace(' ', '-').replace(':','_')

    util.graph_info(file_name, gen_losses, disc_losses)
    util.graph_fid(file_name + "_fid", fid_losses)



def run_gan(ds, gen, disc, training=False):
    gen_loss_values = []
    disc_loss_values = []
    fid_loss_values = []
    images = []
    
    i = 1
    for batch in tqdm(ds):
        batch = util.normalize_images(batch)
        noise = tf.random.normal([batch.shape[0], 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # run network
            generated_images = gen(noise, training=training)

            real_output = disc(batch, training=training)
            fake_output = disc(generated_images, training=training)

            gen_loss = gen.calc_loss(fake_output)
            disc_loss = disc.calc_loss(real_output, fake_output)


            if training:
                gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

                gen.optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
                disc.optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
        
        if (not training) and i == len(ds):
            fid_loss = util.calc_fid(batch , generated_images)
            fid_loss_values.append(fid_loss)

        i+=1

        gen_loss_values.append(gen_loss)
        disc_loss_values.append(disc_loss)
        

    gen_loss = tf.math.reduce_mean(gen_loss_values).numpy()
    disc_loss = tf.math.reduce_mean(disc_loss_values).numpy()
    fid_loss = tf.math.reduce_mean(fid_loss_values).numpy()

    return gen_loss, disc_loss, fid_loss


if __name__ == '__main__':
    main()
