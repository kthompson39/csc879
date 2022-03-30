import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import numpy as np
import matplotlib.pyplot as plt
import visualkeras
from PIL import ImageFont

DATA_DIR = '$WORK/tensorflow-datasets/'
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, shuffle=False):
    if shuffle:
      ds = ds.shuffle(1024)
  
    # Batch all datasets.
    ds = ds.batch(batch_size)
  
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)
  

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical")
    #tf.keras.layers.RandomRotation(0.2),
])

def augment_data(ds):
    # Use data augmentation only on the training set.
    ds = ds.map(lambda x: (data_augmentation(x, training=True)),
              num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)



def load_data():

    #full_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    #'/work/cse479/kthompson/cat_dataset/', image_size=(64,64), label_mode=None
                #)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    '/work/cse479/kthompson/cat_dataset/dataset-part2/', image_size=(64,64), label_mode=None
                )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    '/work/cse479/kthompson/cat_dataset/dataset-part1/', image_size=(64,64), label_mode=None
                )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    '/work/cse479/kthompson/cat_dataset/dataset-part3/', image_size=(64,64), label_mode=None
                )

    return train_ds, val_ds, test_ds



def convert_labels_to_onehot(labels, total_labels):
    """
    given a list of expected labels, convert to list of list of one
    hot vectors.
    ex:
        labels = [0,2]; total_labels = 3
        returns [[1,0,0],[0,0,1]]
    """
    one_hots = []
    for label in labels:
        temp = np.zeros((total_labels,))
        temp[label] = 1.
        one_hots.append(temp)
                                                                                    
    return one_hots

def batch_cross_entropy(labels, logits):
    total = 0
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # batch size
    for batch in range(len(logits)):
        total += bce(labels[batch], logits[batch])

    return total

def graph_info(fig_name, gen_loss, disc_loss):
    fig, ax = plt.subplots()

    x = np.arange(1,len(gen_loss)+1)

    ax.set_title("Loss While Training")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Generator Loss")

    lns1 =ax.plot(x, gen_loss,     'r-', linewidth=1.2, label='Generator Loss')

    ax2 = ax.twinx()
    lns3 = ax2.plot(x, disc_loss,'g-', linewidth=1.0, label='Discriminator Loss')

    ax2.tick_params(axis ='y', labelcolor = 'g')
    ax2.set_ylabel("Discriminator Loss", color = 'g')

    lns = lns1+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.savefig(fig_name + '.png', format='png')


def generate_and_save_images(model, epoch, test_input):
    # Notice atraining` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)

        #cast all channels to integer [0...255]
        img = predictions[i]*127.5 + 127.5
        img = tf.cast(img, dtype=tf.int32)

        plt.imshow(img, interpolation='nearest')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def model_visual(gen,disc):
    # font = ImageFont.truetype("arial.ttf", 32)
    font = ImageFont.load_default()
    visualkeras.layered_view(gen.model, legend=True, font=font, to_file='genModel.png') 
    visualkeras.layered_view(disc.model, legend=True, font=font, to_file='discModel.png') 

class EarlyStopping:
    """
    Class to determine when to stop training
    Taken from Hackathon #3
    """
    def __init__(self, patience=5, epsilon=1e-4):
        """
        Args:
            patience (int): how many epochs of not improving before stopping training
            epsilon (float): minimum amount of improvement required to reset counter
        """
        self.patience = patience
        self.epsilon = epsilon
        self.best_loss = float('inf')
        self.epochs_waited = 0

    def __str__(self):
        return "Early stopping has waited {} epochs out of {} at loss {}".format(self.epochs_waited, self.patience, self.best_loss)

    def check(self, loss):
        """
        Call after each epoch to check whether training should halt

        Args:
            loss (float): loss value from the most recent epoch of training

        Returns:
            True if training should halt, False otherwise
        """
        if loss < (self.best_loss - self.epsilon):
            self.best_loss = loss
            self.epochs_waited = 0
            return False
        else:
            self.epochs_waited += 1
            return self.epochs_waited > self.patience

