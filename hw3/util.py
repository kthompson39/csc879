import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import numpy as np
import matplotlib.pyplot as plt

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

def graph_info(fig_name, train_loss, validation_loss, accuracy):
    fig, ax = plt.subplots()
    accuracy = np.array(accuracy) * 100

    x = np.arange(1,len(train_loss)+1)

    ax.set_title("Cross Entropy Loss While Training")
    ax.set_xlabel("Epochs")
    #ax.set_xticks(x)
    ax.set_ylabel("Cross Entropy Loss")

    lns1 =ax.plot(x, validation_loss,     'r-', linewidth=1.2, label='Validation set Loss')
    lns2 =ax.plot(x, train_loss,   'b-', linewidth=1.2, label='Training set Loss')

    ax2 = ax.twinx()
    lns3 = ax2.plot(x, accuracy,'g-', linewidth=1.0, label='Testing set Accuracy')

    ax2.tick_params(axis ='y', labelcolor = 'g')
    ax2.set_ylabel("Accuracy (%)", color = 'g')
    ax2.set(ylim=(0, 100))

    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.savefig(fig_name + '.png', format='png')



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

