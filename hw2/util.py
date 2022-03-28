import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = './tensorflow-datasets/'
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, shuffle=False):
    if shuffle:
      ds = ds.shuffle(1024)
  
    # Batch all datasets.
    ds = ds.batch(batch_size)
  
    # Use buffered prefetching on all datasets.
    return ds#.prefetch(buffer_size=AUTOTUNE)
  

def load_data():
    (train_ds, val_ds, test_ds) = tfds.load(
        'imdb_reviews',
        split=['train[:90%]', 'train[90%:]', 'test'],
    )

    global full_training_ds
    full_training_ds = train_ds.map(lambda x: x['text'])

    train_ds = prepare(train_ds, shuffle=True)
    val_ds = prepare(val_ds)
    test_ds = prepare(test_ds)

    return train_ds, val_ds, test_ds


def create_vectorize_layer(train_text):
    MAX_SEQ_LEN = 128
    MAX_TOKENS = 5000

    #ds = tfds.load('imdb_reviews', data_dir=DATA_DIR)


    # Create TextVectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode='int',
        output_sequence_length=MAX_SEQ_LEN)

    # Use `adapt` to create a vocabulary mapping words to integers
    #train_text = ds['train'].map(lambda x: x['text'])
    global full_training_ds
    vectorize_layer.adapt(full_training_ds)

    return vectorize_layer


def create_embedding_layer(vectorize_layer):
    VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
    EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))

    embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)

    return embedding_layer


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

