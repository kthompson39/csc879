import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import numpy as np
import matplotlib.pyplot as plt
import visualkeras
from PIL import ImageFont
import scipy

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
    tf.keras.layers.RandomFlip("vertical")
    #tf.keras.layers.RandomRotation(0.2),
])

def augment_data(ds):
    # Use data augmentation only on the training set.
    ds = ds.map(lambda x: (data_augmentation(x, training=True)),
              num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)

def normalize_dataset(ds):
    return ds.map(lambda x: (normalize_images(x)), num_parallel_calls=AUTOTUNE)
    

def normalize_images(images):
    #normailze RGB channels to be [-1,1]
    norm_img = (images - 127.5) / 127.5
    return norm_img


inception_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                              #weights="imagenet",
                              pooling='avg',
                              input_shape=(75,75,3))

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, eps=1E-16):
    # load inception v3 model
    model = tf.keras.applications.inception_v3.InceptionV3()
    
    #subset = tf.keras.applications.inception_v3.preprocess_input(img)
    subset = images

    # scale images to the required size
    subset = tf.image.resize(subset, [299,299], method='nearest')
    # predict p(y|x)
    p_yx = model.predict(subset)
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # calculate KL divergence using log probabilities
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the log
    is_score = exp(avg_kl_d)

    # average across images
    #is_avg, is_std = mean(scores), std(scores)
    return is_score


# Running mean and std
class Welford():
    def __init__(self,a_list=None):
        self.n = 0
        self.M = 0
        self.S = 0

    def update(self, x):
        self.n += 1

        newM = self.M + (x - self.M) / self.n
        newS = self.S + (x - self.M) * (x - newM)

        self.M = newM
        self.S = newS

    @property
    def mean(self):
        return self.M

    @property
    def std(self):
        if self.n == 1:
            return 0
        return math.sqrt(self.S / (self.n - 1))



def calc_fid(real_images, generated_images):
    images1 = tf.image.resize(real_images, [75,75], method='nearest')
    images2 = tf.image.resize(generated_images, [75,75], method='nearest')
    #breakpoint()

    # calculate activations
    act1 = inception_model.predict(images1)
    act2 = inception_model.predict(images2)

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def load_data():

    #full_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    #'/work/cse479/kthompson/cat_dataset/', image_size=(64,64), label_mode=None
                #)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    '/work/cse479/kthompson/cat_dataset/dataset-part2/', image_size=(64,64), label_mode=None
                )

    # val_ds broken?
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    '/work/cse479/kthompson/cat_dataset/dataset-part1/', image_size=(64,64), label_mode=None
                )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    '/work/cse479/kthompson/cat_dataset/dataset-part3/', image_size=(64,64), label_mode=None, validation_split=0.5, subset="training", seed=1
                )
    test_2_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    '/work/cse479/kthompson/cat_dataset/dataset-part3/', image_size=(64,64), label_mode=None, validation_split=0.5, subset="validation", seed=1
                )

    #train_ds = normalize_dataset(train_ds)
    #val_ds = normalize_dataset(train_ds)
    #test_ds = normalize_dataset(train_ds)

    train_ds = train_ds.concatenate(test_2_ds)

    return train_ds, test_ds



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

    plt.savefig('image4_at_epoch_{:04d}.png'.format(epoch))
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

