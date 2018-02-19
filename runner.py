# This repo is a heavily modified version of Taehoon Kim's
#
# https://github.com/carpedm20/DCGAN-tensorflow
#
# for my own exploratory purposes. Visit his github page for it is golden!


import numpy as np

import tensorflow as tf

from code.dcgan import DCGAN

flags = tf.app.flags

# 1. DIRECTORIES, SAVE FILES
IS_DATED_DATA = False
DATASET = 'mnist'
CKPT_DIR = None

# 2. DCGAN INSTANCE CONFIGURATION
ZDIM = 128
DATA_HEIGHT = 28
DATA_WIDTH = 28
YDIM = 10
BATCH_SIZE = 64

# 3. TRAINING CONFIGURATION
LEARNING_RATE = 2e-4
BETA1 = 0.5
NUM_EPOCHS = 1

# 1.
flags.DEFINE_boolean('is_dated_data', IS_DATED_DATA, "Should I add the date to the data dir?")
flags.DEFINE_string('dataset', DATASET, "The name of the dataset. Assumes the directory \
                    structure ./data/[dataset]")
flags.DEFINE_string('ckpt_dir', CKPT_DIR, "The directory to store the checkpoints")

# 2.
flags.DEFINE_integer('zDim', ZDIM, "The dimension of the latent representation")
flags.DEFINE_integer('data_height', DATA_HEIGHT, "The y-dimension of the data")
flags.DEFINE_integer('data_width', DATA_WIDTH, "The x-dimension of the data. Set equal to \
                    `data_height if None")
flags.DEFINE_integer('yDim', YDIM, "The number of categories in the data. Set to None if \
                    uncategorized")
flags.DEFINE_integer('batch_size', BATCH_SIZE, "The batch size duh")

# 3.
flags.DEFINE_float('learning_rate', LEARNING_RATE, "The Adam learning rate")
flags.DEFINE_float('beta1', BETA1, "The beta1 parameter of the Adam optimization")
flags.DEFINE_integer('num_epochs', NUM_EPOCHS, "The number of epochs to run the algorithm")

params = flags.FLAGS


def main(_):
    dcgan = DCGAN(zDim=params.zDim,
                  data_height=params.data_height,
                  data_width=params.data_width,
                  dataset=params.dataset,
                  ckpt_dir=params.ckpt_dir,
                  yDim=params.yDim,
                  batch_size=params.batch_size)
    dcgan.train(params)
    

if __name__ ==  '__main__':
    tf.app.run()
