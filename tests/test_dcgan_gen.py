# This code is a heavily modified version of Taehoon Kim's
#
# https://github.com/carpedm20/DCGAN-tensorflow
#
# for my own exploratory purposes. Visit his github page for it is golden!
from __future__ import print_function
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from code.dcgan import Generator

class GeneratorTest(tf.test.TestCase):
    """
    """
    output_height = 28
    output_width = 28
    
    yDim = 10
    zDim = 200
    batch_size = 64
    num_data_channels = 1
    graph1 = tf.Graph()
    with graph1.as_default():
        Z = tf.placeholder(tf.float32, [None, zDim], 'Z')
        gen = Generator(Z, output_height, output_width, num_data_channels=num_data_channels,
                        batch_size=batch_size)
    
    graph2 = tf.Graph()
    with graph2.as_default():
        Z = tf.placeholder(tf.float32, [None, zDim], 'Z')
        Y = tf.placeholder(tf.float32, [batch_size, yDim], name='Y')
        gen_withY = Generator(Z, output_height, output_width, Y=Y, num_data_channels=num_data_channels,
                        batch_size=batch_size)
    
    def test_gen_nn(self):
        """
        """
#         with tf.Session(graph=self.graph1) as sess:
#             sampleZ = np.random.randn(1, self.zDim)
#             sess.run(tf.global_variables_initializer())
#             out = sess.run(self.gen.gen_output, feed_dict={'Z:0' : sampleX})
#             print(out.shape)
        with tf.Session(graph=self.graph2) as sess:
            sampleZ = np.random.randn(self.batch_size, self.zDim)
            sampleY = np.random.randn(self.batch_size, self.yDim)
            sess.run(tf.global_variables_initializer())
            out = sess.run(self.gen_withY.gen_output, feed_dict={'Z:0' : sampleZ,
                                                                 'Y:0' : sampleY})
            print(out.shape)
            
        

if __name__ == '__main__':
    tf.test.main()
        