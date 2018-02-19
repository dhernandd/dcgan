# Copyright 2018 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
from __future__ import print_function
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from code.dcgan import Discriminator

class DiscriminatorTest(tf.test.TestCase):
    """
    """
    output_height = 28
    output_width = 28
    
    yDim = 10
    batch_size = 64
    num_data_channels = 1
    
    graph1 = tf.Graph()
    with graph1.as_default():
        X = tf.placeholder(tf.float32, [batch_size, output_height, output_width, num_data_channels],
                           name='X')
        Y = tf.placeholder(tf.float32, [batch_size, yDim], name='Y')
        disc_withY = Discriminator(X, Y=Y, num_exitnodes=512)
        
    def test_disc_nn(self):
        """
        Simple test for the correctness of the Discriminator NN.
        """
        with tf.Session(graph=self.graph1) as sess:
            sampleX = np.random.randn(self.batch_size, self.output_height,
                                      self.output_width, self.num_data_channels)
            sampleY = np.random.randn(self.batch_size, self.yDim)
            sess.run(tf.global_variables_initializer())
            out = sess.run(self.disc_withY.disc_trueprobs, feed_dict={'X:0' : sampleX,
                                                                      'Y:0' : sampleY})
            assert out.shape == (self.batch_size, 1)
            print('A probability per sample -> (batch_size, 1):', out.shape)
            
    def test_add_disc_output_to_graph(self):
        """
        Simple test that shows how to augment the computational graph reusing
        the discriminator subgraph
        """
        graph = self.graph1
        with graph.as_default():
            bsize = 64
            Xnew = tf.placeholder(tf.float32,
                                  [bsize, self.output_height, self.output_width, self.num_data_channels],
                                  'Xnew')
            Ynew = tf.placeholder(dtype=tf.float32, shape=[bsize, self.yDim], name='Ynew')
            new_disc_probs, _ = self.disc_withY.get_discriminator_output(Xnew, Ynew)
        with tf.Session(graph=graph) as sess:
            sampleX = np.random.randn(self.batch_size, self.output_height,
                                      self.output_width, self.num_data_channels)
            sampleY = np.random.randn(self.batch_size, self.yDim)
            sess.run(tf.global_variables_initializer())
            out = sess.run(new_disc_probs, feed_dict={'Xnew:0' : sampleX,
                                                      'Ynew:0' : sampleY})
            assert out.shape == (self.batch_size, 1)
            print('A probability per sample -> (batch_size, 1):', out.shape)
            

if __name__ == '__main__':
    tf.test.main()



