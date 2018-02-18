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

from code.dcgan import Generator

class GeneratorTest(tf.test.TestCase):
    """
    """
    height = 50
    width = 50
    
    xDim = 200
    cDim = 3
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None, xDim], 'X')
        gen = Generator(X, height, width, cDim=cDim)
    
    def test_gen_nn(self):
        """
        """
        with tf.Session(graph=self.graph) as sess:
            sampleX = np.random.randn(1, self.xDim)
            sess.run(tf.global_variables_initializer())
            out = sess.run(self.gen.gen_output, feed_dict={'X:0' : sampleX})
            print(out.shape)
            
        

if __name__ == '__main__':
    tf.test.main()
        