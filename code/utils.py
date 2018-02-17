# Copyright 2017 Daniel Hernandez Diaz, Columbia University
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

import numpy as np

import tensorflow as tf


def variable_in_cpu(name, shape, initializer):
    """
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, dtype=tf.float64, 
                              initializer=initializer)
    return var


class FullLayer():
    """
    """
    def __init__(self, collections=None):
        self.collections = collections
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax}
        
    
    def __call__(self, Input, nodes, input_dim, nl='softplus', scope=None):
        nonlinearity = self.nl_dict[nl]
        
        with tf.variable_scope(scope or 'full_L'):
            weights_full1 = variable_in_cpu('weights', [input_dim, nodes], 
                                  initializer=tf.orthogonal_initializer())
            biases_full1 = variable_in_cpu('biases', [nodes], 
                                     initializer=tf.zeros_initializer(dtype=tf.float64))
            full = nonlinearity(tf.matmul(Input, weights_full1) + biases_full1,
                                  name='output')
        
        return full


class BatchNormalizationLayer():
    """
    """
    def __init__(self):
        """
        """
        pass
    
    def __call__(self, Input, momentum=0.9, eps=1e-5, scope=None):
        with tf.variable_scope(scope or 'bn_L'):
            return tf.contrib.layers.batch_norm(Input, decay=momentum, epsilon=eps,
                      scale=True)
            
        


