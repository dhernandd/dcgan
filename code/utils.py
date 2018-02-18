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

DTYPE = tf.float32

def variable_in_cpu(name, shape, initializer):
    """
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, dtype=DTYPE, 
                              initializer=initializer)
    return var


class FullLayer():
    """
    """
    def __init__(self, collections=None):
        """
        """
        self.collections = collections
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu}
        
    def __call__(self, Input, nodes, nl='softplus', scope=None):
        """
        """
        nonlinearity = self.nl_dict[nl]
        input_dim = Input.get_shape()[1]
        
        with tf.variable_scope(scope or 'fullL'):
            weights_full1 = variable_in_cpu('weights', [input_dim, nodes], 
                                  initializer=tf.orthogonal_initializer())
            biases_full1 = variable_in_cpu('biases', [nodes], 
                                     initializer=tf.zeros_initializer(dtype=tf.float64))
            full = nonlinearity(tf.matmul(Input, weights_full1) + biases_full1,
                                  name='output_'+nl)
        
        return full


class BatchNormalizationLayer():
    """
    """
    def __init__(self):
        """
        """
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu,
                        'lkyrelu' : lambda x : tf.maximum(x, 0.1*x)}
    
    def __call__(self, Input, momentum=0.9, eps=1e-5, scope=None, nl='relu'):
        """
        """
        nonlinearity = self.nl_dict[nl]
        with tf.variable_scope(scope or 'bnL'):
            bn = nonlinearity(tf.contrib.layers.batch_norm(Input, decay=momentum, epsilon=eps,
                                                           scale=True) )
            return tf.identity(bn, name='batch_norm')


class ConvLayer():
    """
    """
    def __init__(self, collections=None):
        """
        """
        self.collections = collections
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu,
                        'tanh' : tf.nn.tanh, 'lkyrelu' : lambda x : tf.maximum(x, 0.1*x)}

    
    def __call__(self, Input, num_filters_out, kernel_height=5, kernel_width=5, strides_height=2,
                 strides_width=2, scope=None, stddev=0.02, nl='relu'):
        """
        """
        nonlinearity = self.nl_dict[nl]
        
        kernel_shape = [kernel_height, kernel_width, Input.get_shape()[-1], num_filters_out]
        with tf.variable_scope(scope or 'convL'):
            kernel = tf.get_variable(name='kernel', shape=kernel_shape,
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(Input, kernel, strides=[1, strides_height, strides_width, 1],
                                padding='SAME')
            bias = tf.get_variable('bias', [num_filters_out], initializer=tf.constant_initializer(0.0))
            
            conv = nonlinearity(tf.nn.bias_add(conv, bias))
            
        return tf.identity(conv, name='conv_'+nl)
    
# def conv2d(input_, num_filters_out, 
#        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#        name="conv2d"):
#   with tf.variable_scope(name):
#     w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], num_filters_out],
#               initializer=tf.truncated_normal_initializer(stddev=stddev))
#     conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
# 
#     biases = tf.get_variable('biases', [num_filters_out], initializer=tf.constant_initializer(0.0))
#     conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
# 
#     return conv

class ConvTransposeLayer():
    """
    """
    def __init__(self, collections=None):
        """
        """
        self.collections = collections
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu,
                        'tanh' : tf.nn.tanh, 'sigmoid' : tf.nn.sigmoid,
                        'lkyrelu' : lambda x : tf.maximum(x, 0.1*x)}

    
    def __call__(self, Input, output_shape_hxwxm, kernel_height=5, kernel_width=5, strides_height=2,
                 strides_width=2, scope=None, stddev=0.02, nl='relu'):
        """
        """
        nonlinearity = self.nl_dict[nl]
        
        # TODO: Support some old versions of tensorflow?
        kernel_shape = [kernel_height, kernel_width, output_shape_hxwxm[-1], Input.get_shape()[-1]]
        with tf.variable_scope(scope or 'convtL'):
            kernel = tf.get_variable(name='kernel', shape=kernel_shape,
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv_traspose = tf.nn.conv2d_transpose(Input, kernel, output_shape=output_shape_hxwxm,
                                                   strides=[1, strides_height, strides_width, 1])
            bias = tf.get_variable('bias', [output_shape_hxwxm[-1]],
                                   initializer=tf.constant_initializer(0.0))
            conv_traspose = nonlinearity(tf.nn.bias_add(conv_traspose, bias))
            
        return tf.identity(conv_traspose, name='convt_'+nl)

#     biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#     deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())


            
        


