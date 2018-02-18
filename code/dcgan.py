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
import os

import numpy as np
import tensorflow as tf

from code.utils import FullLayer, ConvTransposeLayer, BatchNormalizationLayer, ConvLayer

import math
from click.core import batch

def conv_output_size(isize, ksize=None, padding='SAME', stride=1, transpose=False,
                     a=0):
    """
    Computes the output size over one dimension of a CNN. 
    
    Essentially follows 'A guide to convolution arithmetic for deep learning'
    https://arxiv.org/abs/1603.07285 
    """
    if padding == 'SAME':
        if not transpose:
            return math.ceil(isize/stride)
        else:
            # TODO: Check this one
            return isize*stride + a
    else:
        if not isinstance(padding, int):
            raise ValueError('padding must be "SAME" or an integer type')
        if ksize is None:
            raise ValueError('for padding different than "SAME", you must specify a kernel size')
        if not transpose:
            if isize + 2*padding - ksize <= 0:
                raise ValueError('the provided combination of input size, padding and \
                                    kernel size leads to negative size output. Check kernel size')
            return math.ceil(isize + 2*padding - ksize/stride)
        else:
            if stride*(isize-1) + ksize - 2*padding <= 0:
                raise ValueError('the provided combination of input size, padding and \
                                    kernel size leads to negative size output. Check padding')
            return stride*(isize-1) + a + ksize - 2*padding
        
    

    
        
    
    

class Discriminator():
    """
    """
    def __init__(self, Y, num_initfilters=64, batch_size=1):
        """
        """
        self.num_initfilters = num_initfilters
        self.batch_size = batch_size
        self.yDim = None
        
        self.Y = Y
        with tf.variable_scope('discriminator'):
            self.disc_trueprobs, self.disc_truelogits = self.get_discriminator_output()
            
            
    def get_discriminator_output(self, Input=None):
        """
        """
        if Input is None:
            Input = self.Y
            
        num_initfilters = self.num_initfilters
        if not self.yDim:
            # Define the Discriminator Network
            conv_layer = ConvLayer()
            batch_norm_layer = BatchNormalizationLayer()
            fully_connected_layer = FullLayer()
            
            conv1 = conv_layer(Input, num_initfilters, scope='conv1', nl='lkyrelu')
            
            conv2 = conv_layer(conv1, 2*num_initfilters, scope='conv2', nl='linear')
            bn2 = batch_norm_layer(conv2, scope='bn2', nl='lkyrelu') 

            conv3 = conv_layer(bn2, 4*num_initfilters, scope='conv3', nl='linear')
            bn3 = batch_norm_layer(conv3, scope='bn3', nl='lkyrelu') 

            conv4 = conv_layer(bn3, 8*num_initfilters, scope='conv4', nl='linear')
            bn4 = batch_norm_layer(conv4, scope='bn4', nl='lkyrelu') 
            
            full5 = fully_connected_layer(tf.reshape(bn4, [self.batch_size, -1]), 1, scope='full5')
            
            return tf.nn.sigmoid(full5), full5
        else:
            pass
        

class Sampler():
    """
    """
    def __init__(self):
        pass
    

class DCGAN():
    """
    """
    def __init__(self, gen_zDim, input_height=108, input_width=108, output_height=64, 
                 output_width=64, dataset='smnist', ckpt_dir=None, batch_size=1,
                 yDim=None):
        """
        """
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        
        self.gen_zDim = gen_zDim
        self.batch_size = batch_size
        self.yDim = yDim
        

        self.dataset = dataset
        self.ckpt_dir = ckpt_dir
        if dataset == 'mnist':
            self.Ydata, self.Xdata = self.load_mnist()
            self.num_channels = num_channels = self.Ydata.shape[-1]
        else:
            # TODO:
            pass
        image_dims = [self.input_height, self.input_width, num_channels]

        self.graph = graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('DCGAN', reuse=tf.AUTO_REUSE):
                # TODO: Set the dtype as a global
                # The latent representation of the data (images)
                self.Z = Z = tf.placeholder(tf.float32, [None, self.gen_zDim], name='Z')

                if yDim is not None:
                    # The categories of the data if they exist, else None
                    self.Y = Y = tf.placeholder(tf.float32, [batch_size, yDim], name='Y')
                
                # The data (images, etc)
                self.X = X = tf.placeholder(tf.float32, [batch_size] + image_dims,
                                            name='images')
                
                # Define the Generator and Discriminator objects
                self.gen = Generator(Z, output_height=output_height, output_width=output_width,
                                     Y)
                self.disc = Discriminator(Y)
                
                self.gen_data = gen_data = self.gen.gen_data
                self.disc_trueprobs = self.disc.disc_truelogits
                self.disc_truelogits = self.disc.disc_trueprobs 
                self.disc_fakeprobs, self.disc_fakelogits = self.disc.get_discriminator_output(gen_data)
                
                self.Dloss_true = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(self.disc_truelogits,
                                                            tf.ones_like(self.disc_trueprobs)))
                self.Dloss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(self.disc_fakelogits,
                                                            tf.zeros_like(self.disc_fakeprobs)))
                self.Gloss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(self.disc_fakelogits,
                                                            tf.ones_like(self.disc_fakeprobs)))

                self.Dloss_true_summ = tf.summary.scalar('Dloss_true', self.Dloss_true)
                self.Dloss_fake_summ = tf.summary.scalar('Dloss_fake', self.Dloss_fake)
                
                self.Dloss = self.Dloss_true + self.Dloss_fake
                
                self.Gloss_summ = tf.summary.scalar("Gloss", self.Gloss)
                self.Dloss_summ = tf.summary.scalar("Dloss", self.Dloss)
                
                self.saver = tf.train.Saver()

                
                
    def load_mnist(self):
        """
        """
        data_dir = os.path.join("./data", self.dataset_name)
           
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape([60000, 28, 28, 1]).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape([60000]).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape([10000,28,28,1]).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape([10000]).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)
        
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0
        
        return X/255., y_vec
        
        
    def train(self):
        """
        """
        self.writer = tf.summary.FileWriter('./logs', graph=self.graph)
        sess = tf.get_default_session()
        
        
class Generator():
    """
    """
    def __init__(self, Z, output_height, output_width, Y=None, yDim=None,
                 num_topfilters=64, num_data_channels=3, batch_size=1):
        """
        """
        self.output_height = output_height
        self.output_width = output_width
        self.batch_size = batch_size
        
        self.Z = Z # The latent representation of the data
        self.xDim = Z.get_shape()[1]
        self.num_data_channels = num_data_channels
        self.num_topfilters = num_topfilters
        
        self.Y = Y # The categories if present, else None
        if Y is not None:
            self.yDim = Y.get_shape()[-1]
        with tf.variable_scope('generator'):        
            self.gen_data = self.get_generator_output(batch_size=batch_size)
        
    def get_generator_output(self, Input=None, batch_size=None, Y=None):
        """
        """
        if Input is None:
            Input = self.Z
        if batch_size is None:
            batch_size = self.batch_size
        if Y is None:
            Y = self.Y
            
        num_topfilters = self.num_topfilters
        num_data_channels = self.num_data_channels
        o_height = self.output_height
        o_width = self.output_width
        batch_size = batch_size
        
        # Define the layer generators
        fully_connected_layer = FullLayer()
        conv_transpose_layer = ConvTransposeLayer()
        batch_norm_layer = BatchNormalizationLayer()
        if not self.yDim:
            # Determine shapes of conv. traspose layers, down from the size of input
            shp0 = [batch_size, o_height, o_width, num_data_channels]
            shp1 = [batch_size, conv_output_size(shp0[1], stride=2),
                    conv_output_size(shp0[2], stride=2), num_topfilters]
            shp2 = [batch_size, conv_output_size(shp1[1], stride=2),
                    conv_output_size(shp1[2], stride=2), num_topfilters*2]
            shp3 = [batch_size, conv_output_size(shp2[1], stride=2),
                    conv_output_size(shp2[2], stride=2), num_topfilters*4]
            shp4 = [batch_size, conv_output_size(shp3[1], stride=2),
                    conv_output_size(shp3[2], stride=2), num_topfilters*8]
            nodes_full_layer = shp4[1]*shp4[2]*shp4[3]
            
            # Define the Generator network
            full0 = fully_connected_layer(Input, nodes_full_layer, scope='full0',
                                          nl='linear')
            full0 = tf.reshape(full0, shp4)
            bn0 = batch_norm_layer(full0, scope='bn0', nl='relu')
            
            convt1 = conv_transpose_layer(bn0, shp3, scope='convt1', nl='linear')
            bn1 = batch_norm_layer(convt1, scope='bn1', nl='relu')
            
            convt2 = conv_transpose_layer(bn1, shp2, scope='convt2', nl='linear')
            bn2 = batch_norm_layer(convt2, scope='bn2', nl='relu')
             
            convt3 = conv_transpose_layer(bn2, shp1, scope='convt3', nl='linear')
            bn3 = batch_norm_layer(convt3, scope='bn3', nl='relu')
             
            conv4 = conv_transpose_layer(bn3, shp0, scope='convt4', nl='tanh')
            
            return conv4
        else:
#             s_h, s_w = self.output_height, self.output_width
            shpYaug = [batch_size, 1, 1, self.yDim]
            shp0 = [batch_size, o_height, o_width, num_data_channels]
            shp1 = [batch_size, int(o_height/2), int(o_width/2), num_topfilters*2]
            shp2 = [batch_size, int(o_height/4), int(o_width/4), num_topfilters*2]

#             s_h2, s_h4 = int(o_height/2), int(/4)
#             s_w2, s_w4 = int(s_w/2), int(s_w/4)

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            Yaug = tf.reshape(Y, shpYaug)
#             yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            # Concatenate the latent rep. of data with the category it is coding. 
            InputY = tf.concat([Input, Y], axis=1)
#             z = concat([z, y], 1)
            
            num_nodes_initlayer = self.num_nodes_initlayer
            full1 = fully_connected_layer(InputY, num_nodes_initlayer, scope='full1',
                                          nl='linear')
            bn1 = batch_norm_layer(full1, scope='bn1', nl='relu')
#             h0 = tf.nn.relu(
#                     self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            bn1Y = tf.concat([bn1, Y], axis=1)
#             h0 = concat([h0, y], 1)
            
            full2 = fully_connected_layer(bn1Y, shp2[1]*shp2[2]*shp2[3], scope='full2',
                                          nl='linear')
            bn2 = batch_norm_layer(full2, scope='bn2', nl='relu')
#             h1 = tf.nn.relu(self.g_bn1(
#                     linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
            bn2 = tf.reshape(bn2, shp2)
#             h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
            def concat_with_Yaug(out, Yaug):
                out_shape = out.get_shape()
                Yaug_shape = Yaug.get_shape()
                return tf.concat([out, Yaug*tf.ones(out_shape[:-1] + [Yaug_shape[-1]])] )
              
            bn2Yaug = concat_with_Yaug(bn2, Yaug)  
#             h1 = conv_cond_concat(h1, yb)

            convt3 = conv_transpose_layer(bn2Yaug, shp1, scope='convt3', nl='linear')
            bn3 = batch_norm_layer(convt3, scope='bn3', nl='relu')

#             h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
#                     [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
            bn3Yaug = concat_with_Yaug(bn3, Yaug)
#             h2 = conv_cond_concat(h2, yb)
            
            convt4 = conv_transpose_layer(bn3Yaug, shp0, scope='convt4', nl='sigmoid')
#             return tf.nn.sigmoid(
#                     deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
            return convt4
            
            
            
