# This repo is a heavily modified version of Taehoon Kim's
#
# https://github.com/carpedm20/DCGAN-tensorflow
#
# for my own exploratory purposes. Visit his github page for it is golden!

import os
import time

import numpy as np
import tensorflow as tf

from code.utils import FullLayer, ConvTransposeLayer, BatchNormalizationLayer, ConvLayer

import math
from code.datetools import addDateTime

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

def concat_with_Yaug(out, Yaug):
    out_shape = out.get_shape().as_list()
    Yaug_shape = Yaug.get_shape().as_list()
    return tf.concat([out, Yaug*tf.ones(out_shape[:-1] + Yaug_shape[-1:])], axis=3)

        

class Sampler():
    """
    """
    def __init__(self):
        pass
    

class DCGAN():
    """
    Assumes 2D data
    """
    def __init__(self, zDim=100, data_height=28, data_width=28, output_height=64, 
                 output_width=64, dataset='mnist', ckpt_dir=None, batch_size=1,
                 yDim=None):
        """
        """
        self.data_height = data_height
        self.data_width = data_width
        self.output_height = output_height
        self.output_width = output_width
        
        self.zDim = zDim
        self.batch_size = batch_size
        self.yDim = yDim
        

        self.dataset = dataset
        self.ckpt_dir = ckpt_dir
        if dataset == 'mnist':
            self.Xdata, self.Ydata = self.load_mnist()
            self.num_channels = num_channels = self.Xdata.shape[-1]
        else:
            # TODO:
            pass
        data_dims = [self.data_height, self.data_width, num_channels]

        self.graph = graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('DCGAN', reuse=tf.AUTO_REUSE):
                # TODO: Set the dtype as a global
                self.Z = Z = tf.placeholder(tf.float32, [None, self.zDim], name='Z') # Latent representation of the data (images)
                self.Z_hist = tf.summary.histogram('Zhist', self.Z)
                
                if yDim is not None:
                    self.Y = Y = tf.placeholder(tf.float32, [batch_size, yDim], name='Y') # The categories of the data if they exist, else None
                
                self.X = X = tf.placeholder(tf.float32, [batch_size] + data_dims, name='data') # The data (images, etc)
                
                self.gen = Generator(Z, output_height=data_height, output_width=data_width, Y=Y,
                                     batch_size=self.batch_size)
                self.disc = Discriminator(X, Y=Y)
                
                self.gen_output = gen_output = self.gen.gen_output
                self.disc_trueprobs = self.disc.disc_trueprobs
                self.disc_truelogits = self.disc.disc_truelogits 
                self.disc_fakeprobs, self.disc_fakelogits = self.disc.get_discriminator_output(gen_output)
                
                self.disc_trueprobs_hist = tf.summary.histogram(name='DhistT', values=self.disc_trueprobs)
                self.disc_fakeprobs_hist = tf.summary.histogram(name='DhistF', values=self.disc_fakeprobs)
                self.gen_output_imge = tf.summary.image('Gimge', self.gen_output)
                
                # Push the discriminator to output 1 when evaluated on true data
                self.disc_loss_true = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_truelogits,
                                                            labels=tf.ones_like(self.disc_trueprobs)))
                # Push the discriminator to output 0 when evaluated on the generator data
                self.disc_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fakelogits,
                                                            labels=tf.zeros_like(self.disc_fakeprobs)))
                self.disc_loss = self.disc_loss_true + self.disc_loss_fake

                # Push the generator to cheat the discriminator to output 1,
                # when evaluated on its own generated data
                self.gen_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fakelogits,
                                                            labels=tf.ones_like(self.disc_fakeprobs)))

                self.disc_loss_true_summ = tf.summary.scalar('disc_loss_true', self.disc_loss_true)
                self.disc_loss_fake_summ = tf.summary.scalar('disc_loss_fake', self.disc_loss_fake)
                
                self.gen_loss_summ = tf.summary.scalar("Gloss", self.gen_loss)
                self.disc_loss_summ = tf.summary.scalar("disc_loss", self.disc_loss)
                
                self.saver = tf.train.Saver()
                                
                
    def train(self, params):
        """
        """
        # Add the optimizer ops to the DCGAN graph
        with self.graph.as_default():
            self.train_step = tf.get_variable("global_step", [], tf.int32,
                                              tf.zeros_initializer(),
                                              trainable=False)

            # Leave room for clipping grads, etc.
            d_opt = tf.train.AdamOptimizer(params.learning_rate, beta1=params.beta1)
            d_opt_varsgrads = d_opt.compute_gradients(self.disc_loss)
            d_train_op = d_opt.apply_gradients(d_opt_varsgrads, global_step=self.train_step,
                                               name='Dtrain')
            
            g_opt = tf.train.AdamOptimizer(params.learning_rate, beta1=params.beta1)
            g_opt_varsgrads = g_opt.compute_gradients(self.gen_loss)
            g_train_op = g_opt.apply_gradients(g_opt_varsgrads, name='Gtrain')

            self.gen_summ = tf.summary.merge([self.Z_hist, self.disc_fakeprobs_hist, self.gen_output_imge,
                                              self.disc_loss_fake_summ, self.gen_loss_summ])
            self.disc_summ = tf.summary.merge([self.Z_hist, self.disc_trueprobs_hist,
                                               self.disc_loss_true_summ, self.disc_loss_summ])

#         try:
#             tf.global_variables_initializer().run()
#         except:
#             tf.initialize_all_variables().run()
# 
#         sess = tf.get_default_session()
        start_time = time.time()
        with tf.Session(graph=self.graph) as sess:
            logdir = './logs/' + addDateTime()
            self.writer = tf.summary.FileWriter(logdir, sess.graph)
            sess.run(tf.global_variables_initializer())
            ctr = 0
            for epoch in range(params.num_epochs):
                if params.dataset == 'mnist':
                    num_batches = len(self.Xdata) // params.batch_size
                else:
                    pass        

                for idx in range(0, num_batches):
                    if params.dataset == 'mnist':
                        batch_images = self.Xdata[idx*params.batch_size:(idx+1)*params.batch_size]
                        batch_labels = self.Ydata[idx*params.batch_size:(idx+1)*params.batch_size]
                    else:
                        pass
                    
                    # z should never be random. That's just wrong :)
                    Zbatch = np.random.uniform(-1, 1, [params.batch_size, self.zDim]) \
                                .astype(np.float32)
 
                    if params.dataset == 'mnist':
                        # Update D network
                        _, summary = sess.run([d_train_op, self.disc_summ],
                                                   feed_dict={self.X : batch_images,
                                                              self.Y : batch_labels,
                                                              self.Z : Zbatch})
                        self.writer.add_summary(summary, ctr)
     
                        # Update G network
                        _, summary = sess.run([g_train_op, self.gen_summ],
                                                    feed_dict={self.Z : Zbatch,
                                                               self.Y : batch_labels})
                        self.writer.add_summary(summary, ctr)
     
                        # Run g_optim twice to make sure that d_loss does not go to
                        # zero (different from paper)
    #                     _, summary_str = self.sess.run([g_optim, self.g_sum],
    #                         feed_dict={ self.z: batch_z, self.y:batch_labels })
    #                     self.writer.add_summary(summary_str, counter)
    #                     
                        errD_fake = self.disc_loss_fake.eval({self.Z : Zbatch,
                                                              self.Y : batch_labels })
                        errD_real = self.disc_loss_true.eval({self.X : batch_images,
                                                              self.Y : batch_labels})
                        errG = self.gen_loss.eval({self.Z : Zbatch, self.Y : batch_labels})
#                 else:
#                     # Update D network
#                     _, summary_str = self.sess.run([d_optim, self.d_sum],
#                         feed_dict={ self.inputs: batch_images, self.z: batch_z })
#                     self.writer.add_summary(summary_str, counter)
# 
#                     # Update G network
#                     _, summary_str = self.sess.run([g_optim, self.g_sum],
#                         feed_dict={ self.z: batch_z })
#                     self.writer.add_summary(summary_str, counter)
# 
#                     # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
#                     _, summary_str = self.sess.run([g_optim, self.g_sum],
#                         feed_dict={ self.z: batch_z })
#                     self.writer.add_summary(summary_str, counter)
#                     
#                     errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
#                     errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
#                     errG = self.g_loss.eval({self.z: batch_z})
# 
                    ctr += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, num_batches,
                            time.time() - start_time, errD_fake+errD_real, errG))
# 
#                 if np.mod(counter, 100) == 1:
#                     if config.dataset == 'mnist':
#                         samples, d_loss, g_loss = self.sess.run(
#                             [self.sampler, self.d_loss, self.g_loss],
#                             feed_dict={
#                                     self.z: sample_z,
#                                     self.inputs: sample_inputs,
#                                     self.y:sample_labels,
#                             }
#                         )
#                         save_images(samples, image_manifold_size(samples.shape[0]),
#                                     './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
#                         print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
#                     else:
#                         try:
#                             samples, d_loss, g_loss = self.sess.run(
#                                 [self.sampler, self.d_loss, self.g_loss],
#                                 feed_dict={
#                                         self.z: sample_z,
#                                         self.inputs: sample_inputs,
#                                 },
#                             )
#                             save_images(samples, image_manifold_size(samples.shape[0]),
#                                         './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
#                             print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
#                         except:
#                             print("one pic error!...")
# 
#                 if np.mod(counter, 500) == 2:
#                     self.save(config.checkpoint_dir, counter)

                       
    def load_mnist(self):
        """
        """
        BASE_DIR = '/Users/danielhernandez/work/dcgan-d/data'
        data_dir = os.path.join(BASE_DIR, self.dataset)
           
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
        
        y_vec = np.zeros((len(y), self.yDim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0
        
        return X/255., y_vec
                
    
class Generator():
    """
    """
    def __init__(self, Z, output_height, output_width, Y=None, num_topfilters=64,
                 num_data_channels=3, batch_size=1, nnodes_initlayer_withY=512):
        """
        """
        self.output_height = output_height
        self.output_width = output_width
        self.batch_size = batch_size
        
        self.Z = Z # The latent representation of the data
        self.nnodes_initlayer_withY = nnodes_initlayer_withY
        self.xDim = Z.get_shape()[1]
        self.num_data_channels = num_data_channels
        self.num_topfilters = num_topfilters
        
        self.Y = Y # The categories if present, else None
        if Y is not None:
            self.yDim = Y.get_shape()[-1]
        else:
            self.yDim = None
            
        with tf.variable_scope('generator'):        
            self.gen_output = self.get_generator_output(batch_size=batch_size)
        
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
        
        # Get some layer generators
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
            # Determine shapes of conv. traspose layers, down from the size of input
            shpYaug = [batch_size, 1, 1, self.yDim]
            shp0 = [batch_size, o_height, o_width, num_data_channels]
            shp1 = [batch_size, conv_output_size(shp0[1], stride=2),
                    conv_output_size(shp0[2], stride=2), num_topfilters*2]
            shp2 = [batch_size, conv_output_size(shp1[1], stride=2),
                    conv_output_size(shp1[2], stride=2), num_topfilters*2]

            nnodes_initlayer = self.nnodes_initlayer_withY

            # Augment each layer with Y
            InputY = tf.concat([Input, Y], axis=1)
            
            full1 = fully_connected_layer(InputY, nnodes_initlayer, scope='full1', nl='linear')
            bn1 = batch_norm_layer(full1, scope='bn1', nl='relu')
            bn1Y = tf.concat([bn1, Y], axis=1)
            
            # Reshape before moving on to the conv layers
            full2 = fully_connected_layer(bn1Y, shp2[1]*shp2[2]*shp2[3], scope='full2',
                                          nl='linear')
            bn2 = batch_norm_layer(full2, scope='bn2', nl='relu')
            bn2 = tf.reshape(bn2, shp2)
            # Augment Y for concatenation with conv layers. Use little
            # auxiliary concat function
            Yaug = tf.reshape(Y, shpYaug)
            bn2Yaug = concat_with_Yaug(bn2, Yaug)  

            convt3 = conv_transpose_layer(bn2Yaug, shp1, scope='convt3', nl='linear')
            bn3 = batch_norm_layer(convt3, scope='bn3', nl='relu')
            bn3Yaug = concat_with_Yaug(bn3, Yaug)
            
            convt4 = conv_transpose_layer(bn3Yaug, shp0, scope='convt4', nl='sigmoid')

            return convt4
            

class Discriminator():
    """
    """
    def __init__(self, X, num_initfilters=64, Y=None, num_exitnodes=512):
        """
        """
        self.num_initfilters = num_initfilters
        self.num_exitnodes = num_exitnodes

        self.X = X
        Xshape = X.get_shape()
        self.batch_size, self.num_data_channels = int(Xshape[0]), int(Xshape[-1])
        
        self.Y = Y
        if Y is not None:
            Yshape = Y.get_shape()
            self.yDim = int(Yshape[-1])
            if self.batch_size != int(Yshape[0]):
                raise ValueError("The batch sizes of the input data and the category data do not coincide")
        else:
            self.yDim = None
    
        with tf.variable_scope('discriminator'):
            self.disc_trueprobs, self.disc_truelogits = self.get_discriminator_output()
            
    def get_discriminator_output(self, XInput=None, Y=None):
        """
        """
        num_initfilters = self.num_initfilters
        batch_size = self.batch_size

        if XInput is None:
            XInput = self.X
        else:
            if XInput.get_shape()[0] != batch_size:
                raise ValueError("The 0th dimension of `XInput` must match `self.batch_size`")
        
        yDim = self.yDim
        
        # Get some layer generators
        conv_layer = ConvLayer()
        batch_norm_layer = BatchNormalizationLayer()
        fully_connected_layer = FullLayer()
        if not yDim:
            # Define the Discriminator Network
            
            conv1 = conv_layer(XInput, num_initfilters, scope='conv1', nl='lkyrelu')
            
            conv2 = conv_layer(conv1, 2*num_initfilters, scope='conv2', nl='linear')
            bn2 = batch_norm_layer(conv2, scope='bn2', nl='lkyrelu') 

            conv3 = conv_layer(bn2, 4*num_initfilters, scope='conv3', nl='linear')
            bn3 = batch_norm_layer(conv3, scope='bn3', nl='lkyrelu') 

            conv4 = conv_layer(bn3, 8*num_initfilters, scope='conv4', nl='linear')
            bn4 = batch_norm_layer(conv4, scope='bn4', nl='lkyrelu') 
            
            full5 = fully_connected_layer(tf.reshape(bn4, [self.batch_size, -1]), 1, scope='full5')
            
            return tf.nn.sigmoid(full5), full5
        else:
            num_exitnodes = self.num_exitnodes
            
            if Y is None: 
                Y = self.Y
            else:
                if Y.get_shape()[0] != self.batch_size:
                    raise ValueError("The 0th dimension of `Y` must match `self.batch_size`")
                if Y.get_shape()[-1] != yDim:
                    raise ValueError("The last dimension of `Y` must match `self.yDim`")
            
            # Augment XInput with the category data. Use littel concat function
            Yaug = tf.reshape(Y, [batch_size, 1, 1, yDim])
            XInputY = concat_with_Yaug(XInput, Yaug)
            conv1 = conv_layer(XInputY, self.num_data_channels + yDim, scope='conv1', nl='lkyrelu')
            conv1Yaug = concat_with_Yaug(conv1, Yaug)

            conv2 = conv_layer(conv1Yaug, num_initfilters + yDim, scope='conv2', nl='linear')
            bn2 = batch_norm_layer(conv2, scope='bn2', nl='lkyrelu')
            bn2 = tf.reshape(bn2, [batch_size, -1])
            bn2Y = tf.concat([bn2, Y], axis=1)

            full3 = fully_connected_layer(bn2Y, num_exitnodes, scope='full3', nl='linear')
            bn3 = batch_norm_layer(full3, scope='bn3', nl='lkyrelu')

            full4 = fully_connected_layer(bn3, 1, scope='full4', nl='linear')
            
            return tf.nn.sigmoid(full4), full4

            
