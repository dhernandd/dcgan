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

from code.dcgan import DCGAN

class DCGANTest(tf.test.TestCase):
    """
    """
    # MNIST properties
    height = 28
    yDim = 10
    width = 28
    
    zDim = 128
    num_datachannels = 1
    dcgan = DCGAN(zDim=zDim, data_height=height, data_width=width, yDim=10)


if __name__ == '__main__':
    tf.test.main()