"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from functools import wraps, reduce

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


L2_FACTOR = 1e-5

@wraps(tf.keras.layers.Conv2D)
def YoloConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for Conv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    #yolo_conv_kwargs = kwargs
    return tf.keras.layers.Conv2D(*args, **yolo_conv_kwargs)


@wraps(YoloConv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for YoloConv2D."""
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    #darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get('strides')==(2,2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloConv2D(*args, **darknet_conv_kwargs)


def CustomBatchNormalization(*args, **kwargs):
    if tf.__version__ >= '2.2':
        BatchNorm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
        BatchNorm = tf.keras.layers.BatchNormalization

    return BatchNorm(*args, **kwargs)


class Darknet:
    """Class to build CSPDarknet53"""

    def mish(self, x):
        return x * K.tanh(K.softplus(x))


    def DarknetConv2D_BN_Mish(self, *args, **kwargs):
        """Darknet Convolution2D followed by CustomBatchNormalization and Mish."""
        no_bias_kwargs = {'use_bias': False}
        no_bias_kwargs.update(kwargs)
        return compose(
            DarknetConv2D(*args, **no_bias_kwargs),
            CustomBatchNormalization(),
            tf.keras.layers.Activation(self.mish))


    def csp_resblock_body(self, x, num_filters, num_blocks, all_narrow=True):
        '''A series of resblocks starting with a downsampling Convolution2D'''
        # Darknet uses left and top padding instead of 'same' mode
        x = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(x)
        x = self.DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(x)

        res_connection = self.DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)
        x = self.DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)

        for i in range(num_blocks):
            y = compose(
                    self.DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                    self.DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(x)
            x = tf.keras.layers.Add()([x,y])

        x = self.DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)
        x = tf.keras.layers.Concatenate()([x , res_connection])

        return self.DarknetConv2D_BN_Mish(num_filters, (1,1))(x)


    def __call__(self, x):
        '''CSPDarknet53 body having 52 Convolution2D layers'''
        x = self.DarknetConv2D_BN_Mish(32, (3,3))(x)
        x = self.csp_resblock_body(x, 64, 1, False)
        x = self.csp_resblock_body(x, 128, 2)
        x = self.csp_resblock_body(x, 256, 8)
        x = self.csp_resblock_body(x, 512, 8)
        x = self.csp_resblock_body(x, 1024, 4)
        return x