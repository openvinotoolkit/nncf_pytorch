"""
 Copyright (c) 2021 Intel Corporation
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

import tensorflow as tf

from tensorflow.python.keras.utils.control_flow_util import smart_cond

from beta.nncf.tensorflow.functions import logit
from beta.nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from beta.nncf.tensorflow.layers.operation import InputType
from beta.nncf.tensorflow.layers.operation import NNCFOperation
from beta.nncf.tensorflow.sparsity.magnitude.functions import apply_mask
from beta.nncf.tensorflow.sparsity.rb.functions import calc_rb_binary_mask, st_binary_mask, binary_mask

OP_NAME = '_rb_sparsity_mask_apply'

@NNCF_CUSTOM_OBJECTS.register()
class RBSparsifyingWeight(NNCFOperation):

    def __init__(self, name, eps=1e-6):
        '''
        :param name: model scope unique operation name
        :param eps: minimum value and the gap from the maximum value in
            distributed mask
        '''
        super().__init__(name=name)
        self.eps = eps

    def build(self, input_shape, input_type, name, layer):
        '''
        :param input_shape: shape of weights which needs to be sparsifyed
        :param input_type: type of operation input, must be InputType.WEIGHTS
        :param name: name of weight attribute which needs to be sparsifyed
        :param layer: layer which needs to be sparsifyed
        '''
        if input_type is not InputType.WEIGHTS:
            raise ValueError(
                'RB Sparsity mask operation could not be applied to input of the layer: {}'.
                    format(layer.name))

        mask = layer.add_weight(
            name + '_mask',
            shape=input_shape,
            initializer=tf.keras.initializers.Constant(logit(0.99)),
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN)

        trainable = layer.add_weight(
            name + '_trainable',
            initializer=tf.keras.initializers.Constant(True),
            trainable=False,
            dtype=tf.bool)

        return {
            'mask': mask,
            'trainable': trainable,
        }

    def call(self, layer_weights, op_weights, trainable):
        '''
        Apply rb sparsity mask to given weights

        :param layer_weights: target weights to sparsify
        :param op_weights: operation weights contains
           mask and param `trainable`
        :param trainable: 1 if operation called in training mode
            else 0
        '''
        if not isinstance(trainable, tf.Tensor):
            trainable = tf.constant(bool(trainable), tf.bool)
        return smart_cond(tf.math.logical_and(op_weights['trainable'], trainable),
                       true_fn=lambda: apply_mask(layer_weights, calc_rb_binary_mask(op_weights['mask'], self.eps)),
                       false_fn=lambda: apply_mask(layer_weights, binary_mask(op_weights['mask'])))

    def freeze(self, trainable_weight):
        '''
        Freeze rb mask from operation weights

        :param trainable_weight: trainable weight of rb operation
        '''
        trainable_weight.assign(False)

    @staticmethod
    def loss(mask):
        '''
        Return count of non zero weight in mask

        :param mask: given mask
        '''
        return tf.reduce_sum(st_binary_mask(mask))
