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

import tensorflow as tf
from tensorflow.python.ops.init_ops import Constant
import numpy as np

from beta.nncf import create_compressed_model
from beta.nncf import NNCFConfig

from beta.examples.tensorflow.common.object_detection.datasets.builder import COCODatasetBuilder


def get_conv_init_value(shape, value):
    mask = np.eye(shape[0], shape[1])
    mask = np.expand_dims(mask, axis=(2, 3))
    value = np.full(shape, value)
    value += mask
    return value


def get_empty_config(input_sample_sizes=None):
    if input_sample_sizes is None:
        input_sample_sizes = [1, 4, 4, 1]

    def _create_input_info():
        if isinstance(input_sample_sizes, tuple):
            return [{"sample_size": sizes} for sizes in input_sample_sizes]
        return [{"sample_size": input_sample_sizes}]

    config = NNCFConfig({
        "model": "basic_sparse_conv",
        "input_info": _create_input_info()
    })
    return config


def get_mock_model(input_shape=(1,)):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = tf.keras.layers.Dense(1)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_basic_conv_test_model(input_shape=(4, 4, 1), out_channels=2, kernel_size=2, weight_init=-1., bias_init=-2.):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = create_conv(input_shape[-1], out_channels, kernel_size, weight_init, bias_init)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_compressed_model_and_algo_for_test(model, config):
    assert isinstance(config, NNCFConfig)
    tf.keras.backend.clear_session()
    algo, model = create_compressed_model(model, config)
    return model, algo


def create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init):
    weight_init = get_conv_init_value((kernel_size, kernel_size, in_channels, out_channels), weight_init)
    conv = tf.keras.layers.Conv2D(out_channels,
                                  kernel_size,
                                  kernel_initializer=Constant(weight_init),
                                  bias_initializer=Constant(bias_init))
    return conv


def check_equal(test, reference, rtol=1e-4):
    test = test.numpy()
    reference = reference.numpy()
    for i, (x, y) in enumerate(zip(test, reference)):
        np.testing.assert_allclose(x, y, rtol=rtol, err_msg="Index: {}".format(i))


class MockCOCODatasetBuilder(COCODatasetBuilder):
    @property
    def num_examples(self):
        return 5


def get_coco_dataset_builders(config, strategy):
    num_devices = strategy.num_replicas_in_sync if strategy else 1

    train_builder = MockCOCODatasetBuilder(config=config,
                                           is_train=True,
                                           num_devices=num_devices)

    val_builder = MockCOCODatasetBuilder(config=config,
                                         is_train=False,
                                         num_devices=num_devices)

    return train_builder, val_builder
