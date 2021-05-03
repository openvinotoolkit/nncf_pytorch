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

from nncf.api.compression import CompressionAlgorithmController
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.utils.logger import logger
from nncf.common.utils.registry import Registry
from nncf.common.compression import StubStatistics
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from beta.nncf.tensorflow.loss import TFZeroCompressionLoss

TF_COMPRESSION_ALGORITHMS = Registry('compression algorithm')


@TF_COMPRESSION_ALGORITHMS.register('NoCompressionAlgorithm')
class NoCompressionAlgorithmBuilder(TFCompressionAlgorithmBuilder):
    def get_transformation_layout(self, _):
        return TransformationLayout()

    def build_controller(self, model) -> CompressionAlgorithmController:
        return NoCompressionAlgorithmController(model)


class NoCompressionAlgorithmController(TFCompressionAlgorithmController):
    def __init__(self, target_model):
        super().__init__(target_model)
        self._loss = TFZeroCompressionLoss()
        self._scheduler = StubCompressionScheduler()

    @property
    def loss(self) -> TFZeroCompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> StubCompressionScheduler:
        return self._scheduler

    def statistics(self, quickly_collected_only: bool = False) -> StubStatistics:
        return StubStatistics()


def get_compression_algorithm_builder(config):
    algorithm_key = config.get('algorithm', 'NoCompressionAlgorithm')
    logger.info('Creating compression algorithm: {}'.format(algorithm_key))
    return TF_COMPRESSION_ALGORITHMS.get(algorithm_key)
