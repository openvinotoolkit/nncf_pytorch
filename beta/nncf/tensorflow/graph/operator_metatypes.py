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

from typing import List

from nncf.common.graph import NNCFNode
from nncf.common.graph.module_attributes import ConvolutionModuleAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName

TF_OPERATOR_METATYPES = OperatorMetatypeRegistry("operator_metatypes")


class TFOperatorMetatype(OperatorMetatype):
    keras_layer_names = []
    tf_function_names = []

    @classmethod
    def compliance_check(cls, node: NNCFNode) -> bool:
        return node.node_type in cls.get_all_aliases()

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.keras_layer_names + cls.tf_function_names
