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

from typing import List, Optional

from nncf.common.graph import NNCFNode
from nncf.common.graph.version_agnostic_op_names import get_version_agnostic_name
from nncf.common.utils.backend import __nncf_backend__
from nncf.common.utils.registry import Registry


class OperatorMetatype:
    """
    Base class for grouping PyTorch operators based on their semantic meaning.
    Each derived class represents a single semantic group - for example, AddMetatype would
    group together '__iadd__', '__add__' and '__radd__' operations which all define elementwise
    tensor addition.
    Derived classes also specify which PyTorch functions in which modules should be patched
    and in what manner, so that the entire group of operations is visible in the internal graph
    representation. Grouping also allows efficient application of HW specifics to compression of
    certain operation groups.
    """
    name = ''
    hw_config_names = []  # type: List[str]
    subtypes = []  # type: List['OperatorMetatype']
    is_subtype = False

    @classmethod
    def compliance_check(cls, node: NNCFNode) -> bool:
        raise NotImplementedError()

    @classmethod
    def get_all_aliases(cls) -> List[str]:
       raise NotImplementedError()

    @classmethod
    def determine_subtype(cls, node: NNCFNode) -> Optional['OperatorMetatype']:
        matches = []
        for subtype in cls.subtypes:
            if subtype.compliance_check(node):
                matches.append(subtype)
        if len(matches) > 1:
            raise RuntimeError('Multiple subtypes match operator call '
                               '- cannot determine single subtype.')
        if not matches:
            return None
        return matches[0]


class OperatorMetatypeRegistry(Registry):
    def __init__(self, name):
        super().__init__(name)
        self._op_name_to_op_meta_dict = {}

    def register(self, name=None):
        name_ = name
        super_register = super()._register

        def wrap(obj: OperatorMetatype):
            cls_name = name_
            if cls_name is None:
                cls_name = obj.__name__
            super_register(obj, cls_name)
            if obj.is_subtype:
                return obj
            op_names = obj.get_all_aliases()
            for name in op_names:
                name = get_version_agnostic_name(name)
                if name not in self._op_name_to_op_meta_dict:
                    self._op_name_to_op_meta_dict[name] = obj
                else:
                    assert self._op_name_to_op_meta_dict[name] == obj, \
                        'Inconsistent operator metatype registry - ' \
                        'single patched op name maps to multiple metatypes!'
            return obj

        return wrap

    def get_operator_metatype_by_op_name(self, op_name: str) -> OperatorMetatype:
        if op_name not in self._op_name_to_op_meta_dict:
            return self._op_name_to_op_meta_dict['noop']
        return self._op_name_to_op_meta_dict[op_name]


def get_operator_metatypes():
    if __nncf_backend__ == 'Torch':
        from nncf.dynamic_graph.operator_metatypes \
            import PT_OPERATOR_METATYPES
        return PT_OPERATOR_METATYPES
    if __nncf_backend__ == 'TensorFlow':
        from beta.nncf.tensorflow.graph.operator_metatypes \
            import TF_OPERATOR_METATYPES
        return TF_OPERATOR_METATYPES
    return None