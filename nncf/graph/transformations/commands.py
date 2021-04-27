from typing import Callable
from typing import Dict

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType
from nncf.graph.graph import InputAgnosticOperationExecutionContext


class PTTargetPoint(TargetPoint):
    _OPERATION_TYPES = [TargetType.PRE_LAYER_OPERATION,
                        TargetType.POST_LAYER_OPERATION,
                        TargetType.OPERATION_WITH_WEIGHTS]
    _HOOK_TYPES = [TargetType.OPERATOR_PRE_HOOK,
                   TargetType.OPERATOR_POST_HOOK]

    def __init__(self, target_type: TargetType, *,
                 ia_op_exec_context: InputAgnosticOperationExecutionContext = None,
                 module_scope: 'Scope' = None,
                 input_port_id: int = None):
        super().__init__(target_type)
        self.target_type = target_type
        if self.target_type in self._OPERATION_TYPES:
            if module_scope is None:
                raise ValueError("Should specify module scope for module pre- and post-op insertion points!")

        elif self.target_type in self._HOOK_TYPES:
            if ia_op_exec_context is None:
                raise ValueError("Should specify an operator's InputAgnosticOperationExecutionContext "
                                 "for operator pre- and post-hook insertion points!")
        else:
            raise NotImplementedError("Unsupported target type: {}".format(target_type))

        self.module_scope = module_scope
        self.ia_op_exec_context = ia_op_exec_context
        self.input_port_id = input_port_id

    def __eq__(self, other: 'PTTargetPoint'):
        return self.target_type == other.target_type and self.ia_op_exec_context == other.ia_op_exec_context \
               and self.input_port_id == other.input_port_id and self.module_scope == other.module_scope

    def __str__(self):
        prefix = str(self.target_type)
        retval = prefix
        if self.target_type in self._OPERATION_TYPES:
            retval += " {}".format(self.module_scope)
        elif self.target_type in self._HOOK_TYPES:
            if self.input_port_id is not None:
                retval += " {}".format(self.input_port_id)
            retval += " " + str(self.ia_op_exec_context)
        return retval

    def __hash__(self):
        return hash(str(self))

    def get_state(self) -> Dict:
        state = {'target_type': self.target_type.get_state(),
                 'input_port_id': self.input_port_id}
        if self.target_type in self._OPERATION_TYPES:
            state['module_scope'] = str(self.module_scope)
        elif self.target_type in self._HOOK_TYPES:
            state['ia_op_exec_context'] = str(self.ia_op_exec_context)
        return state

    @classmethod
    def from_state(cls, state: Dict) -> 'PTTargetPoint':
        kwargs = {'target_type': TargetType.from_state(state['target_type']),
                  'input_port_id': state['input_port_id']}
        if 'module_scope' in state:
            from nncf.dynamic_graph.context import Scope
            kwargs['module_scope'] = Scope.from_str(state['module_scope'])
        if 'ia_op_exec_context' in state:
            ia_op_exec_context_str = state['ia_op_exec_context']
            kwargs['ia_op_exec_context'] = InputAgnosticOperationExecutionContext.from_str(ia_op_exec_context_str)
        return cls(**kwargs)


class PTInsertionCommand(TransformationCommand):
    def __init__(self, point: PTTargetPoint, fn: Callable,
                 priority: TransformationPriority = TransformationPriority.DEFAULT_PRIORITY):
        super().__init__(TransformationType.INSERT, point)
        self.fn = fn  # type: Callable
        self.priority = priority  # type: TransformationPriority

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # TODO: keep all TransformationCommands atomic, refactor TransformationLayout instead
        raise NotImplementedError()
