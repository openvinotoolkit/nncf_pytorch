import json

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizerConfig
from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext
from nncf.graph.transformations.commands import PTTargetPoint
from nncf.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.quantization.quantizer_setup import SingleConfigQuantizerSetup


def check_serialization(obj):
    state = obj.get_state()

    serialized_state = json.dumps(state, sort_keys=True, indent=4)
    print(serialized_state)
    deserialized_state = json.loads(serialized_state)

    assert obj == obj.__class__.from_state(state)
    assert obj == obj.__class__.from_state(deserialized_state)


def test_quantizer_setup_serialization():
    target_type_1 = TargetType.OPERATOR_POST_HOOK
    check_serialization(target_type_1)

    target_type_2 = TargetType.POST_LAYER_OPERATION
    check_serialization(target_type_2)

    scope = Scope.from_str('MyConv/1[2]/3[4]/5')
    assert scope == Scope.from_str(str(scope))

    ia_op_exec_ctx = InputAgnosticOperationExecutionContext(operator_name='MyConv', scope_in_model=scope, call_order=1)
    assert ia_op_exec_ctx == InputAgnosticOperationExecutionContext.from_str(str(ia_op_exec_ctx))

    pttp_1 = PTTargetPoint(target_type_1, ia_op_exec_context=ia_op_exec_ctx, input_port_id=7)
    check_serialization(pttp_1)

    pttp_2 = PTTargetPoint(target_type_2, module_scope=scope, input_port_id=7)
    check_serialization(pttp_2)

    qc = QuantizerConfig()
    check_serialization(qc)

    scqp_1 = SingleConfigQuantizationPoint(pttp_1, qc, scopes_of_directly_quantized_operators=[scope])
    check_serialization(scqp_1)
    scqp_2 = SingleConfigQuantizationPoint(pttp_2, qc, scopes_of_directly_quantized_operators=[scope])
    check_serialization(scqp_2)

    scqs = SingleConfigQuantizerSetup()
    scqs.quantization_points = {0: scqp_1, 1: scqp_2}
    scqs.unified_scale_groups = {2: {0, 1}}
    scqs.shared_input_operation_set_groups = {2: {0, 1}}
    check_serialization(scqs)
