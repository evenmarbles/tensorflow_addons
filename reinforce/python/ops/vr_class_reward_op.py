from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

REWARD = 'reward'

_vr_class_reward_op_so = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_vr_class_reward_op.so"))
assert _vr_class_reward_op_so, "Could not load _vr_class_reward_op.so."

vr_class_reward = _vr_class_reward_op_so.vr_class_reward


@ops.RegisterShape("VRClassReward")
def _VRClassReward(op):
    """Shape function for VRClassReward op."""
    logits_shape = op.inputs[0].get_shape()
    input_shape = logits_shape.with_rank(2)
    batch_size = input_shape[0]
    # reward_shape
    op.inputs[1].get_shape().merge_with(tensor_shape.vector(batch_size))
    # labels_shape
    op.inputs[2].get_shape().merge_with(tensor_shape.vector(batch_size))
    return [tensor_shape.scalar(),
            tensor_shape.vector(batch_size.value),
            tensor_shape.vector(batch_size.value)]


@ops.RegisterGradient("VRClassReward")
def _VRClassRewardGrad(op, *grad):
    collection = ops.get_collection_ref(REWARD)
    for _ in range(len(collection)):
        collection.pop()

    ops.add_to_collection(REWARD, op.outputs[2])

    # learn the baseline reward
    return array_ops.zeros_like(op.inputs[0]), grad[1], None
