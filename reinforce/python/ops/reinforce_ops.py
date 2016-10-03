from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.reinforce.python.ops.vr_class_reward_op import REWARD

_reinforce_ops_so = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_reinforce_ops.so"))
assert _reinforce_ops_so, "Could not load _reinforce_ops.so."


def _get_reward(input):
    collection = ops.get_collection_ref(REWARD)
    if not collection:
        logging.warn("No reward was calculated. Use `vr_class_reward` "
                     "operation to calculate loss.")
        return array_ops.zeros_like(input)
    reward = collection[-1]

    input_shape = input.get_shape()
    reward_shape = reward.get_shape()

    assert reward_shape.ndims == 1

    if reward_shape == input_shape:
        return reward
    if input_shape.ndims == 2:
        reward = array_ops.tile(array_ops.expand_dims(reward, 1), [1, 2])
    else:
        logging.warn("input with ndims > 2 not supported.")

    return reward


def reinforce_normal(means,
                     stddev=1.0,
                     stochastic=True,
                     seed=None):
    """Samples from a multivariate normal distribution using the REINFORCE algorithm.

    Args:
      means: A `Tensor`. The means of the multivariate normal distribution.
      stddev: A `Tensor` of the same shape as `means` or a 0-D Tensor or Python
        value. The standard deviation of the normal distributions.
      seed: A Python integer. Used to create a random seed for the distribution.
        See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      stochastic: Indicates whether the operation is stochastic. Allows to turn of
        stochasticity during evaluation.

    Returns:
      A `tensor` of the same shape as `means` filled with random normal values
    drawn from the normal distribution specified by mean and stddev.
    """
    dtype = means.dtype
    stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    stochastic_tensor = ops.convert_to_tensor(stochastic, dtype=dtypes.bool, name="stochastic")
    seed1, seed2 = random_seed.get_seed(seed)
    return _reinforce_ops_so.reinforce_normal(means,
                                              stddev_tensor,
                                              stochastic_tensor,
                                              seed=seed1,
                                              seed2=seed2)


@ops.RegisterShape("ReinforceNormal")
def _ReinforceNormal(op):
    """Shape function for ReinforceNormal op."""
    input_shape = op.inputs[0].get_shape().with_rank_at_least(1)
    return [input_shape]


@ops.RegisterGradient("ReinforceNormal")
def _ReinforceNormalGrad(op, _):
    # f: normal probability density funciton
    # x: the sampled values (output)
    # u: mean (input)
    # s: standard deviation (stddev)
    #
    # Derivative of log normal w.r.t. means:
    # d ln(f(x,u,s))   (x - u)
    # -------------- = -------
    #       d u          s^2
    #
    # Derivative of log normal w.r.t. stddevs:
    # d ln(f,u, s))   (x - u)^2 - s^2
    # ------------- = ---------------
    #       d s             s^3
    #
    means = op.inputs[0]
    stddevs = op.inputs[1]

    # x - u
    grad_means = op.outputs[0] - means
    # divide by squared standard deviation
    grad_means = math_ops.truediv(grad_means, math_ops.square(stddevs))
    # multiply by reward
    grad_means = grad_means * _get_reward(means)
    # multiply by -1 (gradient descent on mean)
    grad_means = -grad_means

    grad_stddevs = None
    if stddevs.get_shape().ndims > 0 and stddevs.get_shape()[0] == means.get_shape()[0]:
        # (x - u)^2
        grad_stddevs = math_ops.square(op.outputs[0] - means)
        # subtract s^2
        stddevs_squared = math_ops.square(stddevs)
        grad_stddevs = grad_stddevs - stddevs_squared
        # divide by s^3
        stddevs_cubed = math_ops.add(stddevs_squared * stddevs, ops.convert_to_tensor(0.00001, dtype=stddevs.dtype))
        grad_stddevs = math_ops.truediv(grad_stddevs, stddevs_cubed)
        # multiply by reward
        grad_stddevs = grad_stddevs * _get_reward(stddevs)
        # multiply by -1 (gradient descent on stddev)
        grad_stddevs = -grad_stddevs

    return grad_means, grad_stddevs, None
