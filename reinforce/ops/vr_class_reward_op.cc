
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// loss = vf_class_reward(scale, objective) computes the variance reduced
// classification reinforcement cost and the reward.
//
REGISTER_OP("VRClassReward")
    .Input("logits: T")
    .Input("baseline: T")
    .Input("labels: Tlabels")
    .Output("loss: T")
    .Output("baseline_loss: T")
    .Output("reward: T")
    .Attr("T: {float, double}")
    .Attr("Tlabels: {int32, int64} = DT_INT64")
    .Attr("scale: float = 1")
    .Attr("average: bool = true")
    .Attr("objective: string = 'mse'")
    .Doc(R"doc(
Computes the variance reduced classification reinforcement cost and the
reward.

Inputs are logits, not probabilities.

This operation accepts a single label per row of features.  This label is
considered to have probability 1.0 for the given row.

logits: A 2-D `Tensor` of shape `[batch_size, num_classes]`, representing
  the class predictions.
baseline: A 1-D `Tensor` (batch_size vector). This is the predicted
  baseline reward.
labels: A 1-D `Tensor` (batch_size vector) with values in [0, num_classes).
  This is the label for the given minibatch entry and represents the
  baseline reward.
loss: Per example loss (batch_size vector).
baseline_loss: The loss of the baseline (batch_size vector).
reward: The reward for the given minibatch entry (batch_size vector).
scale: The reward scale. The reward is 1 for success, 0 otherwise.
objective: The objective function to use to learn the baseline reward.
)doc");


// --------------------------------------------------------------------------

}  // end namespace tensorflow
