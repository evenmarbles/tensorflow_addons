
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// r = reinforce_normal(input, stddev, stochastic) samples from a multivariate
// normal distribution, where the mean is given by `input` and the standard
// deviation is given by `stddev`.
//
REGISTER_OP("ReinforceNormal")
    .SetIsStateful()
    .Input("means: T")
    .Input("stddev: T")
    .Input("stochastic: bool")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Doc(R"doc(
Outputs random values from multivariate normal distribution while
using the REINFORCE algorithm..

means: A `Tensor` representing the means of the multivariate normal distribution.
stddev: A 0-D `Tensor` or a `Tensor of the same size as `means`. The standard
  deviation of the multivariate normal distribution.
stochastic: Boolean Tensor. Indicates if the operation is stochastic. Allows
  to turn of stochasticity during evaluation.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A 1-D `Tensor` (batch size vector) filled with random normal values
  drawn from the normal distribution specified by mean and stddev.
)doc");


// --------------------------------------------------------------------------

}  // end namespace tensorflow
