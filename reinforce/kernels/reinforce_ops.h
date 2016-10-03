
#ifndef TENSORFLOW_KERNELS_REINFPRCE_OPS_H_
#define TENSORFLOW_KERNELS_REINFPRCE_OPS_H_
// Functor definition for VRClassRewardOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by VRClassRewardOp to do the computations.
template <typename Device, typename TYPE>
struct ReinforceNormalFunctor {
    // Computes the mean of the Gaussian.
    //
    // input: batch_size.
    // output: batch_size.
    // stddev: temporary tensor, dims: batch_size
    void operator()(const Device& d,
                    typename TTypes<TYPE>::ConstFlat means,
                    typename TTypes<TYPE>::Flat stddev,
                    typename TTypes<TYPE>::Flat output,
                    bool stochastic);
};

// Eigen code implementing ReinforceNormalFunctor::operator().
// This code works for both CPU and GPU and is used by the functor
// specializations for both device types.
template <typename Device, typename TYPE>
struct ReinforceNormalEigenImpl {
static void Compute(const Device& d,
                    typename TTypes<TYPE>::ConstFlat means,
                    typename TTypes<TYPE>::Flat stddev,
                    typename TTypes<TYPE>::Flat output,
                    bool stochastic) {

    if (stochastic) {
        output.device(d) = output * stddev;
        // re-center the means to the mean
        output.device(d) += means;
    }
    else {
        // use maximum a posteriori (MAP) estimate
        output.device(d) = means;
    }
}
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_REINFPRCE_OPS_H_
