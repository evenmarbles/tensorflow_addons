#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/reinforce/kernels/reinforce_ops.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization for a GPUDevice, that uses the Eigen implementation
// from ReinforceNormalEigenImpl.
namespace functor {
    template <typename TYPE>
    struct ReinforceNormalFunctor<GPUDevice, TYPE> {
      void operator()(const GPUDevice& d,
                      typename TTypes<TYPE>::ConstFlat means,
                      typename TTypes<TYPE>::Flat stddev,
                      typename TTypes<TYPE>::Flat output,
                      bool stochastic) {
        ReinforceNormalEigenImpl<GPUDevice, TYPE>::Compute(d, means, stddev,
                                                           output, stochastic);
      }
    };
}  // end namespace functor

// Instantiate the GPU implementation for half, float and double.
template struct functor::ReinforceNormalFunctor<GPUDevice, float>;
template struct functor::ReinforceNormalFunctor<GPUDevice, double>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA