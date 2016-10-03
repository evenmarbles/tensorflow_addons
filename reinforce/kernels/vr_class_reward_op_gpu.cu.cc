#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/reinforce/kernels/vr_class_reward_op.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization for a GPUDevice, that uses the Eigen implementation
// from VRClassRewardEigenImpl.
namespace functor {
    template <typename T, typename Index>
    struct VRClassRewardFunctor<GPUDevice, T, Index> {
      void operator()(const GPUDevice& d,
                      typename TTypes<T>::ConstMatrix logits,
                      typename TTypes<T>::ConstVec baseline,
                      typename TTypes<Index>::ConstVec labels,
                      typename TTypes<Index>::Vec scratch,
                      typename TTypes<T>::Scalar loss,
                      typename TTypes<T>::Vec baseline_loss,
                      typename TTypes<T>::Vec rewards,
                      typename TTypes<T>::Vec scale,
                      bool average) {
        VRClassRewardEigenImpl<GPUDevice, T, Index>::Compute(d, logits, baseline, labels, scratch,
                                                             loss, baseline_loss, rewards, scale,
                                                             average);
      }
    };

//    struct VRClassRewardFunctor<GPUDevice, Eigen::half, Index> {
//        void operator()(const GPUDevice& d,
//                        typename TTypes<Eigen::half>::ConstMatrix logits,
//                        typename TTypes<Eigen::half>::ConstVec baseline,
//                        typename TTypes<Index>::ConstVec labels,
//                        typename TTypes<Index>::Vec scratch,
//                        typename TTypes<Eigen::half>::Scalar loss,
//                        typename TTypes<Eigen::half>::Vec baseline_loss,
//                        typename TTypes<Eigen::half>::Vec rewards,
//                        typename TTypes<T>::Vec scale,
//                        bool average) {
//            VRClassRewardEigenImpl<GPUDevice, float, Index>::Compute(d, logits.cast<float>(),
//                                   baseline.cast<float>(), labels, scratch, loss.cast<float>(),
//                                   baseline_loss.cast<float>(), rewards.cast<float>(),
//                                   scale.cast<float>(), average);
//            loss.cast<Eigen::half>();
//            reward.cast<Eigen::half>();
//        }
//    };
}  // end namespace functor

// Instantiate the GPU implementation for half, float and double.
#define REGISTER(Index)                                                         \
  template struct functor::VRClassRewardFunctor<GPUDevice, float, Index>;       \
  template struct functor::VRClassRewardFunctor<GPUDevice, double, Index>;
//  template struct functor::VRClassRewardFunctor<GPUDevice, Eigen::half, Index>;
REGISTER(int32)
REGISTER(int64)
#undef REGISTER

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA