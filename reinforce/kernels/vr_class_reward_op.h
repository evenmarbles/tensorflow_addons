
#ifndef TENSORFLOW_KERNELS_VR_CLASS_REWARD_OP_H_
#define TENSORFLOW_KERNELS_VR_CLASS_REWARD_OP_H_
// Functor definition for VRClassRewardOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by VRClassRewardOp to do the computations.
template <typename Device, typename T, typename Index>
struct VRClassRewardFunctor {
    // Computes Cross Entropy loss and backprop.
    //
    // logits: batch_size, num_classes.
    // labels: batch_size, num_classes.
    // scratch: temporary tensor, dims: batch_size, 1
    // loss: output tensor for the loss, dims: batch_size.
    void operator()(const Device& d,
                    typename TTypes<T>::ConstMatrix logits,
                    typename TTypes<T>::ConstVec baseline,
                    typename TTypes<Index>::ConstVec labels,
                    typename TTypes<Index>::Vec scratch,
                    typename TTypes<T>::Scalar loss,
                    typename TTypes<T>::Vec baseline_loss,
                    typename TTypes<T>::Vec rewards,
                    typename TTypes<T>::Vec scale,
                    bool average);
};

// Eigen code implementing VRClassRewardFunctor::operator().
// This code works for both CPU and GPU and is used by the functor
// specializations for both device types.
template <typename Device, typename T, typename Index>
struct VRClassRewardEigenImpl {
    static void Compute(const Device& d,
                        typename TTypes<T>::ConstMatrix logits,
                        typename TTypes<T>::ConstVec baseline,
                        typename TTypes<Index>::ConstVec labels,
                        typename TTypes<Index>::Vec scratch,
                        typename TTypes<T>::Scalar loss,
                        typename TTypes<T>::Vec baseline_loss,
                        typename TTypes<T>::Vec rewards,
                        typename TTypes<T>::Vec scale,
                        bool average) {

    const int kBatchDim = 0;
    const int kClassDim = 1;

    int batch_size = logits.dimension(kBatchDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> along_class;
    along_class[0] = kClassDim;
#else
    Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
#endif

    // reward = scale when correctly classified
    scratch.device(d) = logits.argmax(kClassDim).template cast<Index>();
    rewards.device(d) = (scratch == labels).template cast<T>();
    rewards.device(d) = rewards * scale;

    // loss = -sum(reward)
    loss.device(d) = -rewards.sum();

    if (average) {
        loss.device(d) = loss / static_cast<T>(batch_size);
    }

    baseline_loss.device(d) = (baseline - rewards).square().mean(along_class);

    // reduce variance of reward using baseline
    rewards.device(d) = rewards - baseline;
    if (average) {
        scratch.setConstant(batch_size);
        rewards.device(d) = rewards / scratch.template cast<T>();
    }
}
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_VR_CLASS_REWARD_OP_H_
