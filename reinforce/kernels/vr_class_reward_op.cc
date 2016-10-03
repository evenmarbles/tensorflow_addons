
// See docs in ../contrib/reinforce/ops/vr_class_reward_op.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/contrib/reinforce/kernels/vr_class_reward_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class VRClassRewardOp : public OpKernel {
public:
    explicit VRClassRewardOp(OpKernelConstruction* context)
            : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
        OP_REQUIRES_OK(context, context->GetAttr("average", &average_));
        OP_REQUIRES_OK(context, context->GetAttr("objective", &objective_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& logits = context->input(0);
        const Tensor& baseline = context->input(1);
        const Tensor& labels = context->input(2);
        OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits.shape()),
                    errors::InvalidArgument("logits must be 2-D, but got shape ",
                                            logits.shape().DebugString()));
        OP_REQUIRES(context, TensorShapeUtils::IsVector(baseline.shape()),
                    errors::InvalidArgument("baseline must be 1-D, but got shape ",
                                            baseline.shape().DebugString()));
        OP_REQUIRES(context, TensorShapeUtils::IsVector(labels.shape()),
                    errors::InvalidArgument("labels must be 1-D, but got shape ",
                                            labels.shape().DebugString()));
        OP_REQUIRES(context, baseline.dim_size(0) == labels.dim_size(0),
                    errors::InvalidArgument(
                            "baseline and labels must have the same first dimension, "
                                    "got logits shape ",
                            logits.shape().DebugString(), " and baseline shape ",
                            baseline.shape().DebugString()));
        OP_REQUIRES(context, logits.dim_size(0) == labels.dim_size(0),
                    errors::InvalidArgument(
                            "logits and labels must have the same first dimension, "
                                    "got logits shape ",
                            logits.shape().DebugString(), " and labels shape ",
                            labels.shape().DebugString()));
        OP_REQUIRES(context, logits.dim_size(1) > 0,
                    errors::InvalidArgument(
                            "Must have at least one class, but got logits shape ",
                            logits.shape().DebugString()));

        Tensor scale;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                       labels.shape(), &scale));
        auto scale_tensor = scale.vec<T>();
        scale_tensor.setConstant(scale_);

        Tensor scratch;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Index>::value,
                                                       labels.shape(), &scratch));

        // loss is 0-D.
        Tensor* loss_out = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, TensorShape({}), &loss_out));
        auto loss = loss_out->scalar<T>();

        Tensor* baseline_loss_out = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, labels.shape(), &baseline_loss_out));

        // Create an reward tensor
        Tensor* reward_out = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(2, labels.shape(), &reward_out));

        if (logits.dim_size(0) > 0) {
            functor::VRClassRewardFunctor<Device, T, Index> functor;
            functor(context->eigen_device<Device>(), logits.matrix<T>(),
                    baseline.vec<T>(), labels.vec<Index>(), scratch.vec<Index>(),
                    loss, baseline_loss_out->vec<T>(), reward_out->vec<T>(), scale.vec<T>(), average_);
        }
    }

    private:
        float scale_;
        bool average_;
        string objective_;
};

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from VRClassRewardEigenImpl.
namespace functor {
    template <typename T, typename  Index>
    struct VRClassRewardFunctor<CPUDevice, T, Index> {
        void operator()(const CPUDevice& d,
                        typename TTypes<T>::ConstMatrix logits,
                        typename TTypes<T>::ConstVec baseline,
                        typename TTypes<Index>::ConstVec labels,
                        typename TTypes<Index>::Vec scratch,
                        typename TTypes<T>::Scalar loss,
                        typename TTypes<T>::Vec baseline_loss,
                        typename TTypes<T>::Vec rewards,
                        typename TTypes<T>::Vec scale,
                        bool average) {
            VRClassRewardEigenImpl<CPUDevice, T, Index>::Compute(d, logits, baseline, labels, scratch,
                                                                 loss, baseline_loss, rewards, scale, average);
        }
    };

//    struct VRClassRewardFunctor<CPUDevice, Eigen::half, Index> {
//        void operator()(const CPUDevice& d,
//                        typename TTypes<Eigen::half>::ConstMatrix logits,
//                        typename TTypes<Eigen::half>::ConstVec baseline,
//                        typename TTypes<Index>::ConstVec labels,
//                        typename TTypes<Index>::Vec scratch,
//                        typename TTypes<Eigen::half>::Scalar loss,
//                        typename TTypes<Eigen::half>::Vec baseline_loss,
//                        typename TTypes<Eigen::half>::Vec rewards,
//                        typename TTypes<T>::Vec scale,
//                        bool average) {
//            VRClassRewardEigenImpl<CPUDevice, float, Index>::Compute(d, logits.cast<float>(),
//                                   baseline.cast<float>(), labels, scratch, loss.cast<float>(),
//                                   baseline_loss.cast<float>(), rewards.cast<float>(),
//                                   scale.cast<float>(), average);
//            loss.cast<Eigen::half>();
//            reward.cast<Eigen::half>();
//        }
//    };
}  // namespace functor

#define REGISTER(Dev, T, Index)                                     \
  REGISTER_KERNEL_BUILDER(                                          \
        Name("VRClassReward")                                       \
            .Device(DEVICE_##Dev)                                   \
            .TypeConstraint<T>("T")                                 \
            .TypeConstraint<Index>("Tlabels"),                      \
        VRClassRewardOp<Dev##Device, T, Index>);
REGISTER(CPU, float, int32)
REGISTER(CPU, float, int64)
REGISTER(CPU, double, int32)
REGISTER(CPU, double, int64)
//REGISTER(CPU, Eigen::half, int32)
//REGISTER(CPU, Eigen::half, int64)

//#if GOOGLE_CUDA
//REGISTER(GPU, float, int32)
//REGISTER(GPU, float, int64)
//REGISTER(GPU, double, int32)
//REGISTER(GPU, double, int64)
////REGISTER(GPU, Eigen::half, int32)
////REGISTER(GPU, Eigen::half, int64)
//#endif  // GOOGLE_CUDA

#undef REGISTER

}  // namespace tensorflow
