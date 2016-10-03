
// See docs in ../contrib/reinforce/ops/reinforce_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/contrib/reinforce/kernels/reinforce_ops.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
using random::PhiloxRandom;
using random::SingleSampleAdapter;

// The default implementation of the functor, which should never be invoked
// But we still need to provide implementation for now for the linker to work,
// since we do not support all the distributions yet.
template <typename Device, class Distribution>
struct FillPhiloxRandom {
    typedef typename Distribution::ResultElementType T;
    void operator()(OpKernelContext*, const Device&, random::PhiloxRandom gen,
                    T* data, int64 size, Distribution dist) {
        LOG(FATAL) << "Default FillPhiloxRandom should not be executed.";
    }
};

// A class to fill a specified range of random groups
template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomTask;

// Specialization for distribution that takes a fixed number of samples for
// each output.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, false> {
    typedef typename Distribution::ResultElementType T;
    static void Run(random::PhiloxRandom gen, T* data, int64 size,
                    int64 start_group, int64 limit_group, Distribution dist) {
        const int kGroupSize = Distribution::kResultElementCount;

        gen.Skip(start_group);
        int64 offset = start_group * kGroupSize;

        // First fill all the full-size groups
        int64 limit_group_full = std::min(limit_group, size / kGroupSize);
        for (int64 index = start_group; index < limit_group_full; ++index) {
            auto samples = dist(&gen);
            std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
            offset += kGroupSize;
        }

        // If there are any remaining elements that need to be filled, process them
        if (limit_group_full < limit_group) {
            int64 remaining_size = size - limit_group_full * kGroupSize;
            auto samples = dist(&gen);
            std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
        }
    }
};

// Specialization for distribution that takes a variable number of samples for
// each output. This will be slower due to the generality.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, true> {
    typedef typename Distribution::ResultElementType T;
    static const int64 kReservedSamplesPerOutput = 256;

    static void Run(random::PhiloxRandom base_gen, T* data, int64 size,
                    int64 start_group, int64 limit_group, Distribution dist) {
        const int kGroupSize = Distribution::kResultElementCount;

        static const int kGeneratorSkipPerOutputGroup =
                kGroupSize * kReservedSamplesPerOutput /
                PhiloxRandom::kResultElementCount;

        int64 offset = start_group * kGroupSize;

        // First fill all the full-size groups
        int64 limit_group_full = std::min(limit_group, size / kGroupSize);
        int64 group_index;
        for (group_index = start_group; group_index < limit_group_full;
             ++group_index) {
            // Reset the generator to the beginning of the output group region
            // This is necessary if we want the results to be independent of order
            // of work
            PhiloxRandom gen = base_gen;
            gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
            SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

            auto samples = dist(&single_samples);
            std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
            offset += kGroupSize;
        }

        // If there are any remaining elements that need to be filled, process them
        if (limit_group_full < limit_group) {
            PhiloxRandom gen = base_gen;
            gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
            SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

            int64 remaining_size = size - limit_group_full * kGroupSize;
            auto samples = dist(&single_samples);
            std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
        }
    }
};

// Partial specialization for CPU to fill the entire region with randoms
// It splits the work into several tasks and run them in parallel
template <class Distribution>
struct FillPhiloxRandom<CPUDevice, Distribution> {
    typedef typename Distribution::ResultElementType T;
    void operator()(OpKernelContext* context, const CPUDevice&,
                    random::PhiloxRandom gen, T* data, int64 size,
                    Distribution dist) {
        const int kGroupSize = Distribution::kResultElementCount;

        auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

        int64 total_group_count = (size + kGroupSize - 1) / kGroupSize;

        const int kGroupCost =
                random::PhiloxRandom::kResultElementCount *
                (random::PhiloxRandom::kElementCost + Distribution::kElementCost);
        Shard(worker_threads.num_threads, worker_threads.workers, total_group_count,
              kGroupCost,
              [&gen, data, size, dist](int64 start_group, int64 limit_group) {
                  FillPhiloxRandomTask<
                          Distribution,
                          Distribution::kVariableSamplesPerOutput>::Run(gen, data, size,
                                                                        start_group,
                                                                        limit_group,
                                                                        dist);
              });
    }
};

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from ReinforceNormalEigenImpl.
template <typename TYPE>
struct ReinforceNormalFunctor<CPUDevice, TYPE> {
    void operator()(const CPUDevice& d,
                    typename TTypes<TYPE>::ConstFlat means,
                    typename TTypes<TYPE>::Flat stddev,
                    typename TTypes<TYPE>::Flat output,
                    bool stochastic) {
        ReinforceNormalEigenImpl<CPUDevice, TYPE>::Compute(d, means, stddev,
                                                           output, stochastic);
    }
};
}

template <typename Device, typename TYPE, class Distribution>
class ReinforceNormalOp : public OpKernel {
public:
    explicit ReinforceNormalOp(OpKernelConstruction* context)
            : OpKernel(context) {
        OP_REQUIRES_OK(context, generator_.Init(context));
    }

    void Compute(OpKernelContext* context) override {

        const Tensor& means = context->input(0);
        const Tensor& stddevs = context->input(1);
        const Tensor& stochastic = context->input(2);
//        OP_REQUIRES(context, TensorShapeUtils::IsVector(means.shape()),
//                    errors::InvalidArgument("mean must be 1-D, but got shape ",
//                                            means.shape().DebugString()));
//        OP_REQUIRES(context, stddevs.dims() == 1 || stddevs.shape() == means.shape(),
//                    errors::InvalidArgument(
//                            "Input stddevs should be a scalar or tensor of the same shape as means, got shape: ",
//                            stddevs.shape().DebugString()));

        Tensor stddevs_tmp;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<TYPE>::value,
                                                       means.shape(), &stddevs_tmp));
        auto stddevs_tensor = stddevs_tmp.flat<TYPE>();
        if (stddevs.dims() == 0) {
            TYPE stddevs_ = stddevs.flat<TYPE>().data()[0];
            stddevs_tensor.setConstant(stddevs_);
        }
        else {
            stddevs_tensor = stddevs.flat<TYPE>();
        }

        bool stochastic_ = stochastic.flat<bool>().data()[0];

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, means.shape(), &output));
        auto output_flat = output->flat<TYPE>();

        functor::FillPhiloxRandom<Device, Distribution>()(
                context, context->eigen_device<Device>(),
                // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
                // it just here.
                generator_.ReserveRandomOutputs(output_flat.size(), 256),
                output_flat.data(), output_flat.size(), Distribution());

        functor::ReinforceNormalFunctor<Device, TYPE> functor;
        functor(context->eigen_device<Device>(), means.flat<TYPE>(),
                stddevs_tensor, output->flat<TYPE>(), stochastic_);
    }

private:
    GuardedPhiloxRandom generator_;
};

#define REGISTER(Dev, TYPE)                                                        \
  REGISTER_KERNEL_BUILDER(                                                         \
        Name("ReinforceNormal")                                                    \
            .Device(DEVICE_##Dev)                                                  \
            .TypeConstraint<TYPE>("T"),                                            \
        ReinforceNormalOp<Dev##Device, TYPE, random::NormalDistribution<           \
                                                random::PhiloxRandom, TYPE> >);
REGISTER(CPU, float)
REGISTER(CPU, double)

//#if GOOGLE_CUDA
//REGISTER(GPU, float)
//REGISTER(GPU, double)
//#endif  // GOOGLE_CUDA

#undef REGISTER

}  // namespace tensorflow
