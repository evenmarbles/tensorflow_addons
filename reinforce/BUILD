# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")


# OpKernel libraries ----------------------------------------------------------

tf_custom_op_library(
    name = "python/ops/_vr_class_reward_op.so",
    srcs = [
        "kernels/vr_class_reward_op.cc",
        "kernels/vr_class_reward_op.h",
        "ops/vr_class_reward_op.cc",
    ],
    gpu_srcs = [
        "kernels/vr_class_reward_op_gpu.cu.cc",
        "kernels/vr_class_reward_op.h",
    ],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/kernels:bounds_check_lib",
    ],
)

tf_custom_op_library(
    name = "python/ops/_reinforce_ops.so",
    srcs = [
        "kernels/reinforce_ops.cc",
        "kernels/reinforce_ops.h",
        "ops/reinforce_ops.cc",
    ],
    gpu_srcs = [
        "kernels/reinforce_ops_gpu.cu.cc",
        "kernels/reinforce_ops.h",
    ],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/kernels:bounds_check_lib",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

py_library(
    name = "reinforce_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    data = [
        ":python/ops/_vr_class_reward_op.so",
        ":python/ops/_reinforce_ops.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

cuda_py_tests(
    name = "vr_class_reward_op_test",
    size = "small",
    srcs = ["python/kernel_tests/vr_class_reward_op_test.py"],
    additional_deps = [
        ":reinforce_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
