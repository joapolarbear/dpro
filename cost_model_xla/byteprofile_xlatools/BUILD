# Byteprofile tools related to XLA

load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
    "tf_cc_binary",
    "tf_cc_test",
)

package(
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],  # Apache 2.0
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["//tensorflow/compiler/xla:internal"],
)

cc_library(
    name = "gen_feature_vector_lib",
    srcs = ["gen_feature_vector.cc", "gen_feature_vector.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/core/platform:errors",
    ],
)

tf_cc_binary(
    name = "gen_feature_vector",
    srcs = ["gen_feature_vector.cc"],
    deps = [
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/core/platform:errors",
    ],
)

tf_cc_binary(
    name = "analyse_hlo",
    srcs = ["analyse_hlo.cc"],
    deps = [
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc",
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/core/platform:errors",
    ],
)

tf_cc_binary(
    name = "extract_cycles",
    srcs = ["extract_cycles.cc"],
    deps = [
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc",
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/core/platform:errors",
    ],
)

cc_library(
    name = "tfcompile_hlo_lib",
    srcs = [
        "compile_to_hlo.cc", "compile_to_hlo.h",
    ],
    deps = [
        "//tensorflow/compiler/aot:embedded_protocol_buffers",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/compiler/tf2xla",
        "//tensorflow/compiler/tf2xla:tf2xla_proto_cc",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service/gpu:nvptx_compiler",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ]
)

tf_cc_binary(
    name = "tfcompile_hlo",
    srcs = [
        "compile_to_hlo.cc",
    ],
    deps = [
        "//tensorflow/compiler/aot/embedded_protocol_buffers",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/compiler/tf2xla",
        "//tensorflow/compiler/tf2xla:tf2xla_proto_cc",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service/gpu:nvptx_compiler",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ]
)

tf_cc_binary(
    name = "byteprofile_xlatools_c_api.so",
    srcs = ["c_api.cc", "c_api.h"],
    #hdrs = ["c_api.h"],
    deps = [
        ":tfcompile_hlo_lib",
        ":gen_feature_vector_lib",
    ],
    linkshared=True,
)