/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include <stdio.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <dirent.h>
#include <cstdio>
#include <chrono>

#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/testing.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/compiler/xla/tools/kernel_utils.h"

namespace xla {
namespace tools {
namespace {

// Command-line opts to this tool.  See main() for descriptions of these
// fields.
struct Options {
  Options() {}
  string hlo_path;
  string temp_dir_path;
};

int RealMainForExtract(absl::Span<char* const> args, const Options& opts) {
    string hlo_path = opts.hlo_path;
    string temp_dir_path = opts.temp_dir_path;
    std::cout << "HLO path: " << hlo_path << std::endl;
    std::cout << "temp dir path: " << temp_dir_path << std::endl;
    std::unordered_map<std::string, FusedOp> name2subops;
    std::unordered_map<std::string, ElementaryOp> name2elementaries;
    std::tie(name2subops, name2elementaries) = ExtractKernelFeature(hlo_path);
    // we need to generate 2 type of files:
    // 1. files containing individual kernel's features (features.txt)
    // 2. module config file
    // compute paths
    std::string module_config_path = temp_dir_path + "/module_config.txt";
    std::ofstream module_config_file;
    module_config_file.open(module_config_path);
    std::cout << "Writing config file to " << module_config_path << " ." << std::endl;
    // write elementary ops to cache
    for (auto kv: name2elementaries) {
        auto ele_op = kv.second;
        auto name = kv.first;
        module_config_file << "0, " << ele_op.hash << ", "  << ele_op.op_code << std::endl;
    }
    // write features and labels
    int32_t fused_op_counter = 0;
    for (auto kv: name2subops) {
        auto name = kv.first;
        auto fused_op = kv.second;
        // features
        std::string feature_path = temp_dir_path + "/" + std::to_string(fused_op_counter) + ".txt";
        std::cout << "Writing feature file to " << feature_path << " ." << std::endl;
        std::ofstream feature_file;
        feature_file.open(feature_path);
        feature_file << fused_op.SerializeToString();
        feature_file.close();
        fused_op_counter ++;
        module_config_file << "1, " << fused_op.Hash() << ", " << feature_path << std::endl;
    }
    module_config_file.close();
    return 0;
}

}  // namespace
}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  xla::tools::Options opts;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("hlo_path", &opts.hlo_path,
                       "Path to the HLO file."),
      tensorflow::Flag("temp_dir_path", &opts.temp_dir_path,
                       "Path to temp folder used for xla dump."),
  };
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << usage;
  }

  absl::Span<char* const> args(argv, argc);
  args.remove_prefix(1);  // Pop off the binary name, argv[0]
  return xla::tools::RealMainForExtract(args, opts);
}