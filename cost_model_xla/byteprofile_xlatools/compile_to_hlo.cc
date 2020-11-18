#include <memory>
#include <string>
#include <fstream>
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/stream_executor/stream_executor.h"

#include "tensorflow/compiler/byteprofile_xlatools/compile_to_hlo.h"

namespace byteprofile {
namespace xlatools {

using namespace tensorflow;

const char kUsageHeader[] =
    "tfcompile performs ahead-of-time compilation of a TensorFlow graph,\n"
    "resulting in an object file compiled for your target architecture, and a\n"
    "header file that gives access to the functionality in the object file.\n"
    "A typical invocation looks like this:\n"
    "\n"
    "   $ tfcompile [graph_path] [config_path] [dump_path_unopt] [dump_path_opt]"
    "\n";

Status CompileGraphToOptimizedHLO(GraphDef graph_def, const tf2xla::Config& config, 
                                std::string dump_path_before_opts, std::string dump_path_after_opts) {
    xla::XlaComputation computation;
    se::Platform* cuda_platform =
      se::MultiPlatformManager::PlatformWithName("CUDA").ValueOrDie();
    // std::cout << "Created CUDA platform with name " << cuda_platform->Name() << std::endl;
    xla::LocalClient* client =
      xla::ClientLibrary::GetOrCreateLocalClient(cuda_platform)
          .ValueOrDie();
    // std::cout << "Created local client." << std::endl;
    TF_RETURN_IF_ERROR(ConvertGraphDefToXla(std::move(graph_def), config, 
                        client, &computation));
    // std::cout << "Converted graph to HLO (before optimization)." << std::endl;
    const xla::HloModuleProto& module_proto = computation.proto();
    xla::HloModuleConfig module_config = xla::HloModule::CreateModuleConfigFromProto(module_proto, xla::DefaultDebugOptionsIgnoringFlags()).ValueOrDie();
    std::unique_ptr<xla::HloModule> module = xla::HloModule::CreateFromProto(module_proto, module_config).ValueOrDie();
    // std::cout << "Created HLO module." << std::endl;
    std::string hlo_unopt_str = module->ToString();
    std::ofstream out_file_unopt(dump_path_before_opts);
    out_file_unopt << hlo_unopt_str;
    out_file_unopt.close();
    // std::cout << "Dumped unoptimized HLO module to " << dump_path_before_opts << std::endl;
    // Create gpu compiler
    xla::gpu::NVPTXCompiler nvptx_compiler;
    // std::cout << "Created NVPTX compiler." << std::endl;
    se::StreamExecutor* executor = cuda_platform->ExecutorForDevice(0).ValueOrDie();
    // std::cout << "Created stream executor." << std::endl;
    se::StreamExecutorMemoryAllocator memory_allocator(executor);
    // std::cout << "Created memory allocator." << std::endl;
    std::unique_ptr<xla::HloModule> optimized_module = nvptx_compiler.RunHloPasses(std::move(module), executor, &memory_allocator).ValueOrDie();
    // std::cout << "Finished running HLO passes." << std::endl;
    std::string hlo_str = optimized_module->ToString();
    std::ofstream out_file(dump_path_after_opts);
    out_file << hlo_str;
    out_file.close();
    // std::cout << "Dumped optimized HLO module to " << dump_path_after_opts << std::endl;
    return Status::OK();
}

static Status ReadProtoFile(const string& fname, protobuf::Message* proto) {
  if (absl::EndsWith(fname, ".pbtxt")) {
    return ReadTextProto(Env::Default(), fname, proto);
  } else {
    return ReadBinaryProto(Env::Default(), fname, proto);
  }
}

Status CompileToHlo(std::string& graph_path, std::string& config_path, 
            std::string& dump_path_unopt, std::string& dump_path_opt) {
  // Process config
  tf2xla::Config config;
  // std::cout << "CompileToHlo called." << std::endl;
  // std::cout << "Compile using config: " << config_path << std::endl;
  TF_RETURN_IF_ERROR(ReadProtoFile(config_path, &config));
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  // std::cout << "Config valid." << std::endl;
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(ReadProtoFile(graph_path, &graph_def));
  // std::cout << "Graph file valid." << std::endl;
  
  TF_RETURN_IF_ERROR(CompileGraphToOptimizedHLO(graph_def, config, dump_path_unopt, dump_path_opt));
  // std::cout << "Compilation finished." << std::endl;
  return Status::OK();
}

}  // namespace xlatools
}  // namespace byteprofile

int main(int argc, char** argv) {

  tensorflow::string usage = byteprofile::xlatools::kUsageHeader;
  if (argc > 1 && absl::string_view(argv[1]) == "--help" || argc != 5) {
    std::cerr << usage << "\n";
    return 0;
  }
  std::string graph_path(argv[1]);
  std::string config_path(argv[2]);
  std::string dump_path_unopt(argv[3]);
  std::string dump_path_opt(argv[4]);

//   tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  tensorflow::Status status = byteprofile::xlatools::CompileToHlo(graph_path, config_path, dump_path_unopt, dump_path_opt);
  return 0;
}