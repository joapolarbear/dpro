#ifndef BYTEPROFILE_COST_MODEL_COMPILE_TO_HLO_H
#define BYTEPROFILE_COST_MODEL_COMPILE_TO_HLO_H

#include <string>
#include "tensorflow/core/lib/core/errors.h"

namespace byteprofile {
namespace xlatools {
    tensorflow::Status CompileToHlo(std::string& graph_path, std::string& config_path, 
            std::string& dump_path_unopt, std::string& dump_path_opt);
}  // namespace tfcompile
}  // namespace tensorflow

#endif // BYTEPROFILE_COST_MODEL_COMPILE_TO_HLO_H