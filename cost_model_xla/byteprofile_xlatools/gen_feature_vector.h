#ifndef BYTEPROFILE_COST_MODEL_GEN_FEATURE_VECTOR_H
#define BYTEPROFILE_COST_MODEL_GEN_FEATURE_VECTOR_H

#include <string>

namespace byteprofile {
namespace xlatools {

int GenFeatureVector(std::string& hlo_module_path, std::string& output_path, 
            float gflops_per_second, float gbytes_per_second);

} // namespace xlatools
} // namespace byteprofile

#endif // BYTEPROFILE_COST_MODEL_GEN_FEATURE_VECTOR_H

