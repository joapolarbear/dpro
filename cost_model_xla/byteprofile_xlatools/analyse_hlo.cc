#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <iostream>
#include <bits/stdc++.h> 
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace byteprofile {
namespace xlatools {

int PrintCustomCall(std::string& hlo_module_path) {
    // load hlo module from dumped file
    auto hlo_module = xla::LoadModuleFromFile(hlo_module_path).ValueOrDie();
    int32_t counter = 0;

    // iterate through hlo module
    for (xla::HloComputation* comp : hlo_module->computations()) {
        std::cout << "[comp] Name: " << comp->name() << std::endl;
        for (xla::HloInstruction* instr : comp->instructions()) {
            std::cout << "\t[instr] Name: " << instr->name() << std::endl;
            if (instr->opcode() == xla::HloOpcode::kCustomCall) {
                auto custom_call_instr = xla::Cast<xla::HloCustomCallInstruction>(instr);
                std::cout << "[" + std::to_string(counter)+ "] Target String: " + custom_call_instr->custom_call_target() << std::endl;
                int32_t operand_count = custom_call_instr->operands().size();
                int32_t operand_counter = 0;
                auto backend_config = custom_call_instr->backend_config<xla::gpu::GemmBackendConfig>().ValueOrDie();
                auto dot_dim_numbers = backend_config.dot_dimension_numbers();
                int32_t lhs_contract_dim = dot_dim_numbers.lhs_contracting_dimensions()[0];
                std::cout << "LHS contract dim : " + std::to_string(lhs_contract_dim) << std::endl;
                int32_t rhs_contract_dim = dot_dim_numbers.rhs_contracting_dimensions()[0];
                std::cout << "RHS contract dim : " + std::to_string(rhs_contract_dim) << std::endl;
                int32_t lhs_sum_dims = lhs_contract_dim;
                int32_t rhs_sum_dims = rhs_contract_dim;
                for (auto dim: dot_dim_numbers.lhs_batch_dimensions()) {
                    std::cout << "LHS batch dim : " + std::to_string(dim) << std::endl;
                    lhs_sum_dims += dim;
                }
                for (auto dim: dot_dim_numbers.rhs_batch_dimensions()) {
                    std::cout << "RHS batch dim : " + std::to_string(dim) << std::endl;
                    rhs_sum_dims += dim;
                }
                std::cout << "LHS sum dim : " + std::to_string(lhs_sum_dims) << std::endl;
                std::cout << "RHS sum dim : " + std::to_string(rhs_sum_dims) << std::endl;
                int32_t lhs_remain_dim = custom_call_instr->operands()[0]->shape().rank()*(custom_call_instr->operands()[0]->shape().rank() - 1)/2 - lhs_sum_dims;
                std::cout << "LHS remain dim : " + std::to_string(lhs_remain_dim) << std::endl;
                int32_t rhs_remain_dim = custom_call_instr->operands()[1]->shape().rank()*(custom_call_instr->operands()[1]->shape().rank() - 1)/2 - rhs_sum_dims;
                std::cout << "RHS remain dim : " + std::to_string(rhs_remain_dim) << std::endl;
                int32_t batch_size = backend_config.batch_size();
                std::cout << "batch_size : " + std::to_string(batch_size) << std::endl;
                int32_t M = custom_call_instr->operands()[0]->shape().dimensions(lhs_remain_dim);
                int32_t K = custom_call_instr->operands()[0]->shape().dimensions(lhs_contract_dim);
                assert(K == custom_call_instr->operands()[1]->shape().dimensions(rhs_contract_dim));
                int32_t N = custom_call_instr->operands()[1]->shape().dimensions(rhs_remain_dim);
                int64_t flops = 0;
                int64_t memory = 0;
                if (operand_count == 3) {
                    flops = 2*M*N*(K+2) * batch_size;
                    memory = (M*K + N*K + 2*M*N) * batch_size;
                } else {
                    flops = 2*K*M*N * batch_size;
                    memory = (M*K + N*K + M*N) * batch_size;
                }
                std::cout << "M: " + std::to_string(M) + ", K: " + std::to_string(K) + ", N: " + std::to_string(N) << std::endl;
                std::cout << "FLOPS: " + std::to_string(flops) + ", memory: " + std::to_string(memory) << std::endl;
                for (auto operand: custom_call_instr->operands()) {
                    std::cout << "[" + std::to_string(counter)+ "] Operand " + std::to_string(operand_counter) + " shape : " << operand->shape() << std::endl;
                    std::cout << "shape [0]:" + std::to_string(operand->shape().dimensions(0)) + ", shape [1]: " + std::to_string(operand->shape().dimensions(1)) << std::endl;
                    operand_counter++;
                }
                counter ++;
            }
        }
    }
    return 0;
}

} // namespace xlatools
} // namespace byteprofile

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: analyse_hlo [hlo_module_path]" << std::endl;
        return 0;
    }
    std::string hlo_module_path(argv[1]);
    byteprofile::xlatools::PrintCustomCall(hlo_module_path);
    return 0;
}