#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"

Status HloCostAnalysis::HandleCustomCall(const HloInstruction* custom_call) {
  // Mark applicable fields as "unknown", since we don't know what CustomCall
  // does.  This is better than returning an error, which would stop iteration,
  // and therefore would prevent us from getting *any* stats for a computation
  // which contains a CustomCall.
  auto custom_call_instr = Cast<HloCustomCallInstruction>(custom_call);
  if (custom_call_instr->custom_call_target() == "__cublas$gemm") {
    int32_t operand_count = custom_call_instr->operands().size();
    auto backend_config = custom_call_instr->backend_config<gpu::GemmBackendConfig>().ValueOrDie();
    auto dot_dim_numbers = backend_config.dot_dimension_numbers();
    int32_t lhs_contract_dim = dot_dim_numbers.lhs_contracting_dimensions()[0];
    int32_t rhs_contract_dim = dot_dim_numbers.rhs_contracting_dimensions()[0];
    int32_t lhs_sum_dims = lhs_contract_dim;
    int32_t rhs_sum_dims = rhs_contract_dim;
    for (auto dim: dot_dim_numbers.lhs_batch_dimensions()) lhs_sum_dims += dim;
    for (auto dim: dot_dim_numbers.rhs_batch_dimensions()) rhs_sum_dims += dim;
    int32_t lhs_remain_dim = custom_call_instr->operands()[0]->shape().rank()*(custom_call_instr->operands()[0]->shape().rank() - 1)/2 - lhs_sum_dims;
    int32_t rhs_remain_dim = custom_call_instr->operands()[1]->shape().rank()*(custom_call_instr->operands()[1]->shape().rank() - 1)/2 - rhs_sum_dims;
    int32_t batch_size = backend_config.batch_size();
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
    current_properties_[kBytesAccessedKey] = memory;
    current_properties_[kFlopsKey] = flops;
  } else {
    current_properties_[kOptimalSecondsKey] = -1;
    current_properties_[kBytesAccessedKey] = -1;
    SetOutputBytesAccessed(-1);
    for (int i = 0; i < custom_call->operand_count(); ++i) {
      SetOperandBytesAccessed(i, -1);
    }
    current_properties_[kFlopsKey] = -1;
    current_should_compute_bottleneck_time_ = false;
  }
  return Status::OK();
}