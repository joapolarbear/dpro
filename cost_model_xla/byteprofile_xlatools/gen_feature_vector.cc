#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <iostream>
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/errors.h"

#include "tensorflow/compiler/byteprofile_xlatools/gen_feature_vector.h"

namespace byteprofile {
namespace xlatools {

class HloOpStats {
public:
    HloOpStats(xla::HloOpcode op_code): op_code_(op_code) {}
    
    void add_record(int64_t flops_count, int64_t transcendental_count, 
                    int64_t bytes_accessed, float optimal_seconds, float instr_count) {
        flops_count_ += flops_count;
        transcendental_count_ += transcendental_count;
        bytes_accessed_ += bytes_accessed;
        optimal_seconds_ += optimal_seconds;
        num_ops_recorded_ += 1;
        instr_count_ += instr_count;
    }

    float get_flops_count() { return flops_count_ / num_ops_recorded_;}

    float get_transcendental_count() { return transcendental_count_ / num_ops_recorded_; }

    float get_bytes_accessed() { return bytes_accessed_ / num_ops_recorded_; }

    float get_optimal_seconds() { return optimal_seconds_ / num_ops_recorded_; }

    float get_ops_count() { return num_ops_recorded_; }

    float get_instr_count() { return instr_count_; }

private:
    xla::HloOpcode op_code_;
    float flops_count_ = 0;
    float transcendental_count_ = 0;
    float bytes_accessed_ = 0;
    float optimal_seconds_ = 0;
    float num_ops_recorded_ = 0;
    float instr_count_ = 0;
};

void WriteFloatVector(std::vector<float>& out_vec, std::string& output_path) {
    std::ofstream output_file;
    output_file.open(output_path);
    for (auto value: out_vec) {
        output_file << std::to_string(value) << std::endl;
    }
    output_file.close();
}

int GenFeatureVector(std::string& hlo_module_path, std::string& output_path, 
            float gflops_per_second, float gbytes_per_second) {
    // load hlo module from dumped file
    auto hlo_module = xla::LoadModuleFromFile(hlo_module_path).ValueOrDie();
    // define shape function
    int* dummy_p;
    std::function<int64_t(const xla::Shape&)> shape_func;
    shape_func = [pointer_size=sizeof(dummy_p)] (const xla::Shape& shape) {
        return xla::ShapeUtil::ByteSizeOf(shape, pointer_size);
    };
    xla::HloCostAnalysis analysis(shape_func);
    // set per second rates
    analysis.set_flops_per_second(gflops_per_second * 1e9);
    analysis.set_bytes_per_second(gbytes_per_second * 1e9);
    // run static analysis
    hlo_module->entry_computation()->Accept(&analysis);
    // gerneral stats
    float total_flops_count = analysis.flop_count();
    float total_transcendental_count = analysis.transcendental_count();
    float total_bytes_accessed = analysis.bytes_accessed();
    float total_optimal_seconds = analysis.optimal_seconds();
    // create opcode maps
    std::vector<xla::HloOpcode> all_op_codes = {xla::HloOpcode::kAbs, xla::HloOpcode::kAdd, 
        xla::HloOpcode::kAddDependency, xla::HloOpcode::kAfterAll, xla::HloOpcode::kAllReduce,
        xla::HloOpcode::kAllToAll, xla::HloOpcode::kAtan2, xla::HloOpcode::kBatchNormGrad, 
        xla::HloOpcode::kBatchNormInference, xla::HloOpcode::kBatchNormTraining, 
        xla::HloOpcode::kBitcast, xla::HloOpcode::kBitcastConvert, xla::HloOpcode::kBroadcast, 
        xla::HloOpcode::kCall, xla::HloOpcode::kCeil, xla::HloOpcode::kCholesky, 
        xla::HloOpcode::kClamp, xla::HloOpcode::kCollectivePermute, xla::HloOpcode::kClz, 
        xla::HloOpcode::kCompare, xla::HloOpcode::kComplex, xla::HloOpcode::kConcatenate,
        xla::HloOpcode::kConditional, xla::HloOpcode::kConstant, xla::HloOpcode::kConvert,
        xla::HloOpcode::kConvolution, xla::HloOpcode::kCopy, xla::HloOpcode::kCos, 
        xla::HloOpcode::kCustomCall, xla::HloOpcode::kDivide, xla::HloOpcode::kDomain,
        xla::HloOpcode::kDot, xla::HloOpcode::kDynamicSlice, xla::HloOpcode::kDynamicUpdateSlice,
        xla::HloOpcode::kExp, xla::HloOpcode::kExpm1, xla::HloOpcode::kFft, xla::HloOpcode::kFloor,
        xla::HloOpcode::kFusion, xla::HloOpcode::kGather, xla::HloOpcode::kGetDimensionSize, 
        xla::HloOpcode::kGetTupleElement, xla::HloOpcode::kImag, xla::HloOpcode::kInfeed, 
        xla::HloOpcode::kIota, xla::HloOpcode::kIsFinite, xla::HloOpcode::kLog, 
        xla::HloOpcode::kLog1p, xla::HloOpcode::kAnd, xla::HloOpcode::kNot, xla::HloOpcode::kOr,
        xla::HloOpcode::kXor, xla::HloOpcode::kMap, xla::HloOpcode::kMaximum, xla::HloOpcode::kMinimum,
        xla::HloOpcode::kMultiply, xla::HloOpcode::kNegate, xla::HloOpcode::kOutfeed, 
        xla::HloOpcode::kPad, xla::HloOpcode::kParameter, xla::HloOpcode::kPower, 
        xla::HloOpcode::kReal, xla::HloOpcode::kRecv, xla::HloOpcode::kRecvDone, 
        xla::HloOpcode::kReduce, xla::HloOpcode::kReducePrecision, xla::HloOpcode::kReduceWindow, 
        xla::HloOpcode::kRemainder, xla::HloOpcode::kReplicaId, xla::HloOpcode::kReshape,
        xla::HloOpcode::kReverse, xla::HloOpcode::kRng, xla::HloOpcode::kRoundNearestAfz,
        xla::HloOpcode::kRsqrt, xla::HloOpcode::kScatter, xla::HloOpcode::kSelect, 
        xla::HloOpcode::kSelectAndScatter, xla::HloOpcode::kSend, xla::HloOpcode::kSendDone,
        xla::HloOpcode::kShiftLeft, xla::HloOpcode::kShiftRightArithmetic, 
        xla::HloOpcode::kShiftRightLogical, xla::HloOpcode::kSign, xla::HloOpcode::kSin, 
        xla::HloOpcode::kSlice, xla::HloOpcode::kSort, xla::HloOpcode::kSqrt, xla::HloOpcode::kSubtract,
        xla::HloOpcode::kTanh, xla::HloOpcode::kTrace, xla::HloOpcode::kTranspose,
        xla::HloOpcode::kTriangularSolve, xla::HloOpcode::kTuple, xla::HloOpcode::kTupleSelect,
        xla::HloOpcode::kWhile};
    
    std::unordered_map<xla::HloOpcode, HloOpStats> stats_map;
    for (auto op_code: all_op_codes) {
        stats_map.insert({op_code, HloOpStats(op_code)});
    }
    // iterate through hlo module
    for (xla::HloComputation* comp : hlo_module->computations()) {
        for (xla::HloInstruction* instr : comp->instructions()) {
            int64_t flops_count = analysis.flop_count(*instr);
            int64_t transcendental_count = analysis.transcendental_count(*instr);
            int64_t bytes_accessed = analysis.bytes_accessed(*instr);
            float optimal_seconds = analysis.optimal_seconds(*instr);
            float instruction_count = 0;
            if (instr->opcode() == xla::HloOpcode::kFusion) {
                instruction_count = instr->fused_instruction_count();
            }
            if (stats_map.find(instr->opcode()) != stats_map.end()) {
                stats_map.at(instr->opcode()).add_record(flops_count, transcendental_count, 
                                        bytes_accessed, optimal_seconds, instruction_count);
            }
        }
    }
    // generate vector
    std::vector<float> feature_vector;
    feature_vector.push_back(total_flops_count);
    feature_vector.push_back(total_transcendental_count);
    feature_vector.push_back(total_bytes_accessed);
    feature_vector.push_back(total_optimal_seconds);
    for (auto op_code: all_op_codes) {
        float num_ops = stats_map.at(op_code).get_ops_count();
        float flops_count = stats_map.at(op_code).get_flops_count();
        float transcendental_count = stats_map.at(op_code).get_transcendental_count();
        float bytes_accessed = stats_map.at(op_code).get_bytes_accessed();
        float optimal_seconds = stats_map.at(op_code).get_optimal_seconds();
        feature_vector.push_back(num_ops);
        feature_vector.push_back(std::isnan(flops_count) ? 0 : flops_count);
        feature_vector.push_back(std::isnan(transcendental_count) ? 0 : transcendental_count);
        feature_vector.push_back(std::isnan(bytes_accessed) ? 0 : bytes_accessed);
        feature_vector.push_back(std::isnan(bytes_accessed) ? 0 : optimal_seconds);
        if (op_code == xla::HloOpcode::kFusion) {
            float instr_count = stats_map.at(op_code).get_instr_count();
            feature_vector.push_back(std::isnan(instr_count) ? 0 : instr_count);
        }
    }
    WriteFloatVector(feature_vector, output_path);
    return 0;
}

} // namespace xlatools
} // namespace byteprofile

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "Usage: gen_feature_vector [hlo_module_path] [output_path] [GFLOPS] [GBPS]" << std::endl;
        return 0;
    }
    std::string hlo_module_path(argv[1]);
    std::string output_path(argv[2]);
    float gflops = std::stof(argv[3]);
    float gbps = std::stof(argv[4]);
    byteprofile::xlatools::GenFeatureVector(hlo_module_path, output_path, gflops, gbps);
    return 0;
}