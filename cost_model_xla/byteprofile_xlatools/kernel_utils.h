#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <iostream>
#include <bits/stdc++.h> 
#include <sys/stat.h>
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile_data.pb.h"
#include "tensorflow/compiler/xla/service/human_readable_profile_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace tools {

static int32_t kernel_sample_counter = 0;

struct ElementaryOp {
    // Op code
    xla::HloOpcode op_code;
    // shape
    std::vector<xla::Shape> input_shapes;
    xla::Shape output_shape;
    uint64_t hash;
    std::string SerializeToString() {
        std::string s;
        s.append(std::to_string(static_cast<int32_t>(op_code)) + ", " + std::to_string(input_shapes.size()) + ", ");
        for (auto shape: input_shapes) {
            int32_t rank = shape.dimensions_size();
            for (int32_t dim=0; dim<rank; dim++) {
                s.append(std::to_string(shape.dimensions(dim)));
                if (dim != rank - 1 ) {
                    s.append(":");
                }
            }
            s.append(", ");
        }
        int32_t output_rank = output_shape.dimensions_size();
        for (int32_t dim=0; dim<output_rank; dim++) {
            s.append(std::to_string(output_shape.dimensions(dim)));
            if (dim != output_rank - 1 ) {
                s.append(":");
            }
        }
        s.append(", " + std::to_string(hash));
        return s;
    }
};

struct FusedOp {
    std::string name;
    xla::HloInstruction::FusionKind fusion_kind;
    std::vector<ElementaryOp> sub_ops;
    uint64_t Hash() {
        std::vector<uint64_t> hash_values;
        for (auto elem_op: sub_ops) {
            hash_values.push_back(elem_op.hash);
        }
        std::sort(hash_values.begin(), hash_values.end());
        uint64_t hash = 0;
        for (auto hash_v: hash_values) {
            hash = (hash + (324723947 + hash_v)) ^93485734985;
        }
        return hash;
    }
    std::string SerializeToString() {
        std::string s;
        s.append(std::to_string(static_cast<int32_t>(fusion_kind)) + ", " + std::to_string(Hash()) + "\n");
        for (auto elem_op: sub_ops) {
            s.append(elem_op.SerializeToString() + "\n");
        }
        return s;
    }
};

std::unordered_map<std::string, int64_t> ParseProfile(std::string profile_path) {
    std::ifstream profile_file;
    std::cout << "Before opening profile." << std::endl;
    profile_file.open(profile_path);
    std::cout << "After opening profile, before parsing" << std::endl;
    xla::HloExecutionProfileData profile;
    profile.ParseFromIstream(&profile_file);
    std::cout << "After parsing profile, getting printer data" << std::endl;
    auto printer_data = profile.printer_data();
    std::unordered_map<std::string, int64_t> name2cycle;
    for(auto comp_info: printer_data.computation_infos()) {
        for (auto instr_info: comp_info.instruction_infos()) {
            std::string short_name;
            if (instr_info.short_name().at(0) == '%') {
                short_name = instr_info.short_name().substr(1);
            } else {
                short_name = instr_info.short_name();
            }
            short_name = short_name.substr(0, short_name.find("="));
            // trim whitespaces off short_name
            size_t first = short_name.find_first_not_of(' ');
            size_t last = short_name.find_last_not_of(' ');
            short_name = short_name.substr(first, (last-first+1));
            if (short_name.find("fusion") != std::string::npos) {
                // a fusion node, replace with computation name
                std::string comp_name = instr_info.short_name();
                comp_name = comp_name.substr(comp_name.find("calls") + 7, std::string::npos);
                comp_name = comp_name.substr(0, comp_name.find_first_of(','));
                short_name = comp_name;
            }
            int64_t cycle_count = profile.profile_counters(instr_info.profile_index());
            name2cycle[short_name] = cycle_count;
            std::cout << "name: " << short_name << " , cycles: " << cycle_count << std::endl;
        }
    }
    std::cout << "Finished getting cycle data." << std::endl;
    profile_file.close();
    return name2cycle;
}

std::pair<std::unordered_map<std::string, FusedOp>, std::unordered_map<std::string, ElementaryOp>> ExtractKernelFeature(std::string hlo_path) {
    auto hlo_module = xla::LoadModuleFromFile(hlo_path).ValueOrDie();
    std::unordered_map<std::string, FusedOp> name2subops;
    std::unordered_map<std::string, ElementaryOp> name2elementaries;
    for (xla::HloComputation* comp: hlo_module->computations()) {
        if (comp->IsFusionComputation()) {
            FusedOp fused_op;
            std::vector<ElementaryOp> elem_ops;
            for (xla::HloInstruction* instr : comp->instructions()) {
                ElementaryOp elem_op;
                elem_op.op_code = instr->opcode();
                int64_t operand_count = instr->operand_count();
                for (int64_t i=0; i< operand_count; i++) {
                    const xla::HloInstruction* operand_instr = instr->operand(i);
                    elem_op.input_shapes.push_back(operand_instr->shape());
                }
                elem_op.output_shape = instr->shape();
                elem_op.hash = instr->Hash();
                elem_ops.push_back(elem_op);
            }
            fused_op.sub_ops = elem_ops;
            name2subops[comp->name()] = fused_op;
        }
    }
    for (xla::HloInstruction* instr : hlo_module->entry_computation()->instructions()) {
        if (instr->opcode() != xla::HloOpcode::kFusion) {
            ElementaryOp ele;
            ele.op_code = instr->opcode();
            ele.hash = instr->Hash();
            name2elementaries[instr->name()] = ele;
        } else {
            // fusion op
            instr = dynamic_cast<xla::HloFusionInstruction*>(instr);
            name2subops.at(instr->fused_instructions_computation()->name()).fusion_kind = instr->fusion_kind();
        }
    }
    return std::make_pair(name2subops, name2elementaries);
}

class KernelStatsCollector {
public:
    KernelStatsCollector() = default;

    void Collect(std::string profile_path) {
        std::cout << "Before ParseProfile." << std::endl;
        auto name2cycle = ParseProfile(profile_path);
        for (auto kv: name2cycle) {
            auto name = kv.first;
            auto cycles = kv.second;
            if (name2cyclesum.find(name) == name2cyclesum.end()) {
                name2cyclesum[name] = 0;
            }
            if (name2occurences.find(name) == name2occurences.end()) {
                name2occurences[name] = 0;
            }
            name2cyclesum.at(name) += cycles;
            name2occurences.at(name) += 1;
        }
    }

    bool Dump(float clock_rate_ghz, std::string hlo_path, std::string dump_dir_path) {
        // name2cyclesum and name2occurences contains all instructions' cycle count in entry computation
        // name2subops contains fused op's details in HLO module
        std::unordered_map<std::string, FusedOp> name2subops;
        std::unordered_map<std::string, ElementaryOp> name2elementaries;
        std::tie(name2subops, name2elementaries) = ExtractKernelFeature(hlo_path);
        // compute paths
        std::string elemop_path = dump_dir_path + "/elementary_ops.txt";
        std::string label_path = dump_dir_path + "/labels.txt";
        std::string feature_dir = dump_dir_path + "/features";
        mkdir(feature_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
        std::ofstream elemop_file;
        elemop_file.open(elemop_path, std::ios_base::app);
        // write elementary ops to cache
        for (auto kv: name2elementaries) {
            auto ele_op = kv.second;
            auto name = kv.first;
            if (name2cyclesum.find(name) != name2cyclesum.end()) {
                auto total_cycles = name2cyclesum.at(name);
                auto total_occurences = name2occurences.at(name);
                auto time_in_us = (float)total_cycles / total_occurences / clock_rate_ghz / 1000;
                elemop_file << ele_op.hash << " : " << time_in_us << std::endl;
            } else {
                std::cerr << "Cannot find elementary op " << name << " in name2cycle." << std::endl;
                return false;
            }
        }
        elemop_file.close();
        std::ofstream label_file;
        label_file.open(label_path, std::ios_base::app);
        // write features and labels
        for (auto kv: name2subops) {
            auto name = kv.first;
            auto fused_op = kv.second;
            // features
            std::string feature_path = feature_dir + "/" + std::to_string(kernel_sample_counter) + ".txt";
            std::cout << "Writing features for " << name << " in " << feature_path << std::endl;
            std::ofstream feature_file;
            feature_file.open(feature_path);
            feature_file << fused_op.SerializeToString();
            feature_file.close();
            std::cout << fused_op.SerializeToString() << std::endl;
            // labels
            if (name2cyclesum.find(name) != name2cyclesum.end()) {
                auto total_cycles = name2cyclesum.at(name);
                auto total_occurences = name2occurences.at(name);
                auto time_in_us = (float)total_cycles / total_occurences / clock_rate_ghz / 1000; 
                label_file << std::to_string(kernel_sample_counter) << " : " << time_in_us << std::endl;
            } else {
                std::cerr << "Cannot find fused op " << name << " in name2cycle." << std::endl;
                return false;
            }
            kernel_sample_counter ++;
        }
        return true;
    }

private:
    std::unordered_map<std::string, int64_t> name2cyclesum;
    std::unordered_map<std::string, int64_t> name2occurences;
};


} // namespace tools
} // namespace xla