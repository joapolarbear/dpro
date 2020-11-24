#include <string>
#include <fstream>
#include "tensorflow/compiler/xla/service/hlo_execution_profile_data.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/human_readable_profile_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

void ParseAndPrintProfile(std::string profile_path) {
    std::ifstream profile_file;
    profile_file.open(profile_path);
    HloExecutionProfileData profile;
    profile.ParseFromIstream(&profile_file);
    auto printer_data = profile.printer_data();

    for(auto comp_info: printer_data.computation_infos()) {
        std::cout << "========================================" << std::endl;
        std::cout << "[Computation] "<< comp_info.name() << " : total " << profile.profile_counters(comp_info.profile_index()) << " cycles" << std::endl;
        std::string tmp_name;
        for (auto instr_info: comp_info.instruction_infos()) {
            if (instr_info.short_name().at(0) == '%') {
                tmp_name = instr_info.short_name().substr(1);
            } else {
                tmp_name = instr_info.short_name();
            }
            std::cout << "\t[Instr] " << tmp_name.substr(0, tmp_name.find(":")) << " : " << profile.profile_counters(instr_info.profile_index()) << " cycles" << std::endl;
        }
        std::cout << std::endl;
    }
}
} // namespace xla

int main(int argc, char const *argv[])
{
    if (argc != 2) {
        std::cout << "Usage: prog_name profile_path" << std::endl;
        return 0;
    }
    std::string path(argv[1]);
    xla::ParseAndPrintProfile(path);
    return 0;
}

