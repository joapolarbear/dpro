import argparse
from xlatools_impl import gen_feature_vector

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate feature vectors.')
    parser.add_argument('--hlo_module_path', required=True, type=str, 
                        help='Path to the hlo module to compile.')
    parser.add_argument('--output_path', required=True, type=str, 
                        help='Feature vector path.')
    parser.add_argument('--gflops', required=True, type=float, 
                        help='GFLOPS.')
    parser.add_argument('--gbps', required=True, type=float, 
                        help='GBPS.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    gen_feature_vector(args.hlo_module_path, args.output_path, args.gflops, args.gbps)

