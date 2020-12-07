import argparse
from .xlatools_impl import compile_to_hlo

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compile to HLO.')
    parser.add_argument('--graph_path', required=True, type=str, 
                        help='Path to the subgraph def.')
    parser.add_argument('--config_path', required=True, type=str, 
                        help='Config path.')
    parser.add_argument('--unopt', required=True, type=str, 
                        help='unopt.')
    parser.add_argument('--opt', required=True, type=str, 
                        help='opt.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    compile_to_hlo(args.graph_path, args.config_path, args.unopt, args.opt)
