''' Combine trace files
    * Usage
    python3 combine_trace.py files/to/combine path/to/dump/rst pid/names bias
    The first item of the command parameter is the path of the json file to be combined, separated by commas;
    The second item is the path to store the result;
    The third item is the relabel pid for each input file,
    and the bias is used to manually align the time. 
    An example is python3 combine_trace.py path_a, path_b path_rst pid_a, pid_b 0,0
'''
import ujson as json
import os, sys
ALIGN_TIME = True
KEEP_PID = False

def combine_files(files, names, bias, output_path):
    final_traces = []
    for idx, file in enumerate(files):
        with open(file, 'r') as fp:
            traces = json.load(fp)
        if "traceEvents" in traces:
            traces = traces["traceEvents"]
        ts = None
        for trace in traces:
            if ALIGN_TIME and ts is None:
                ts = trace["ts"]
            if not KEEP_PID:
                trace["pid"] = names[idx]
            else:
                trace["pid"] = names[idx] + "." + trace["pid"]
            if ALIGN_TIME:
                trace["ts"] = trace["ts"] - ts
            trace["ts"] += bias[idx]
        final_traces += traces

    with open(output_path, 'w') as fp:
        json.dump(final_traces, fp)

files = sys.argv[1]
output_path = sys.argv[2]
if len(sys.argv) >= 5:
    bias = [float(n)*1000 for n in sys.argv[4].split(",")]
else:
    bias = [0 for _ in files]

files = files.split(",")

if len(files) == 1 and os.path.isdir(files[0]):
    names = sorted(os.listdir(files[0]))
    files = [os.path.join(files[0], n) for n in names]
else:
    names = sys.argv[3].split(",")

combine_files(files, names, bias, output_path)
