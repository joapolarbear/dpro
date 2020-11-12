import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, sys
import math
import arg_utils
args = arg_utils.SingleArg().args


def init_fig_base(cnt):
    h = math.ceil(math.sqrt(cnt))
    w = math.ceil(cnt / h)
    fig_base = w * 100 + h * 10 + 1
    return fig_base, 0

with open(os.path.join(args.path, 'queue_status.json'), 'r') as fp:
	rst = json.load(fp)

MAXIMUM_GROUP = 4
plt.figure(num=1, figsize=(8, 6))
clrs = sns.color_palette("husl", MAXIMUM_GROUP+1)

### shape = (time+num_of_nodes_queued, num_data)
data = np.array(sorted(rst['data'], key=lambda x:x[0])).T

sample_num = 1000
if sample_num is None:
	mask = np.ones(data.shape[1], dtype=bool)
else:
	mask = np.zeros(data.shape[1], dtype=bool)
	sample_idx = np.random.choice(data.shape[1], sample_num, replace=False)
	mask[sample_idx] = True

group_dict = {}
for idx, n in sorted(enumerate(rst['names']), key=lambda x: x[1]):
	group = n.split('->')[0]
	if group not in group_dict:
		group_dict[group] = []
	group_dict[group].append(idx)

fig_base, _ = init_fig_base(min(MAXIMUM_GROUP, len(group_dict)))
for idx, (group, name_idx_list) in enumerate(group_dict.items()):
	if idx >= MAXIMUM_GROUP:
		break
	ax = plt.subplot(fig_base + idx)
	for idx, name_idx in enumerate(name_idx_list):
		ax.plot(data[0][mask]/1000., data[name_idx+1][mask], c=clrs[idx], label=rst['names'][name_idx])
	plt.legend()
	plt.xlabel('Time (ms)')
	plt.ylabel('# of operators being queued')
	plt.title(group)
plt.show()





