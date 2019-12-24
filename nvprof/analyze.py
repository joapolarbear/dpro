
import os



def handle(path):
	with open(path, 'r') as fp:
		s = fp.readlines()
	i = 0

	sta = {}
	while i < len(s):
		if "Device   Context    Stream" in s[i]:
			i += 1
			break
		i += 1
	while i < len(s):
		if len(s[i]) < 162:
			break
		try:
			stream_id = int(s[i][162:168])
		except:
			print(len(s[i]), s[i-1])
			raise
		name = s[i][170:]
		if stream_id not in sta:
			sta[stream_id] = {"cmp": set(), "mem": set()}
		if "memcpy" in name or "memset" in name:
			sta[stream_id]["mem"].add(name)
		else:
			sta[stream_id]["cmp"].add(name)
		i += 1
	for k, v in sta.items():
		print("Stream ID: %-2d => cmp: %-10d : mem %-10d %s" % (k, len(v["cmp"]), len(v["mem"]), '' if len(v["mem"]) <= 2 else str(v["mem"])))
	# print(len(sta))


cur_dir = os.path.abspath(".")

root, dirs, files = list(os.walk(cur_dir, topdown=True))[0]
for file in files:
	if "txt" in file:
		cur_path = os.path.join(root, file)
		print
		print(cur_path)
		handle(cur_path)