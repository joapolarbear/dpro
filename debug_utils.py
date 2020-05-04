import json
import os
from base import Singleton
import time

@Singleton
class DebugRecorder:
	def __init__(self, path_=None):
		self.debug_traces = []
		self.path_ = path_
		self.base_time = time.time()

	def debug_record(self, name, _ts, pid, tid):
		''' Used for debug, collect traces while replaying
			* to optimize the replay algorithm
		'''
		self.debug_traces.append({
					"name": name,
					"ts": _ts * 10e6,
					"dur": ((time.time() - self.base_time  ) - _ts) * 1e6,
					"pid": pid,
					"ph": "X",
					"tid": tid
				})

	def debug_event_start(self, name, pid, tid):
		self.debug_traces.append({
					"name": name,
					"ts": (time.time() - self.base_time  ) * 1e6,
					"pid": pid,
					"ph": "B",
					"tid": tid
				})

	def debug_event_end(self, name, pid, tid):
		self.debug_traces.append({
					"name": name,
					"ts": (time.time() - self.base_time  ) * 1e6,
					"pid": pid,
					"ph": "E",
					"tid": tid
				})

	def dump_traces(self, path_=None):
		if path_ is not None:
			trace_path = path_ 
		elif self.path_ is not None:
			trace_path = self.path_
		else:
			raise ValueError("Trace path must be given")

		with open(os.path.join(trace_path, "debug.json"), 'w') as f:
			json.dump({"traceEvents": self.debug_traces,
						"displayTimeUnit": "ms"
							}, f, indent=4)

