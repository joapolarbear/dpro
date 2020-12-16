import ujson as json
import os
from base import Singleton
import time

@Singleton
class DebugRecorder:
	def __init__(self, path_=None, is_enable=True):
		self.is_enable = is_enable
		self.debug_traces = []
		self.path_ = path_
		self.base_time = self.get_time()
		self.ts_list = []

	def get_time(self):
		return time.time() * 1e6

	def debug_record(self, name, _ts, pid, tid):
		''' Used for debug, collect traces while replaying
			* to optimize the replay algorithm
		'''
		if not self.is_enable:
			return
		self.debug_traces.append({
					"name": name,
					"ts": _ts * 10e6,
					"dur": ((self.get_time() - self.base_time) - _ts) ,
					"pid": pid,
					"ph": "X",
					"tid": tid
				})

	def debug_event_start(self):
		if not self.is_enable:
			return
		self.ts_list.append(self.get_time() - self.base_time)

	def debug_event_end(self, name, pid, tid):
		if not self.is_enable:
			return
		_ts = self.ts_list.pop()
		self.debug_traces.append({
					"name": name,
					"ts": _ts,
					"dur": (self.get_time() - self.base_time - _ts) ,
					"pid": pid,
					"ph": "X",
					"tid": tid
				})

	def dump_traces(self, path_=None):
		if not self.is_enable:
			return
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

