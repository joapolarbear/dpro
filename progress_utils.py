import sys

class progressBar:
	def __init__(self, start=0, end=0, flush=0.01):
		self.start = start
		self.total_len = end - start
		self.flush = flush
		self.bar_len = 100
		self.percent = -1

	def showBar(self, idx):
		percent = (idx - self.start) / float(self.total_len)
		if percent >= self.percent + self.flush:
			finish = int(self.bar_len * percent)
			sys.stdout.write("[" + "=" * finish + ">" + "." * (self.bar_len-finish) + "] %f %%\r" % (100 * percent))
			sys.stdout.flush()
			self.percent = percent