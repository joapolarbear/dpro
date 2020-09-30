import numpy as np
import math
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.utils import check_random_state

def init_fig_base(cnt):
    h = math.ceil(math.sqrt(cnt))
    w = math.ceil(cnt / h)
    fig_base = w * 100 + h * 10 + 1
    return fig_base, 0

class DimReducer:
	def __init__(self, xdata, ydata):
		'''
		xdata: numpy.ndarray
			data needed to reduce dimension, shape = (n_samples, n_dims)
		ydata: numpy.ndarray
			label data, shape = (n_samples, 1)
		'''
		assert xdata.shape[0] > xdata.shape[1], \
			"n_samples should be larger than the dimension: (%d, %d) is given"%(xdata.shape[0], xdata.shape[1])
		assert len(ydata.shape) == 1 or ydata.shape[1] == 1, \
			"label should be a 1 x ndims vector: {} is given".format(ydata.shape)
		self.xdata = xdata
		self.ydata = ydata.flatten()

		max_y = max(self.ydata)
		min_y = min(self.ydata)
		N_CLASS = 10
		class_step = (max_y - min_y) / N_CLASS
		self.ydata_class = np.floor((self.ydata - min_y) / class_step)

	def do_LLE(self, n_comp=2, n_neib=20, show=None):
		from sklearn.manifold import LocallyLinearEmbedding
		lle = LocallyLinearEmbedding(n_components=n_comp, n_neighbors=n_neib) 
		X_reduced = lle.fit_transform(self.xdata)

		if show is not None:
			ax = self.fig.add_subplot(self.fig_base+show, projection='3d')
			plt.title('LLE with k = {}'.format(n_neib), size=12)
			ax.scatter(X_reduced[:, 0], X_reduced[:, 1], self.ydata, c=self.ydata_class)
			ax.view_init(20, -19)

		return X_reduced

	def do_MDS(self, n_comp=2, show=None):
		from sklearn.manifold import MDS
		model = MDS(n_components=n_comp)
		X_reduced = model.fit_transform(self.xdata)
		if show is not None:
			ax = self.fig.add_subplot(self.fig_base+show, projection='3d')
			plt.title('MDS', size=12)
			ax.scatter(X_reduced[:, 0], X_reduced[:, 1], self.ydata, c=self.ydata_class)
			ax.view_init(20, -19)
		return X_reduced

	def do_LDA(self, n_comp=2, show=None):
		from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
		lda = LDA(n_components=n_comp)
		X_reduced = lda.fit_transform(self.xdata, self.ydata_class)
		if show is not None:
			ax = self.fig.add_subplot(self.fig_base+show, projection='3d')
			plt.title('LDA', size=12)
			ax.scatter(X_reduced[:, 0], X_reduced[:, 1], self.ydata, c=self.ydata_class)
			ax.view_init(20, -19)
		return X_reduced

	def do_reduction(self, n_comp=2, algo='LLE', show=True):
		if show:
			self.fig = plt.figure(figsize = (9, 8))
			plt.style.use('default')

		if isinstance(algo, str):
			if show:
				self.fig_base, _ = init_fig_base(1)
			if algo == 'LLE':
				X_reduced = self.do_LLE(n_comp=n_comp, show=0)
			elif algo == 'MDS':
				X_reduced = self.do_MDS(n_comp=n_comp, show=0)
			elif algo == 'LDA':
				X_reduced = self.do_LDA(n_comp=n_comp, show=0)
			else:
				raise ValueError("Invalid algorithm: {}".format(algo))
			if show:
				plt.show()
			return X_reduced
		elif isinstance(algo, list):
			if show:
				self.fig_base, _ = init_fig_base(len(algo))
			ret = []
			for idx, _algo in enumerate(algo):
				if _algo == 'LLE':
					X_reduced = self.do_LLE(n_comp=n_comp, show=idx)
				elif _algo == 'MDS':
					X_reduced = self.do_MDS(n_comp=n_comp, show=idx)
				elif _algo == 'LDA':
					X_reduced = self.do_LDA(n_comp=n_comp, show=idx)
				else:
					raise ValueError("Invalid algorithm: {}".format(_algo))
				ret.append(X_reduced)
			if show:
				plt.show()
			return ret


		


