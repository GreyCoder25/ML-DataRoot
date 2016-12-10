import numpy as np
from numpy import linalg as LA
import numpy.random as rnd

class KMeansClustering:

	def __init__(self, num_clusters, borders):
		self.num_clusters = num_clusters
		self.xmin, self.xmax, self.ymin, self.ymax = borders

	def clustering(self, X):
		self.N_ = X.shape[0]
		self.labels_ = np.zeros(self.N_)

		self.centers_ = np.array(zip(rnd.uniform(self.xmin, self.xmax, self.num_clusters),
									 rnd.uniform(self.ymin, self.ymax, self.num_clusters)))

		while True:
			centers_prev = self.centers_.copy()
			for i in range(X.shape[0]):
				distances = [LA.norm(X[i] - center) for center in self.centers_]
				self.labels_[i] = np.argmin(distances)

			for i in range(len(self.centers_)):
				self.centers_[i] = np.mean(X[np.where(self.labels_ == i)], axis = 0)
			if (np.array_equal(centers_prev, self.centers_)):
				break

				

			



