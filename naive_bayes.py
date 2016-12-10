import numpy as np

class NaiveBayes:

	def fit(self, X, y):
		self.m_ = X.shape[0]
		self.n_ = X.shape[1]

		y1_prob = np.mean(y)
		self.class_prob_ = (1 - y1_prob, y1_prob)

		y0_samples = np.array([X[i] for i in range(self.m_) if y[i] == 0])
		y1_samples = np.array([X[i] for i in range(self.m_) if y[i] == 1])

		self.features_prob1_cl0 = np.mean(y0_samples, axis = 0)
		self.features_prob1_cl1 = np.mean(y1_samples, axis = 0)


	def predict(self, X):
		prediction = []
		for point in X:
			p0 = self.class_prob_[0]					#here must exist more vectorised solution
			for i in range(len(point)):
				if point[i] == 0:
					p0 *= 1 - self.features_prob1_cl0[i]
				else:
					p0 *= self.features_prob1_cl0[i]

			p1 = self.class_prob_[1]
			for i in range(len(point)):
				if point[i] == 0:
					p1 *= 1 - self.features_prob1_cl1[i]
				else:
					p1 *= self.features_prob1_cl1[i]

			prediction.append(np.argmax((p0, p1)))

		return np.array(prediction)
