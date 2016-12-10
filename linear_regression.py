import numpy as np

class LinearRegression:

	def fit(self, X, y, l):
		self.lambda_ = l
		self.w_ = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X) + self.lambda_ * np.eye(X.shape[1])), X.transpose()), y)

	def predict(self, point):
		return np.dot(point, self.w_)
