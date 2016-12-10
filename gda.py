import numpy as np
import numpy.linalg as LA

class GDA:

	def fit(self, X, y):

		self.m_ = X.shape[0]
		self.n_ = X.shape[1]

		self.phi_ = sum(y == 1) / float(self.m_)
		self.mu0_ = np.mean(np.array([X[i] for i in range(self.m_) if y[i] == 0]), axis = 0)
		self.mu1_ = np.mean(np.array([X[i] for i in range(self.m_) if y[i] == 1]), axis = 0)

		mu_vector = (self.mu0_, self.mu1_)
		dev_vector = np.array([X[i] - mu_vector[int(y[i])] for i in range(self.m_)])
		self.Sigma_ = (1 / float(self.m_)) * np.dot(dev_vector.transpose(), dev_vector)
		self.dist_coef_ = 1 / ( ((2 * np.pi) ** (self.n_ / 2.0)) * (LA.det(self.Sigma_) ** 0.5))

	def p_(self, y):
		return (self.phi_ ** y) * (self.phi_ ** (1 - y))


	def predict(self, X):
		p0 = self.dist_coef_ * np.exp(-0.5 * np.sum(np.dot((X - self.mu0_), LA.inv(self.Sigma_)) * (X - self.mu0_), axis = 1)) * self.p_(0)
		p1 = self.dist_coef_ * np.exp(-0.5 * np.sum(np.dot((X - self.mu1_), LA.inv(self.Sigma_)) * (X - self.mu1_), axis = 1)) * self.p_(1)
		return np.argmax(np.column_stack((p0, p1)), axis = 1)