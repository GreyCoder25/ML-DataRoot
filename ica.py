import numpy as np
import numpy.linalg as LA
import numpy.random as rnd
from scipy.optimize import minimize

class ICA:

	def g_(self, x):

		return 1 / (1 + np.exp(-x))

	def g_der_(self, x):

		return np.exp(x) / ((np.exp(x) + 1) ** 2)

	def log_likelihood_(self, W, X):
		W = W.reshape(self.m_, self.m_)
		-np.log(self.g_der_(np.dot(W, X.transpose()))) - np.log(self.W_det_ ** self.m_)

	def predict_sources(self, X):
		self.m_ = X.shape[1]
		self.W_ = np.eye(self.m_)
		self.W_det_ = LA.det(self.W_)

		#self.alpha_ = 0.01
		#self.num_iter_ = 1000

		#for i in range(self.num_iter_):
		#	W_prev = self.W_
		#	self.W_ += self.alpha_ * (np.dot((1 - self.g_(np.dot(self.W_, X.transpose()))), X) 
		#								+ self.m_ * LA.inv(self.W_.transpose()))

		#	print self.W_
		self.W_ = minimize(self.log_likelihood_, self.W_, args = (X),
								  method = 'CG').x

		print self.W_
	

		self.predicted_sources_ = np.dot(self.W_, X.transpose())


