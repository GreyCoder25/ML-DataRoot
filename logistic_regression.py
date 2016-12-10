import numpy as np
from numpy import linalg as LA
import random as rnd

class LogisticRegression:

	def predict_proba(self, X, y):
		return 1 / (1 + np.exp(- y * np.dot(self.w_, X)))

	def predict(self, X):
		return np.sign(np.dot(self.w_, X))

	def fit(self, X, y):
		self.N_ = X.shape[0]
		self.w_ = np.zeros(X.shape[1])
		self.eta_ = 0.1

		fitting = True
		der_numerator = (X.transpose() * y).transpose()

		while(fitting):
			w_prev = self.w_.copy()
			self.w_ -= - self.eta_ * (1 / float(self.N_)) * sum((der_numerator.transpose() / (1 + np.exp(y * np.dot(X, self.w_)))).transpose())
			if  LA.norm(self.w_ - w_prev) < 0.001:
				fitting = False

