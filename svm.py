import numpy as np
from scipy.optimize import minimize

class SVM:

	def fit(self, X, y):
		self.N_ = X.shape[0]
		initial_alphas = np.zeros(self.N_)

		bnds = np.array([(0, None) for i in range(self.N_)])
		cons = ({'type': 'eq',
          		 'fun' : self.constr_func_,
          		 'args': y
          		})


		self.alphas_ = minimize(self.minimize_func_, initial_alphas, args = (X, y),
								 bounds = bnds, constraints = cons, method = 'SLSQP').x

		self.w_ = np.sum((self.alphas_ * y)[:, np.newaxis] * X, axis = 0)

	def predict(self, X):
		pass

	def minimize_func_(self, alphas, X, y):
		n = X.shape[0]
		y_tmp = y.reshape(y.shape[0], 1)
		Q = np.dot(y_tmp, y_tmp.transpose())
		Q *= np.dot(X, X.transpose())

		return 0.5 * np.dot(np.dot(alphas, Q), alphas) - np.dot(np.ones(n), alphas)

	def constr_func_(self, alphas, y):
		return np.dot(alphas, y)


