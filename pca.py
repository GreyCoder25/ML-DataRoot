import numpy as np
import numpy.linalg as LA

class PCA:
	 def preprocessing_(self, X):
	 	self.m_ = X.shape[0]
	 	self.mu_ = np.mean(X, axis = 0)
	 	X -= self.mu_

	 	self.dev_ = np.mean(X ** 2, axis = 0)
	 	X /= np.sqrt(self.dev_)


	 def dim_reduce(self, X):
	 	self.preprocessing_(X)
	 	self.eigvals_, self.eigvects_ = LA.eig((1 / float(self.m_)) * np.dot(X.transpose(), X))

	 	self.new_basis_ = self.eigvects_[:, np.argmax(self.eigvals_)]

	 	X = np.dot(self.new_basis_, X.transpose())