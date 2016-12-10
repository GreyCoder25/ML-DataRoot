import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt
import gda

x_range_left = -1
x_range_right = 1
y_range_down = -1
y_range_up = 1


def f(point, w_true):
	return np.sign(np.dot(point, w_true[1:]) + w_true[0])

N = 100
w_true = np.array([1.0, 3.0, 4.0])

X = np.array(zip(rnd.uniform(x_range_left, x_range_right, N), rnd.uniform(y_range_down, y_range_up, N)))
y = (np.sign(f(X, w_true)) + 1) / 2

train_size = int(0.8 * X.shape[0])

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

clf = gda.GDA()
clf.fit(X_train, y_train)
print sum(clf.predict(X_test) == y_test) / float(N - train_size)
