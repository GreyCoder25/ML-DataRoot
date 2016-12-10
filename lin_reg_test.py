import linear_regression as linreg
import numpy as np
import numpy.random as rnd


x_range_left = -1
x_range_right = 1
noise_range_down = -0.15
noise_range_up = 0.15

def f(x, k):
	return k * x + rnd.uniform(noise_range_down, noise_range_up)

N = 100
k = 7

X = rnd.uniform(x_range_left, x_range_right, N).reshape(N, 1)
y = f(X, k)


train_size = int(0.8 * X.shape[0])
 

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

reg1 = linreg.LinearRegression()
reg_lambda = 1
reg1.fit(X, y, reg_lambda)

print np.mean(abs(reg1.predict(X_train) - y_train))
