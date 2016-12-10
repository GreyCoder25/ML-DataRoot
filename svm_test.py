import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt
import svm
from scipy.optimize import minimize

x_range_left = -1
x_range_right = 1
y_range_down = -1
y_range_up = 1


def f(point, w):
	return np.sign(np.dot(point, w[1:]) + w[0])

N = 20
w_true = np.array([1.0, 3.0, 4.0])

X = np.array(zip(rnd.uniform(x_range_left, x_range_right, N), rnd.uniform(y_range_down, y_range_up, N)))
#X = np.column_stack((np.ones(X.shape[0])[:, np.newaxis], X))
y = np.sign(f(X, w_true))

clf = svm.SVM()
clf.fit(X, y)
print clf.alphas_



plt.title('SVM')
plt.xlabel('x1')
plt.ylabel('x2')
plt.autoscale(tight = True)


f_plot_x = np.array([1, -1])
f_plot_y = [-w_true[1] / w_true[2] -w_true[0] / w_true[2], w_true[1] / w_true[2] -w_true[0] / w_true[2]]
plt.plot(f_plot_x, f_plot_y, linewidth = 2, color = 'green')


g_plot_x = np.array([1, -1])
w = clf.w_
b = 0
g_plot_y = [-w[0] / w[1] -b / w[1], w[0] / w[1] -b / w[1]]
#g_plot_y = (-w[:-1] / w[-1]) * g_plot_x 
plt.plot(g_plot_x, g_plot_y, linewidth = 2, color = 'red')


x_plot = X[:, 0]
y_plot = X[:, 1]

for i in range(len(y)):
	if y[i] == 1:
		plt.scatter(x_plot[i], y_plot[i], color = 'red', marker = 'o', label = 'red')
	elif y[i] == -1:
		plt.scatter(x_plot[i], y_plot[i], color = 'blue', marker = 'x', label = 'blue')

plt.show()

