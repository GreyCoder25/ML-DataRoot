import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt
import pca

x_range_left = -1
x_range_right = 1
noise_range_down = -0.7
noise_range_up = 0.7

def f(x, k, b):
	return k * x + b

N = 100
k = -1
b = 0

x1_coords = rnd.uniform(x_range_left, x_range_right, N) 
x2_coords = f(x1_coords, k, b) + rnd.uniform(noise_range_down, noise_range_up, N)

X = np.column_stack((x1_coords, x2_coords))
dim_reducer = pca.PCA()
new_basis = dim_reducer.dim_reduce(X)


plt.title('PCA')
plt.xlabel('x1')
plt.ylabel('x2')
plt.autoscale(tight = True)


for i in range(N):
	plt.scatter(X[:, 0], X[:, 1], color = 'blue', marker = 'o')

#for i in range(N):
#	plt.scatter(np.dot(new_basis, X.transpose()), np.dot(new_basis[1], X[:, 1].transpose()), color = 'green', marker = 'o')

u_plot_x = [0, dim_reducer.new_basis_[0]]
u_plot_y = [0, dim_reducer.new_basis_[1]]
plt.plot(u_plot_x, u_plot_y, linewidth = 2, color = 'red')


plt.show()

	