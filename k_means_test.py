import k_means
import matplotlib.pyplot as plt
from k_means import *

x_range_left = -1
x_range_right = 1
y_range_down = -1
y_range_up = 1

borders = (x_range_left, x_range_right, y_range_down, y_range_up)

N = 100
X = np.array(zip(rnd.uniform(x_range_left, x_range_right, N), rnd.uniform(y_range_down, y_range_up, N)))

num_clusters = 3
clusterer = KMeansClustering(num_clusters, borders)
clusterer.clustering(X)
y = clusterer.labels_

plt.title('k_means')
plt.xlabel('x1')
plt.ylabel('x2')
plt.autoscale(tight = True)


x_plot = X[:, 0]
y_plot = X[:, 1]

draw_labels = ('red', 'green', 'blue', 'black', 'yellow')


for i in range(len(y)):
	plt.scatter(x_plot[i], y_plot[i], color = draw_labels[int(y[i])], marker = 'o')
	
plt.show()