import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import ica

coef_range_down = -3
coef_range_up = 3
signal_range_start = -10
signal_range_finish = 10

def f1(x, a, b):
	return a * np.sin(b * x)

def f2(x, a, b):
	return a * np.tanh(b * x)

num_sources = 2
A = rnd.uniform(coef_range_down, coef_range_up, num_sources ** 2).reshape((num_sources, num_sources))

a1, b1 = rnd.uniform(coef_range_down, coef_range_up, 2)
a2, b2 = rnd.uniform(coef_range_down, coef_range_up, 2)

signal_range = np.arange(signal_range_start, signal_range_finish, 0.05)

s1 = f1(signal_range, a1, b1)
s2 = f2(signal_range, a2, b2)

s = np.column_stack((s1, s2))
x = np.dot(A, s.transpose()).transpose()

source_predictor = ica.ICA()
source_predictor.predict_sources(x)
s_predicted = source_predictor.predicted_sources_


plt.figure(1)
plt.subplot(211)
plt.title('true sources')
plt.plot(signal_range, s1, 'bo')

plt.subplot(212)
plt.plot(signal_range, s2, 'ro')

plt.figure(2)
plt.subplot(211)
plt.title('receivers')
plt.plot(signal_range, x.transpose()[0], 'bo')

plt.subplot(212)
plt.plot(signal_range, x.transpose()[1], 'ro')

plt.figure(3)
plt.subplot(211)
plt.title('predicted_sources_')
plt.plot(signal_range, s_predicted[0], 'bo')

plt.subplot(212)
plt.plot(signal_range, s_predicted[1], 'ro')


plt.show()
