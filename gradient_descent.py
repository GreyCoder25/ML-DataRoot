import math as m
import numpy as np

def dEdu(u, v):
	return 2 * (u * m.exp(v) - 2 * v * m.exp(-u)) * (m.exp(v) + 2 * v * m.exp(-u))

def dEdv(u, v):
	return 2 * (u * m.exp(v) - 2 * v * m.exp(-u)) * (u * m.exp(v) - 2 * m.exp(-u))

def E(u, v):
	return (u * m.exp(v) - 2 * v * m.exp(-u)) ** 2

u, v = 1.0, 1.0
epsilon = 0.00001
eta = 0.1

while True:
	u_prev, v_prev = u, v
	u -= eta * dEdu(u, v)
	v -= eta * dEdv(u, v)
	if (abs(E(u, v) - E(u_prev, v_prev)) < epsilon):
		break

print E(u, v)

