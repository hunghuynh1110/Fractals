import numpy as np
import matplotlib.pyplot as plt

N = 3

r = np.arange(N)
points = np.exp(2.0*np.pi*1j *r/N)
print("Points: ", points)

res = 100
w = np.arange(0,res)
unit_circle = np.exp(2.0*np.pi*1j *w/res)

start = np.random.randint(0, N)
start_point = points[start]
print("Start point: ", start_point)

plt.plot(np.real(unit_circle), np.imag(unit_circle), 'b-')
plt.plot(np.real(points), np.imag(points), 'r')
plt.plot(np.real(start_point), np.imag(start_point), 'g.')

plt.show()
print("END")

 