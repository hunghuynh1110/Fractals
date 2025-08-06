import numpy as np
import matplotlib.pyplot as plt

N = 3

r = np.arange(N)
points = np.exp(2.0 * np.pi * 1j * r / N)
print("Points: ", points)

res = 100
w = np.arange(0, res)
unit_circle = np.exp(2.0 * np.pi * 1j * w / res)

start = 0.1 + 0.5j

def comput_new_random_point(start):
    rand_location = np.random.randint(0, N)
    vector = (points[rand_location] - start)/2.0
    new_point = start + vector
    
    return new_point. points[rand_location]


iterations = 20

plt.plot(np.real(unit_circle), np.imag(unit_circle), 'b-')
plt.plot(np.real(points), np.imag(points), 'r')
plt.plot(np.real(start), np.imag(start), 'g.')

next_point = start
for i in range(iterations):
    next_point, point = comput_new_random_point(next_point)
    plt.plot(np.real(next_point), np.imag(next_point), 'k.')
    
plt.show()
print("END")


