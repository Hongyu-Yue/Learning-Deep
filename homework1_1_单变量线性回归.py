import matplotlib.pyplot as plt
import numpy as np

alldata = np.loadtxt('D:\python\homewrok\data_sets\ex1data1.txt', delimiter=",")

x = alldata[:, 0]
y = alldata[:, 1]

n = 2
m = len(x)
time = theta0 = theta1 = 0
toler = 0.001
aerfa = 0.01

def h(theta0, theta1, x):
    y = theta0 + theta1 * x
    return y

plt.subplot(211)
plt.scatter(x,y)

while time < 3000:
    cost0 = cost1 = cost = 0
    for i in range(m):
        cost0 = cost0 + h(theta0, theta1, x[i]) - y[i]
        cost1 = cost1 + (h(theta0, theta1, x[i]) - y[i]) * x[i]
        cost = cost + ( h(theta0, theta1, x[i]) - y[i] ) ** 2
    theta0 -= cost0 * aerfa / m
    theta1 -= cost1 * aerfa / m
    time += 1
    J.append(cost/2/m)

plt.plot(x,h(theta0, theta1, x),C='red')
plt.subplot(212)
plt.ylim((4, 7))
plt.plot(range(0,3000),J)
plt.show()
print('y=', theta0, '+', theta1, 'x')
