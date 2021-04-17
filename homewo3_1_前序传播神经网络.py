import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.optimize as opt

data = scio.loadmat('D:\python\homewrok\data_sets\ex3data1.mat')
weights = scio.loadmat('D:\python\homewrok\data_sets\ex3weights.mat')

x = data['X']
y = data['y']
a = {}
z = {}
theta = {}
theta1 = weights['Theta1']
theta[1] = theta1.T
theta2 = weights['Theta2']
theta[2] = theta2.T

a[1]=x
n=3
def g(x):
    return 1/(1+np.exp(-x))

def plus1(x):
    return np.c_[np.ones(len(x)),x]

for i in range(1,n):
    a[i] = plus1(a[i])
    z[i+1] = a[i] @ theta[i]
    a[i+1] = g(z[i+1])



#z4 = a3 @ theta3T
#a4 = g(z4)


def predict(prob):
    y_predict = np.zeros((prob.shape[0], 1))
    for i in range(prob.shape[0]):
        y_predict[i] = np.unravel_index(np.argmax(prob[i, :]), prob[i, :].shape)[0] + 1
    return y_predict



def accuracy(y_predict, y=y):
    m = y.size
    count = 0
    for i in range(y.shape[0]):
        if y_predict[i] == y[i]:
            j = 1
        else:
            j = 0
        count = j + count  # 计数预测值和期望值相等的项
    return count / m

prob = a[3]
y_predict = predict(prob)
accuracy(y_predict)
print ('accuracy = {0}%'.format(accuracy(y_predict) * 100))

