import numpy as np
import matplotlib.pyplot as plt
import math

#读取数据
data = np.loadtxt('D:\python\homewrok\data_sets\ex2data1.txt', delimiter=",")
x1= data[:,[0]]
x2= data[:,[1]]
x = data[:,[0,1]]
x = np.c_[np.ones(len(data)),x,(x1**2)/100,(x1**3)/10000]                                          #添加第一列
y = data[:,[2]]

#变量初始化
n=5
m=len(data)
time=0                                                                   #迭代次数time
aerfa=0.001                                                              #学习率aerfa
theta=np.zeros(n)
JJ=np.zeros(10000)                                                      #代价函数

#数据可视化
plt.subplot(211)
for i in range(m):
    if y[i]==0:
        plt.scatter(x[i,[1]],x[i,[2]],c='red',marker='x')
    if y[i]==1:
        plt.scatter(x[i,[1]],x[i,[2]],c='blue',marker='o')

#目标函数
def h(theta,x):
    h1= x @ theta
    y=1/(1+np.exp(-h1))
    return y

#代价函数
def J(theta,x,y):
    yn = y.reshape(100)
    cost = yn * np.log(h(theta,x)) + (1-yn) * np.log(1-h(theta,x))
    out = -1/m * cost
    return np.sum(out)

#梯度下降
while time<10000:

    yt=y.reshape(100)       #转置
    daoshu = (aerfa / m) * (x.T @ (h(theta, x) - yt))       #向量运算，关注行和列配对
    theta = theta - daoshu                                  #迭代
    JJ[time]=J(theta,x,y)                                   #代价函数
    time+=1

#结果可视化
a=-theta[0]/theta[2]
b=-theta[1]/theta[2]
c=-theta[3]/theta[2]
d=-theta[4]/theta[2]
plt.plot(range(120),a+b*range(120)+c/100*(range(120))*(range(120))+d/10000*(range(120))*(range(120))*(range(120)))
plt.subplot(212)
plt.plot(range(10000),JJ)

print(theta,time,type(a))
plt.show()





