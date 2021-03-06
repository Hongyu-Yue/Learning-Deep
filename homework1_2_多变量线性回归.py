import numpy as np
import matplotlib.pyplot as plt

#导入数据
data = np.loadtxt('D:\python\homewrok\data_sets\ex1data2.txt', delimiter=",")

x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]

#变量初始化
m = len(data)
n = 3
thetaT = np.zeros((n,1))
aerfa = 0.01
time=0
beilv=max(x1)-min(x1)
JJ=[]
x0=np.ones((m,1))
x1=x1/  beilv
x1=x1.reshape(m,1)
x2=x2.reshape(m,1)
x = np.hstack((x0,x1,x2))
JJcost=0
#原函数h
def h(thetaT,x):
    theta = thetaT.reshape(1, n)                        #参数向量转置
    y = theta @ x
    return y

#代价函数
def J(thetaT,x,y):
    theta = thetaT.reshape(1, n)
    J=(h(thetaT,x)-y)**2*(1/2/m)
#主循环
while time<5000:


    JJcost=0
    costT=np.zeros((n,1))                                            #中间向量归零

    for i in range(m):
        xT=x[i].reshape(n,1)
        costT+=(h(thetaT,xT)-y[i])*xT
        JJcost+=(h(thetaT,xT)-y[i])**2
    JJ.append(JJcost[0,0]/2/m)
    thetaT = thetaT - aerfa * costT / m
    time+=1


plt.subplot(211)
plt.plot(range(0,5000),JJ)
plt.show()

print(thetaT,time,beilv)



