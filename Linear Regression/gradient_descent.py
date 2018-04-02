# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 01:09:33 2018

@author: rishabh
"""

#showing gradient descent can solve the problem of dummy variable
import numpy as np
import matplotlib.pyplot as plt

N=10
D=3

X=np.zeros((N,D))

X[:,0]=1
X[:5,1]=1
X[5:,2]=1

Y=np.array([0]*5+[1]*5).reshape(-1,1)

learning_rate=0.001

cost=[]
w=(np.random.rand(D)/ np.sqrt(D)).reshape(-1,1)
for i in range(1000):
  Yhat=X.dot(w)
  delta=Yhat-Y
  w=w-learning_rate*X.T.dot(delta)
  
  #Mean Absolute Error
  mse=delta.T.dot(delta)/N
  cost.append(mse)

#WRONG PRACTICE TO BE CORRECT  
costs=[]  
for i in range(1000):
  for j in range(1):
    for k in range(1):
      costs.append(cost[i][j][k])
      
plt.plot(costs)

plt.plot(Yhat,label='prediction')
plt.plot(Y,label='target')
plt.legend()
  