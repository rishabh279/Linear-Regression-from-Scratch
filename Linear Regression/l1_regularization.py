# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 03:10:57 2018

@author: rishabh
"""
#Illustrating l1 regulrization
import numpy as np
import matplotlib.pyplot as plt

N=50
D=50

#uniformly distributed numbers between -5 and 5 
X=(np.random.random((N,D))-0.5)*10

#true weights
true_w=np.array([1,0.5,-0.5]+[0]*(D-3))

#add noise with variance 0.5(DOUBT)
Y=X.dot(true_w)+np.random.randn(N)*0.5
Y2=X.dot(true_w)
plt.scatter(Y2)

#perform gradient descent
cost=[]
w=np.random.randn(D)/np.sqrt(D)
learning_rate=0.0001
l1=10
for i in range(500):
  Yhat=X.dot(w)
  delta=Yhat-Y
  w=w-learning_rate*(X.T.dot(delta) +11*np.sin(w))
  mse=delta.dot(delta)/N
  cost.append(mse)
  
plt.plot(cost)

plt.plot(true_w,label='actual')
plt.plot(w,label='predicted')
plt.legend()  
  
  


