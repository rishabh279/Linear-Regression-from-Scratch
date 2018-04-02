# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 22:49:50 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt

N=50

#generate the data
X=np.linspace(0,10,N).reshape(-1,1)
Y=0.5*X+np.random.rand(N).reshape(-1,1)

print
#generate outliers
Y[-1]=Y[-1]+30
Y[-2]=Y[-2]+30

#plot the data
plt.scatter(X,Y)

#add bias term
X=np.concatenate((np.ones((N,1)),X),axis=1)

w=np.linalg.solve(X.T.dot(X),X.T.dot(Y))
Yhat=X.dot(w)
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Yhat)

#regularization
l2=1000
w_reg=np.linalg.solve(l2*np.eye(2)+X.T.dot(X),X.T.dot(Y))
Yhat_reg=X.dot(w_reg)
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Yhat,label='maximum likelihood')
plt.plot(X[:,1],Yhat_reg,label='regularization')
plt.legend()


