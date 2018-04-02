# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:47:19 2018

@author: rishabh
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

dataset=pd.read_csv('E:/RS/ML/lazy/self code lazy/Linear Regression/data_poly.csv',header=None)
X=dataset.iloc[:,0]
X=pd.concat([X,np.square(dataset.iloc[:,0])],axis=1)
Y=dataset.iloc[:,1]

plt.scatter(X.iloc[:,0],Y)

w=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))

Yhat = np.dot(X,w)

d1=Y-Yhat
d2=Y-Y.mean()
r2=1-(np.dot(d1,d1)/np.dot(d2,d2))

print("R-squared",r2)