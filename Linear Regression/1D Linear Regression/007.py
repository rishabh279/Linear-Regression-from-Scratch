# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 22:15:10 2018

@author: rishabh
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('',header=None)
X=dataset.iloc[:,0]
Y=dataset.iloc[:,1]

X.sum()

plt.scatter(X,Y)

denominator=X.dot(X)-X.mean()*X.sum()

a=(X.dot(Y)-X.sum()*Y.mean())/denominator

b=(Y.mean()*X.dot(X)-X.mean()*X.dot(Y))/denominator

Yhat=a*X+b

plt.scatter(X,Y)
plt.plot(X,Yhat)

#Determine the r-squared
d1=Y-Yhat
d2=Y-Y.mean()
r2=1-d1.dot(d1)/d2.dot(d2)
print("rsqaured value is",r2)
