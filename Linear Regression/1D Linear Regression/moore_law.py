# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 08:43:57 2018

@author: rishabh
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset=pd.read_csv('',header=None)

X=[]
Y=[]

non_decimal=re.compile(r'[^\d]+')

for line in open('E:/RS/ML/lazy/self code lazy/Linear Regression/moore.csv'):
    r=line.split('\t')
    
    x=int(non_decimal.sub('',r[2].split('[')[0]))
    y=int(non_decimal.sub('',r[1].split('[')[0]))
    X.append(x)
    Y.append(y)

X=np.array(X) 
Y=np.array(Y)     

plt.scatter(X,Y)

Y = np.log(Y)
plt.scatter(X,Y)

denominator = X.dot(X) - X.mean()*X.sum()
a=(X.dot(Y)-Y.mean()*X.sum())/denominator
b=(Y.mean()*X.dot(X)-X.mean()*X.dot(Y))/denominator

Yhat = a*X + b

plt.scatter(X,Y) 
plt.plot(X,Yhat)

d1= Y-Yhat
d2=Y-Y.mean()
r2=1-d1.dot(d1)/d2.dot(d2)
print("the r-sqaured value ",r2)
print("time to double:",np.log(2)/a,"years")