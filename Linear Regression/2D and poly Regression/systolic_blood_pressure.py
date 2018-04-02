# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:36:53 2018

@author: rishabh
"""

#X1=systolic blood pressure
#X2=age in years
#X3=weight in pounds

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('E:/RS/ML/lazy/self code lazy/Linear Regression/bloodpressure.csv')
dataset['one']=1
X=dataset.loc[:,['X2','X3','one']]
Y=dataset.loc[:,['X1']]
X2only=dataset.loc[:,['X2','one']]
X3only=dataset.loc[:,['X3','one']]

#plot against X2
plt.scatter(X.iloc[:,X.columns.get_loc('X2')],Y)

#plot against X3
plt.scatter(X.iloc[:,X.columns.get_loc('X3')],Y)

#dataset['ones']=1

def get_r2(X,Y): 
  w=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y)) 
  Yhat = np.dot(X,w) 
  d1=Y-Yhat
  #d1=d1.T.squeeze()
  d2=Y-Y.mean()
  #d2=d2.T.squeeze()
  r2=1-(np.dot(d1,d1)/np.dot(d2,d2))
  return r2

print("X2 only")  
get_r2(X2only,Y)
print("X3 only")
get_r2(X3only,Y)  
print("Both")
get_r2(X,Y)  

