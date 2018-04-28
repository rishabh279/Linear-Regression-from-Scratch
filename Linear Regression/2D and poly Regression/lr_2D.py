# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:27:13 2018

@author: rishabh
"""

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

dataset=pd.read_csv('',header=None)

X=dataset.iloc[:,:2]
Y=dataset.iloc[:,2]

#plot the data to see what data looks like
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X.iloc[:,0],X.iloc[:,1],Y)
plt.show()  

#Calculating the weights
w=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
Yhat=np.dot(X,w)

#Since plotting is difficult in 3d we will andlyze through calcukating r2
#compute r squared
d1=Y-Yhat
d2=Y-Yhat.mean()

r2=1-(d1.dot(d1)/d2.dot(d2))

print("r-squared",r2)

