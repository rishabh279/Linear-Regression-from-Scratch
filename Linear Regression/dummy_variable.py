# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 00:40:51 2018

@author: rishabh
"""

import numpy as np
a=np.zeros(shape=(6,4))
a[0]=[1,0,0,1]
a[1]=[1,0,0,1]
a[2]=[0,1,0,1]
a[3]=[0,1,0,1]
a[4]=[0,0,1,1]
a[5]=[0,0,1,1]
b=a.T
c=np.dot(b,a)
np.linalg.inv(c)