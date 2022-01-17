# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:54:48 2021

@author: Bernardo
"""
import numpy as np

U=np.array([[0.2,  0.8 , 0.05],[0.3,  0.8  ,0.2],[0.9 , 0.3 , 0],[0.4 , 0.05, 0.3]])

def typicallity(U):
    #Find the clusters that have typical data. A cluster is considered to have no typical data 
    # if for all points the U values are not the maximum in any point.
    N=len(U)
    indices=np.zeros((N,1), dtype=int)
    
    for i in range(N):
        
          ind = np.argmax(U[i])
          indices[i]=ind    
         
    indices=np.unique(indices)  # list of all clusters with typical datum

    return indices

indices = typicallity(U)