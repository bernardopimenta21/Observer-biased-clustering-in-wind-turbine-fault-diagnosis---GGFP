# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:17:23 2021

@author: Bernardo
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from rand_index import rand_index_score

from fcmfp import fcmfp
import sys
from scipy.spatial.distance import cdist
from sklearn import datasets
from load_data import load_data
import sys 
from sklearn.metrics.cluster import adjusted_rand_score
from sammon import sammon

def project_centers(C,P):
    C_col=len(C[0])  #nr of columns of center matrix = nº of features in data
    C_row=len(C)        #nr of rows = nr of clusters
    P_dim=len(P[0])    #nr of features 
    projectedCenter=np.zeros((C_row,C_col-1))
    
    for i in range(C_row):
        vDirector = C[i, :] - P
        #C[i, C_col - 1] = 0
        t =  - P[:,P_dim-1] / vDirector[:,len(vDirector[0])-1]
        for j in range(C_col-1):
            projectedCenter[i, j] = P[0,j] + np.multiply(t , vDirector[0,j])
            

    return projectedCenter

def typicallity(U,n):
    #Find the clusters that have typical data. A cluster is considered to have no typical data 
    # if for all points the U values are not the maximum in any point.
    
    indices=np.zeros((n,1), dtype=int)
    
    for i in range(n):
        
          ind = np.argmax(U[i])
          indices[i]=ind    
         
    indices=np.unique(indices)  # list of all clusters with typical datum

    return indices
def labelData(u):
    #Find clustering results

    clustering = u.argmax(axis=0)
   
    return clustering
def updateU(data,C, nClusters,m): 
    #Update partition matrix U

    dist = cdist(C, data, metric='sqeuclidean')   
    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u

def fcmfpHD(data,P,nClusters,zeta,m):
    w=np.shape(P)[1]     #nº of columns : dimension of P
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
        #generate seed for random numbers
    rng = np.random.default_rng() #use np.random.default_rng() for random  
    C = rng.random((nClusters,w))    #Initialize centers of clusters
    
    C, U, labels, obj_fcn, projCenter,data_extend= fcmfp(data,P, C,nClusters,zeta,m)   #FCMFP(C,nClusters,zeta,m)
    
    #Remove negligle centers using the definition of typicallity
    indices = typicallity(U.T,n)
    C=C[indices] 
    
    #Get new number of clustesr
    nClusters=len(C)
    
    #Calculate new membership values
    
    U=updateU(data_extend, C,nClusters,m)
    dist = cdist(C, data_extend, metric='sqeuclidean')   
    aux = (1 / dist) ** (1 / (m-1))
    U = (aux / aux.sum(axis=0))
    
    

    label = labelData(U)    
    projCenter= project_centers(C,P)


    return C, U, label,nClusters,projCenter


if __name__ == '__main__':

    dataset='wt_test'
    m=2
    zeta=1.4 #                                      zeta=0.56 -> c=3
                    #                                      zeta=0.4 -> c=4
                    #                                      zeta=0.17 -> c=6
                            #                                zeta=0.136 -> c=7
                    #                                      zeta=0.09 -> c=8
                # WT dataset:  c=2->  m=2, zeta=2                  
                #              c=3->  m=2, zeta=1.1  
                #              c=4->  m=2, zeta=0.6
                #              c=6->  m=2, zeta=0.35   
    nClusters=10    #Initial cluster number, dont change
    data, label_gt =load_data(dataset)
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    
    "Define Focal point"    
    #P=5*np.array([1, 1, 1]).reshape((1, 3))      #for the simple dataset
    #P=np.array([1, 1, 1, 1, 1]).reshape((1, 5))  #for the iris dataset
    datamin=-10000*np.array(np.min(data,axis=0))
    
    #focal point using mean of data
    datamean=np.array(np.mean(data,axis=0))
    Pvalue=np.array([1000])                        #Change value on higher dim
    P=np.append(datamean,Pvalue,axis=0)
    P=np.reshape(P,(1,d+1))    
    w=np.shape(P)[1]     #nº of columns : dimension of P
   

    
    
    C, U, label,nClusters,projCenter= fcmfpHD(data,P,nClusters,zeta,m)   #FCMFP(C,nClusters,zeta,m)
    
    # projCenter= project_centers(C,P)
    
    
    c=len(projCenter)
    if c==2:
          aux=label_gt >1
          label_gt=aux+1 - 1
    else:
        label_gt=label_gt  #ground truth label
    print('FCMFP Rand Index :', rand_index_score(label_gt, label))

    print('FCMFP Adjusted Rand Index :', adjusted_rand_score(label_gt, label))
    
    # "Sammon Projection"
    # # Run the Sammon projection
    # original_data=np.delete(data,-1,axis=1)
    
    # sammon_data=np.concatenate((data, projCenter), axis=0)
    # #Project data points
    
    # (x,index) = np.unique(sammon_data,axis=0,return_index=True)  #the data can't have duplicated rows or sammon will fail
    # labels2=np.concatenate((labels, 50*np.ones((nClusters))), axis=0)
    # target = labels2[index] 
    # [y,E] = sammon(x,2)
    
    
    
    # # Plot
    # arr_str_color = ['g','b','c','orange','y','k','tab:pink','orange','gray','brown'] 
    # arr_str_marker= ['o', 'D', 'v', '+', ',', '^', '<', '>', 's', 'd']
    
    # for i in range(nClusters):
    #     #Plot data points
    #     plt.scatter(y[target ==i, 0], y[target ==i, 1], s=20, c=arr_str_color[i], marker=arr_str_marker[i],label='Cluster %d' % (i+1))
    # #Plot centers
    # plt.scatter(y[target ==50, 0], y[target ==50, 1], s=50,c= 'r',marker='x')
    
    
    # plt.title('FCMFP - Sammon projection of the data ')
    # # plt.legend(loc='lower left')
    # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    # plt.show()
