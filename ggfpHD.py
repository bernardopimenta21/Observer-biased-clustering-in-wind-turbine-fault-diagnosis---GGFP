# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:09:13 2021

@author: Bernardo
"""
import numpy as np, numpy.random
import pandas as pd
from scipy.spatial.distance import cdist
from sammon import sammon
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from rand_index import rand_index_score
from sklearn import datasets, preprocessing
from fcm import fcm
from kmeanspp import KMeans
from load_data import load_data
from fcmfpHD import fcmfpHD
from ggfp_initfcmfp import ggfp


def calculateCenters(data,u,P,nClusters,zeta,m):
    #Calculates centers of clusters

    zetaP = np.dot(zeta,P)

    num=np.dot((u**m), data) + zetaP
    
    den=np.sum(np.power(u,m), axis=1, keepdims=True)   + zeta 
    
    C =  num/den
    #C=C[np.mean(C,axis=1).argsort()]  #organize the centers by mean of the rows, from closest to origin to furthest

    return C



def updateU(nClusters,dist,m): 
    #Update partition matrix U

    # dist = cdist(C, data, metric='sqeuclidean')   
        # #Calculate distance between data points and cluster centers


    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u

def labelData(u):
    #Find clustering results

    clustering = u.argmax(axis=0)
   
    return clustering

def obj_function(u,C,dist,dist_FP,zeta,m):

    # dist = cdist(C, data, metric='sqeuclidean') 
    # distFP=cdist(C, P, metric='sqeuclidean')

    aux=np.sum(dist_FP) 
    aux=np.nan_to_num(aux)
    
    obj_fcn =( np.sum(u**m * dist) + (zeta * aux) )  #obj function  (multiplicado por 100 porque os valores de A sao muito pequenos devido a extender a dimensao)
    
    obj_fcn=np.nan_to_num(obj_fcn)
                                        
    return obj_fcn

def project_centers(C,P):
    (C,index) = np.unique(C,axis=0,return_index=True)
    C=C+1e-10
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
 
       
def ggfpHD(data,P,nClusters,zeta,m, init,z_fcmfp,m_fcmfp,alpha, max_iter=100,min_impro=1e-6):
    # w=np.shape(P)[1]
    #     # Number of samples
    # n = len(data)  # the number of row
    # d = len(data[0]) #number of features/attributes

    # if init=='kmeans++':
     
    #     kmeans = KMeans(nClusters,init='k-means++').fit(data)
    
        
    #     C=kmeans.cluster_centers_     
      
    #     if w>d:
    #         #Extending X and Centers by adding null coordinates
    #         addZerosX = np.zeros((n, w-d))
    #         data = np.append(data, addZerosX, axis=1) 
    #         addZerosC = np.zeros((nClusters, w-d))
    #         C = np.append(C, addZerosC, axis=1)       #extend dimension of centers

    #     else: 
    #         print("Focal point should be a higher dimension than data") 
    #         sys.exit()
      
        
    #     #random inital partition matrix
    #     U = np.random.dirichlet(np.ones(nClusters),size=n).T
    #     #euclidean metric partition matrix (fcm)
    #     # dist = cdist(C, data, metric='sqeuclidean')   
    #     # aux = (1 / dist) ** (1 / (m-1))
    #     # U = (aux / aux.sum(axis=0))
    #     #print('kmeans here')

    # elif init=='fcm':
        

    #     #generate seed for random numbers
    #     rng = np.random.default_rng() #use np.random.default_rng() for random sequences


    #     C = rng.random((nClusters,len(data[0])))    #Initialize centers of clusters


    #     C, U, labels, obj_fcn = fcm(data,C,nClusters,m)   #fcm(C,nClusters,m)

        
    #     if w>d:
    #         #Extending X and Centers by adding null coordinates
    #         addZerosX = np.zeros((n, w-d))
    #         data = np.append(data, addZerosX, axis=1) 
    #         addZerosC = np.zeros((nClusters, w-d))
    #         C = np.append(C, addZerosC, axis=1)       #extend dimension of centers

    #     else: 
    #         print("Focal point should be a higher dimension than data") 
    #         sys.exit()
      
    #     # addZerosX = np.zeros((n, w-d))
    #     # data = np.append(data, addZerosX, axis=1)
    #     # #print('fcm here')
    # elif init=='fcm++':
        
    #     #Initiate fcm with kmeans++ centers
        
    #     kmeans = KMeans(nClusters,init='k-means++').fit(data)
        
            
    #     C=kmeans.cluster_centers_
        
    #     C, U, labels, obj_fcn = fcm(data, C,nClusters,m=2)   #fcm(C,nClusters,m)
    #     if w>d:
    #         #Extending X and Centers by adding null coordinates
    #         addZerosX = np.zeros((n, w-d))
    #         data_extend = np.append(data, addZerosX, axis=1) 
    #         addZerosC = np.zeros((nClusters, w-d))
    #         C = np.append(C, addZerosC, axis=1)       #extend dimension of centers

    #     else: 
    #         print("Focal point should be a higher dimension than data") 
    #         sys.exit()        
    #     #print('fcm++ here')  
    # elif init=='fcmfp' :
        
    #     C, U, label,nClusters,projCenter= fcmfpHD(data,P,nClusters,z_fcmfp,m_fcmfp)
        
    #     if w>d:
    #     #Extending X and Centers by adding null coordinates
    #         addZerosX = np.zeros((n, w-d))
    #         data_extend = np.append(data, addZerosX, axis=1) 

    # C, U, label,obj_fcn,dist,dist_FP,M,Pi,projCenter,distC= ggfp(data_extend,U,C,P,nClusters,zeta,m,alpha)   
    if init=='fcm++'  :  
        kmeans = KMeans(nClusters,init='k-means++').fit(data)  
        C=kmeans.cluster_centers_    
        C, U, labels, obj_fcn = fcm(data, C,nClusters,m)   #fcm(C,nClusters,m)
        if w>d:
                #Extending X and Centers by adding null coordinates
                addZerosX = np.zeros((n, w-d))
                data_extend = np.append(data, addZerosX, axis=1) 
                addZerosC = np.zeros((nClusters, w-d))
                C = np.append(C, addZerosC, axis=1)       #extend dimension of centers

    else:   
        C, U, label,nClusters,projCenter= fcmfpHD(data,P,nClusters,z_fcmfp,m_fcmfp)
        
    if w>d:
        #Extending X and Centers by adding null coordinates
            addZerosX = np.zeros((n, w-d))
            data_extend = np.append(data, addZerosX, axis=1) 
    

    else: 
        print("Focal point should be a higher dimension than data") 
        sys.exit() 
        

    C, U, label,obj_fcn,dist,dist_FP,M,Pi,projCenter,distC= ggfp(data_extend,U,C,P,nClusters,zeta,m,alpha)   
    
    
    
    #Remove negligle centers using the definition of typicallity
    indices = typicallity(U.T,n)
    C=C[indices] 
    
    #Get new number of clustesr
    nClusters=len(C)
    
    #Calculate new membership values
    
    U=U[indices,:] 

    
    label = labelData(U)    
    projCenter= project_centers(C,P)
    
    
    
    return  C, U, label,obj_fcn,dist,dist_FP,M,Pi,projCenter,data_extend




"#Call and run the ggfp algorithm"

if __name__ == '__main__':
    
    #Choose data and parameters
    nClusters=10        #initial number of clusters
    
    dataset='wt_test'      #data='iris','bearing'
    zeta=1
    m=4 
    alpha=1.6
                  #fuzzy parameter
    init='fcmfp'        #init='kmeans++','fcm',fcm++','random','fcmfp'
    
    #If FCMFP chosen as init
    m_fcmfp=2   #to choose number of clusters use zeta:    zeta=1 -> c=2 (expected nº of clusters- see fig from iterfcmfp)
    z_fcmfp=1            #                              zeta=0.56 -> c=3
                    #                                      zeta=0.4 -> c=4
                    #                                      zeta=0.17 -> c=6
                    #                                      zeta=0.135 -> c=7
                    #                                      zeta=0.09 -> c=8
    

    #load data
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    


    "Define Focal point"
    
    #P=5*np.array([1, 1, 1]).reshape((1, 3))      #for the simple dataset
    #P=np.array([1, 1, 1, 1, 1]).reshape((1, 5))  #for the iris dataset
    
    #focal point using mean of data
    datamean=np.array(np.mean(data,axis=0))
    Pvalue=np.array([1000])
    P=np.append(datamean,Pvalue,axis=0)
    P=np.reshape(P,(1,d+1))
    
    w=np.shape(P)[1]     #nº of columns : dimension of P


    
    #  Call GGFP : ... =GGFP(data,P,nClusters,zeta,m,init='fcm','kmeans++')
    C, U, label,obj_fcn,dist,dist_FP,M,Pi,projCenter,data_extend = ggfpHD(data,P,nClusters,zeta,m, init,z_fcmfp,m_fcmfp,alpha)  
        
    c=len(projCenter)
    if c==2:
          aux=label_gt >1
          label_gt=aux+1 -1
    else:
        label_gt=label_gt  #ground truth label

    print('GGFP Adjusted Rand Index :', adjusted_rand_score(label_gt, label))
    
