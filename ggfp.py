# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:29:14 2021

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
import math
from load_data import load_data
from kmeanspp import KMeans
from fcmfpHD import fcmfpHD



def calculateCenters(data,u,P,nClusters,zeta,m):
    #Calculates centers of clusters

    zetaP = np.dot(zeta,P)

    num=np.dot((u**m), data) + zetaP
    
    den=np.sum(np.power(u,m), axis=1, keepdims=True)   + zeta 
    
    C =  num/den
    #C=C[np.mean(C,axis=1).argsort()]  #organize the centers by mean of the rows, from closest to origin to furthest

    return C


def typicallity(U,n):
    #Find the clusters that have typical data. A cluster is considered to have no typical data 
    # if for all points the U values are not the maximum in any point.
    
    indices=np.zeros((n,1), dtype=int)
    
    for i in range(n):
        
          ind = np.argmax(U[i])
          indices[i]=ind    
         
    indices=np.unique(indices)  # list of all clusters with typical datum

    return indices

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
    
    obj_fcn =( np.sum(u**m * dist) + (zeta * aux) )  
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



 
       
def ggfp(data,U,C,P,nClusters,zeta,m,alpha, max_iter=100,min_impro=1e-4):
    w=np.shape(P)[1]
        # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes


    
   #GGFP algorithm
    # obj_fcn = np.zeros((max_iter, 1))
    # Pi=np.zeros((nClusters,1))
    # dist=np.zeros((n,nClusters))
    # dist_FP=np.zeros((1,nClusters))
    it = 0
    obj_fcn = np.zeros((max_iter, 1))
    Pi=np.zeros((nClusters,1))
    dist=np.zeros((n,nClusters))
    dist_FP=np.zeros((1,nClusters))
    distC=np.zeros((nClusters,nClusters))

    for it in range(max_iter):

        M=[]
        C = calculateCenters(data,U,P,nClusters,zeta,m)
        for i in range(nClusters):
            sumfm=np.sum(np.power(U,m), axis=1, keepdims=True)

            A=np.zeros((w,w))
            xv=data-C[i,:]
            cc=C-C[i,:]

            #funciona para o iris com m=3 zeta=0.08
            # Pv=(P-C)
            
            # #Covariance matrix calculation
            # a=np.dot(np.power(U[i,:],m) * np.transpose(xv), xv)
            # b=zeta*np.dot(Pv.T,Pv)


            Pv=(P-C[i,:])
            
            #Covariance matrix calculation
            a=np.dot(np.power(U[i,:],m) * np.transpose(xv), xv)
            b=zeta*np.dot(Pv.T,Pv)

            A=np.divide((a+b ), (sumfm[i,:] + zeta))  
            # A=A+np.identity(w)*0.5         #Covariance matrix regularization
            A=A+np.identity(w)*alpha
            M.append(A)
            
            #Priori probability
            Pi[i,:]=1/n*sumfm[i,:]
            # Pi[i,:] = np.sum(U[i,:]**m)/ np.sum(U**m)
            Pi=Pi + 1e-10

            
            
            #Calculate distance between data points and cluster centers
           
            dist_aux=((np.linalg.det(A)**0.5) * 1/Pi[i,:] 
              * np.exp(0.5*np.sum((np.dot(xv,np.linalg.inv(A))*xv), axis=1,keepdims=True)))  
            dist_aux=dist_aux+1e-10
            dist[:,i]=dist_aux[:,0]
            dist=np.nan_to_num(dist)


            #Calculate distance between Focal Point and cluster centers       

            
            dist_FP[0,i] = (((np.linalg.det(A)**0.5)/ Pi[i,:]) * np.exp((1/2) * 
                                  np.dot( np.dot( Pv, np.linalg.pinv(A)),Pv.T ))) 
            

            #Calculate distance between cluster centers       
            dist_auxC=(1/(np.linalg.det(np.linalg.pinv(A))**0.5) * 1/Pi[i,:] 
              * np.exp(0.5*np.sum((np.dot(cc,np.linalg.pinv(A))*cc), axis=1,keepdims=True)))            
            dist_auxC=dist_auxC+1e-10
            distC[:,i]=dist_auxC[:,0]
            distC=np.nan_to_num(distC)
            

        distC=distC.T * distC      #get symmetric distance matrix
        np.fill_diagonal(distC,0)
            
        U = updateU(nClusters,dist.T,m)
        
        obj_fcn[it]=obj_function(U,C,dist.T,dist_FP.T,zeta,m)
             
            
        if it > 1:
            
            if abs(obj_fcn[it] - obj_fcn[it-1]) < min_impro:  #If objective function does not show improvement stop running algorithm
                
                break
            
                           
        it += 1

    label = labelData(U)    
    projCenter= project_centers(C,P)
    
    return  C, U, label,obj_fcn,dist,dist_FP,M,Pi,projCenter,distC




"#Call and run the ggfp algorithm"

if __name__ == '__main__':
    
    #Choose data and parameters
    nClusters=3
    dataset='iris'      #data='iris','bearing'.'bearing_unprocessed'
    m=2                    #fuzzy parameterv                      c=4 -> m=4,z=0.1
    zeta=0.1
    alpha=0.01
    init='fcmfp'
        
    #load data
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    

    "Define GGFP Focal point"
    
    #P=5*np.array([1, 1, 1]).reshape((1, 3))      #for the simple dataset
    #P=np.array([1, 1, 1, 1, 1]).reshape((1, 5))  #for the iris dataset
    
    #focal point using mean of data
    datamean=np.array(np.mean(data,axis=0))
    Pvalue=np.array([10])
    P=np.append(datamean,Pvalue,axis=0)
    P=np.reshape(P,(1,d+1))
    
    w=np.shape(P)[1]     #nº of columns : dimension of P
        
    if w>d:
        #Extending X and Centers by adding null coordinates
        addZerosX = np.zeros((n, w-d))
        data_extend = np.append(data, addZerosX, axis=1) 




    "Params if FCMFP is init"
    z_fcmfp=0.6
    m_fcmfp=2
    
  
    
    
    

    
    if init=='kmeans++':
         
        kmeans = KMeans(nClusters,init='k-means++', random_state=0).fit(data)
        
            
        C=kmeans.cluster_centers_
        U = np.random.dirichlet(np.ones(nClusters),size=n).T
        # dist = cdist(C, data, metric='sqeuclidean')   
        # aux = (1 / dist) ** (1 / (m-1))
        # U = (aux / aux.sum(axis=0))
        #print('kmeans here')

    elif init=='fcm':
        #generate seed for random numbers
        rng = np.random.default_rng() #use np.random.default_rng() for random sequences


        C = rng.random((nClusters,d))    #Initialize centers of clusters

        
        #C = np.random.rand(nClusters,d)    #Initialize centers of clusters
        C, U, labels, obj_fcn = fcm(data, C,nClusters,m)   #fcm(C,nClusters,m)
        #print('fcm here')
    elif init=='fcm++':
        
        #Initiate fcm with kmeans++ centers
        
        kmeans = KMeans(nClusters,init='k-means++').fit(data)
        
            
        C=kmeans.cluster_centers_
        
        C, U, labels, obj_fcn = fcm(data, C,nClusters,m)   #fcm(C,nClusters,m)
        #print('fcm++ here')   
    elif init=='random':
    
        #generate seed for random numbers
        rng = np.random.default_rng() #use np.random.default_rng() for random sequences or np.random.default_rng(seed=42) for reproducible results


        C = rng.random((nClusters,d))    #Initialize centers of clusters

        
        #C = np.random.rand(nClusters,d)    #Initialize centers of clusters
        
        U = rng.dirichlet(np.ones(nClusters),size=n).T

    elif init=='fcmfp':
        
        nClusters=10
        #Initialize with fcmfpHD
        C, U, label,nClusters,projCenter= fcmfpHD(data,P,nClusters,z_fcmfp,m_fcmfp)
        
    







    # # #Calculate new membership values
    C, U, label,obj_fcn,dist,dist_FP,M,Pi,projCenter,distC= ggfp(data_extend,U,C,P,nClusters,zeta,m,alpha)   

    
    c=len(projCenter)
    if c==2:
          aux=label_gt >1
          label_gt=aux+1 -1
    else:
        label_gt=label_gt  #ground truth label

    print('GGFP Adjusted Rand Index :', adjusted_rand_score(label_gt, label))
