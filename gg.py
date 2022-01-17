# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:00:35 2021

@author: Bernardo
"""

import numpy as np, numpy.random
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



#Calculates centers of clusters
def calculateCenters(data,u,nClusters,m):
    

    num=np.dot((u**m), data) 
    
    den=np.sum(np.power(u,m), axis=1, keepdims=True)
    
    C =  num/den
    

   #C=C[np.mean(C,axis=1).argsort()]  #organize the centers by mean of the rows, from closest to origin to furthest
      
    return C

#Update partition matrix U
def updateU(nClusters,dist,m): 
    
    #dist = cdist(C, data, metric='sqeuclidean')   
    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u

def obj_function(u,C,dist,m):

    obj_fcn = np.sum(u**m * dist)   #obj function  
        
                                            
    return obj_fcn
def labelData(u):
    #Find clustering results

    clustering = u.argmax(axis=0)
   
    return clustering





 
       
def gg(data,U,C,nClusters,m,alpha_gg, max_iter=100,min_impro=1e-4):
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
        C = calculateCenters(data,U,nClusters,m)
        for i in range(nClusters):
            sumfm=np.sum(np.power(U,m), axis=1, keepdims=True)

            A=np.zeros((d,d))
            xv=data-C[i,:]
            cc=C-C[i,:]

            #funciona para o iris com m=3 zeta=0.08
            # Pv=(P-C)
            
            # #Covariance matrix calculation
            # a=np.dot(np.power(U[i,:],m) * np.transpose(xv), xv)
            # b=zeta*np.dot(Pv.T,Pv)


            
            #Covariance matrix calculation
            a=np.dot(np.power(U[i,:],m) * np.transpose(xv), xv)

            A=np.divide((a ), (sumfm[i,:] ))  
            # A=A+np.identity(w)*0.5         #Covariance matrix regularization
            A=A+np.identity(d)*alpha_gg
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



            #Calculate distance between cluster centers       
            dist_auxC=(1/(np.linalg.det(np.linalg.pinv(A))**0.5) * 1/Pi[i,:] 
              * np.exp(0.5*np.sum((np.dot(cc,np.linalg.pinv(A))*cc), axis=1,keepdims=True)))            
            dist_auxC=dist_auxC+1e-10
            distC[:,i]=dist_auxC[:,0]
            distC=np.nan_to_num(distC)
            

        distC=distC.T * distC      #get symmetric distance matrix
        np.fill_diagonal(distC,0)
            
        U = updateU(nClusters,dist.T,m)
        
        obj_fcn[it]=obj_function(U,C,dist.T,m)
             
            
        if it > 1:
            
            if abs(obj_fcn[it] - obj_fcn[it-1]) < min_impro:  #If objective function does not show improvement stop running algorithm
                
                break
            
                           
        it += 1

    label = labelData(U)    
    
    return  C, U, label,obj_fcn,dist,M,Pi




"#Call and run the ggfp algorithm"

if __name__ == '__main__':
    
    "Define parameters"
    nClusters=3
    dataset='iris'      #data='iris','bearing'.'wt_test'
    init='fcmfp'
    m=2 #fuzzy parameter
    alpha_gg=0
    
    
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    
    
    "Parameters if  FCMFP is init"
    z_fcmfp=0.1
    m_fcmfp=2
    
       
    #P=5*np.array([1, 1, 1]).reshape((1, 3))      #for the simple dataset
    #P=np.array([1, 1, 1, 1, 1]).reshape((1, 5))  #for the iris dataset
    
    #focal point using mean of data
    datamean=np.array(np.mean(data,axis=0))
    Pvalue=np.array([10])
    P=np.append(datamean,Pvalue,axis=0)
    P=np.reshape(P,(1,d+1))

    
    
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
            C, U, label,nClusters,projCenter= fcmfpHD(data,P,nClusters, z_fcmfp,m_fcmfp)

    
    # # #Calculate new membership values
    C, U, label,obj_fcn,dist,M,Pi= gg(data,U,C,nClusters,m,alpha_gg)

    
    c=len(C)
    if c==2:
          aux=label_gt >1
          label_gt=aux+1 -1
    else:
        label_gt=label_gt  #ground truth label

    print('GG Adjusted Rand Index :', adjusted_rand_score(label_gt, label))
