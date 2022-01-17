# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:28:51 2021

@author: Bernardo
"""

import numpy as np, numpy.random
import pandas as pd
from scipy.spatial.distance import cdist
from sammon import sammon
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics.cluster import adjusted_rand_score
from rand_index import rand_index_score
from load_data import load_data
from kmeanspp import KMeans





def calculateCenters(data,u,nClusters,m):
    #Calculates centers of clusters

    num=np.dot((u**m), data) 
    
    den=np.sum(np.power(u,m), axis=1, keepdims=True)
    
    C =  num/den

    return C




def updateU(data,C, nClusters,m): 
    #Calculate partition matrix U
    
    dist = cdist(C, data, metric='sqeuclidean')   
    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u


def labelData(u):
    #Find clustering results
    clustering = u.argmax(axis=0)
   
    return clustering

def obj_function(data,u,C,m):

    dist = cdist(C, data, metric='sqeuclidean') 
    
             
    obj_fcn = np.sum(u**m * dist)   #obj function  
        
                                            
    return obj_fcn




    
def fcm(data,C,nClusters,m, max_iter=100,min_impro=1e-6):
    #FCM algorithm
    obj_fcn = np.zeros((max_iter, 1))
    it = 0

    for it in range(max_iter):
        U = updateU(data,C, nClusters,m)
        C = calculateCenters(data,U,nClusters,m)
  
        obj_fcn[it]=obj_function(data,U,C,m)
             
            
        if it > 1:
            
            if abs(obj_fcn[it] - obj_fcn[it-1]) < min_impro:  #If objective function does not have any more improvement stop running algorithm
                
                break
            
                           
        it += 1

    label = labelData(U)
    return  C, U, label,obj_fcn




"#Call and run the fcm algorithm"

if __name__ == '__main__':
    
    
    
    # Define Parameters
    nClusters=3  
    dataset='iris'   #'iris','bearing'.'wt_test'
    m=2             #fuzzy parameter
    init='kmeans++'  #kmeans++ or random centers init
    projection=True    #1:Represent sammon projection
    
    
    
    
    #load data
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes


    if init=='kmeans++':
        
        kmeans = KMeans(nClusters,init='k-means++').fit(data)
        
            
        C=kmeans.cluster_centers_
    else: 
            #generate seed for random numbers
        rng = np.random.default_rng() #use np.random.default_rng() for random  
        #C=np.zeros(nClusters,d)          #Initialize centers of clusters
        C = rng.random((nClusters,d))   
        
    
    #call fcm
    C, U, labels, obj_fcn = fcm(data,C,nClusters,m)   #FCM(C,nClusters,m)

    c=len(C)
    if c==2:
          aux=label_gt >=1
          label_gt=aux+1 -1
    else:
        label_gt=label_gt  #ground truth label
    #adjusted_rand_scoreFCM=adjusted_rand_score(label_gt, labels)
    print('FCM Rand Index :',rand_index_score(label_gt, labels) )

    print('FCM Adjusted Rand Index :',adjusted_rand_score(label_gt, labels) )

    if projection==True:
        
        "Sammon Projection"
        # Run the Sammon projection
        
        sammon_data=np.concatenate((data, C), axis=0)
        #Project data points
        
        (x,index) = np.unique(sammon_data,axis=0,return_index=True)  #the data can't have duplicated rows or sammon will fail
        labels2=np.concatenate((labels, 50*np.ones((nClusters))), axis=0)
        target = labels2[index] 
        [y,E] = sammon(x,2)
        
        
        
        # Plot
        arr_str_color = ['g','b','c','m','y','k','tab:pink','orange','gray','brown'] 
        arr_str_marker= ['o', 'D', 'v', '+', ',', '^', '<', '>', 's', 'd']
        
        for i in range(nClusters):
            #Plot data points
            plt.scatter(y[target ==i, 0], y[target ==i, 1], s=20, c=arr_str_color[i], marker=arr_str_marker[i],label='Cluster %d' % (i+1))
        #Plot centers
        plt.scatter(y[target ==50, 0], y[target ==50, 1], s=50,c= 'r',marker='x')
        
        
        plt.title('Sammon projection of the data - FCM')
        #plt.legend(loc='lower left')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        print("CLustering complete")

    else:
        print("CLustering complete")
        
        