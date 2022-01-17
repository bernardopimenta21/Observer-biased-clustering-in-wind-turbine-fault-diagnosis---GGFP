# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 19:13:36 2021

@author: Bernardo
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from fcmfp import fcmfp
import sys
from scipy.spatial.distance import cdist
from sklearn import datasets
import sys 
from load_data import load_data
# def load_data(dataset):
    
#     if dataset=='iris':
#         "Iris data"
#         iris = datasets.load_iris()
#         data = iris.data 
#         label_iris = iris.target
#         label_gt=label_iris
        
#     elif dataset=='bearing':
                
#         "Bearing data"
#         data = pd.read_csv("bearingdata.csv",header=None)        
#         data=data.to_numpy()
        
#         label_bearing = pd.read_csv("bearingdata_label.csv", header=None)
#         label_bearing=label_bearing.to_numpy().T
#         label_bearing=label_bearing.reshape(len(label_bearing))
        
#         label_gt=label_bearing #ground truth label
        
#     elif dataset=='hepta': 
#         dataset = "fcps/hepta" # e.g., "wut/smile" (UNIX-like) or r"wut\smile" (Windows)
#         data    = np.loadtxt(dataset+".data.gz", ndmin=2)
#         label_gt  = np.loadtxt(dataset+".labels0.gz", dtype=np.intc)
#     return data , label_gt


#Import Iris dataset

# df_full = pd.read_csv("iris.csv")
# columns = list(df_full.columns)
# features = columns[0:len(columns) - 1]
# # data = df_full[features]

# "Bearing data"
# data = pd.read_csv("bearingdata.csv",header=None)

# data=data.to_numpy()

# label_bearing = pd.read_csv("bearingdata_label.csv", header=None)
# label_bearing=label_bearing.to_numpy().T
# label_bearing=label_bearing.reshape(len(label_bearing))

# # Simple dataset:
# # data = pd.DataFrame([
# #         [1,2], 
# #         [2,3], 
#         [9,4], 
#         [10,1],])

#data=data.to_numpy()



# # Number of samples
# n = len(data)  # the number of row
# d = len(data[0]) #number of features/attributes
# #d = len(data.columns)

    
# #Define Focal point

# #P=5*np.array([1, 1, 1]).reshape((1, 3))      #for the simple dataset
# #P=np.array([5, 5, 5, 5, 10]).reshape((1, 5))  #for the iris dataset

# #focal point using mean of data
# datamean=np.array(np.mean(data,axis=0))
# Pvalue=np.array([10])
# P=np.append(datamean,Pvalue,axis=0)
# P=np.reshape(P,(1,d+1))




# #Extending X and Centers by adding null coordinates
# w=np.shape(P)[1]     #nº of columns : dimension of P

# if w>d:
#     addZerosX = np.zeros((n, w-d))
#     data = np.append(data, addZerosX, axis=1)
    
    
# else:
#     print("Focal point should be a higher dimension than data") 
#     sys.exit()
     
    

   
 
def xie_beni_inv(data, U, C, nClusters,m):
    #sum_cluster_distance = 0
    minimum =math.inf
    dist = cdist(C, data, metric='sqeuclidean')    
    num= np.sum(np.sum(np.multiply(U**m,dist)))    
    distC=cdist(C, C, metric='sqeuclidean')    
    aux= np.min(distC[np.nonzero(distC)])
    
    if minimum>aux:
        minimum=aux
        
    XB_inv= n*minimum  / num 
    
    return XB_inv

def relativeDegree(U,nClusters):

    h=-np.sum(U * np.log(U), axis=0,keepdims=True)  
    rd=0

    for k in range(n):
        for j in range(nClusters-1):
            i=j+1
            
            for i in range(nClusters):            
                rd=(nClusters* np.minimum(U[i,k],U[j,k]) * h[:,k]) + rd
    
        

    # rd=rd/(math.comb(n,2)*nClusters)
    
    # rd=rd*n    
    if nClusters<2:
        rd=0
    else:
        rd=(2/(nClusters*(nClusters-1))) * rd
    #rd= (nClusters*(nClusters-1)/2) #* 1/rd
    rd=1/rd #using the inverse, maximize is the objective
    return rd
def updateU(data,C, nClusters,m): 
    #Update partition matrix U

    dist = cdist(C, data, metric='sqeuclidean')   
    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u

def typicallity(U):
    #Find the clusters that have typical data. A cluster is considered to have no typical data 
    # if for all points the U values are not the maximum in any point.
    
    indices=np.zeros((n,1), dtype=int)
    
    for i in range(n):
        
          ind = np.argmax(U[i])
          indices[i]=ind    
         
    indices=np.unique(indices)  # list of all clusters with typical datum

    return indices



def calculateCenters(data,u,P,nClusters,zeta,m):
    #Calculates centers of clusters

    zetaP = np.dot(zeta,P)

    num=np.dot((u**m), data) + zetaP
    
    den=np.sum(np.power(u,m), axis=1, keepdims=True)   + zeta 
    
    C =  num/den
    #C=C[np.mean(C,axis=1).argsort()]  #organize the centers by mean of the rows, from closest to origin to furthest

    return C


def iterfcmfp(data, P, nClusters , zeta , zetashift, m, int_validity):
    #Some initialiations for needed variables
    zetaVar=np.arange(0, zeta, zetashift)   #range of the iteration 0:zetashift:zeta
    XB_inv= np.zeros((len(zetaVar), 1))    # initialize XB
    ClusterNum=np.zeros((len(zetaVar), 1)) #auxiliar function to keep track of cluster number
    numZeta=0
    rds= np.zeros((len(zetaVar), 1))    # initialize XB
    
    
    #C = np.zeros((nClusters,w))    #Initialize centers of clusters
    
    #generate seed for random numbers
    rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random
    
    #C = np.random.rand(nClusters,w)    #Initialize centers of clusters
    C = rng.random((nClusters,w))    #Initialize centers of clusters
    
    for zetaIter in zetaVar:
        #Run the fcmfp algorithm
        C, U, labels, obj_fcn, projCenter,data_extend= fcmfp(data,P, C,nClusters,zetaIter,m)   #FCMFP(C,nClusters,zeta,m)
        
        #Remove negligle centers using the definition of typicallity
        indices = typicallity(U.T)
        C=C[indices] 
        
        #Get new number of clustesr
        nClusters=len(C)
        
        #Calculate new membership values

        U=updateU(data_extend, C,nClusters,m)
        # U=U[indices,:] 

        #Calculate internal validity
        if int_validity=='XB':
            
            XB_inv[numZeta]=xie_beni_inv(data_extend, U, C, nClusters,m)
            
        elif int_validity=='KL':
            rds[numZeta]= relativeDegree(U,nClusters)
    
        else:
            print("Select a valid Internal Validity Index")
            sys.exit()
        
        #keeping track of cluster number
        ClusterNum[numZeta] = nClusters 
        numZeta += 1
        print("Zeta:",zetaIter)    
    
    return zetaVar, XB_inv, rds,ClusterNum

"Run iterative algorithm"



if __name__ == '__main__':
    
    #Choose parameters
    dataset='iris'      #dataset= 'iris,'bearing','hepta'
    int_validity='XB'   #int_validity='XB','KL'
    
    nClusters=10        # Initial number of clusters
    zetashift=0.01     #zeta increment
    zeta=2.2            #zeta final
    m=2              #fuzzy parameter

    
    #loads dataset
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes 
    
    "Define Focal point"  
    #P=5*np.array([1, 1, 1]).reshape((1, 3))      #for the simple dataset
    #P=np.array([1, 1, 1, 1, 1]).reshape((1, 5))  #for the iris dataset    
    #focal point using mean of data
    datamean=np.array(np.mean(data,axis=0))
    Pvalue=np.array([10])
    P=np.append(datamean,Pvalue,axis=0)
    P=np.reshape(P,(1,d+1))    
    w=np.shape(P)[1]     #nº of columns : dimension of P

    #Run iterative algorithm
    zetaVar, XB_inv, rds ,ClusterNum =iterfcmfp(data, P, nClusters , zeta , zetashift, m, int_validity)
    
    "Plots"
    fig, (ax1,ax2) = plt.subplots(2,sharex=True)

    if int_validity=='XB' and dataset=='iris':
        
                
        ax1.plot(zetaVar,XB_inv)
        ax1.set_title("FCMFP - Xie Beni inverted - IRIS dataset")
    
    elif int_validity=='XB' and dataset=='bearing':
        
                
        ax1.plot(zetaVar,XB_inv)
        ax1.set_title("FCMFP - Xie Beni inverted - Bearing dataset")
    
                
    elif int_validity=='KL' and dataset=='iris':
        ax1.plot(zetaVar,rds)
        ax1.set_title("FCMFP - KL index - IRIS dataset")
        
    elif int_validity=='KL' and dataset=='bearing':
        ax1.plot(zetaVar,rds)
        ax1.set_title("FCMFP - KL index - Bearing dataset") 
        
    elif int_validity=='KL' and dataset=='hepta':
        ax1.plot(zetaVar,rds)
        ax1.set_title("FCMFP - KL index - hepta dataset")    
    elif int_validity=='XB' and dataset=='hepta':
        
        ax1.plot(zetaVar,XB_inv)
        ax1.set_title("FCMFP - XB index - hepta dataset")   
        
    elif int_validity=='XB' and dataset=='wt_test':
        
        ax1.plot(zetaVar,XB_inv)
        ax1.set_title("GGFP - WT-test dataset")   
        ax1.set_ylabel("XB inverted")
    elif int_validity=='KL' and dataset=='wt_test':
        
        ax1.plot(zetaVar,rds)
        ax1.set_title("GGFP - WT-test dataset")   
        ax1.set_ylabel(" KL Index")
    else:
        print("Select a valid Internal Validity Index")
        sys.exit()
        

    
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    ax1.grid(which='minor',color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    #ax1.set_ylabel("KIM Index")
    plt.xlabel('Zeta')
    # Add major gridlines in the y-axis
    
    ax2.plot(zetaVar,ClusterNum, 'r')
    #ax2.set_title("Nº of Clusters")
    ax2.set_ylabel("Cluster Number")
    # Add major gridlines in the y-axis
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    ax2.grid(which='minor',color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
   #Turn off minor ticks

    
    fig.tight_layout()
    
    # plt.plot(zetaVar,XB_inv)
    # plt.plot(zetaVar,ClusterNum)        