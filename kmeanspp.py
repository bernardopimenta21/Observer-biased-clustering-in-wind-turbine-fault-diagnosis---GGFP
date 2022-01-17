# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:36:13 2021
From:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

@author: Bernardo
"""

from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import datasets
import pandas as pd 


def load_data(dataset):
    
    if dataset=='iris':
        "Iris data"
        iris = datasets.load_iris()
        data = iris.data 
        label_iris = iris.target
        label_gt=label_iris
        
    elif dataset=='bearing':
                
        "Bearing data"
        data = pd.read_csv("bearingdata.csv",header=None)        
        data=data.to_numpy()
        
        label_bearing = pd.read_csv("bearingdata_label.csv", header=None)
        label_bearing=label_bearing.to_numpy().T
        label_bearing=label_bearing.reshape(len(label_bearing))
        
        
        if nClusters==2:
             a=label_bearing >1
             label_gt=a+1
        else:
            label_gt=label_bearing #ground truth labell

    return data , label_gt

if __name__ == '__main__':

#initialization: kmeans++ or random
    dataset='bearing'
    nClusters=2

    data, label_gt =load_data(dataset,nClusters)
    
    kmeans = KMeans(nClusters,init='k-means++').fit(data)
    
    C=kmeans.cluster_centers_
    
    
    print('Kmeans++ Adjusted Rand Index :', adjusted_rand_score(label_gt, kmeans.predict(data)))
    
             
    
