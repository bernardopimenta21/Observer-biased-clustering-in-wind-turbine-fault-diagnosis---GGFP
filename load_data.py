# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:27:19 2021

@author: Bernardo
"""
from sklearn import datasets
import numpy as np
import pandas as pd

def load_data(dataset):
    
    if dataset=='iris':
        "Iris data"
        iris = datasets.load_iris()
        data = iris.data 
        label_gt = iris.target

        
    elif dataset=='bearing':
                
        "Bearing data"
        data = pd.read_csv("bearingdata.csv",header=None)        
        data=data.to_numpy()
        
        label_bearing = pd.read_csv("bearingdata_label.csv", header=None)
        label_bearing=label_bearing.to_numpy().T
        label_bearing=label_bearing.reshape(len(label_bearing))
        
        # if nClusters==2:
        #       a=label_bearing >1
        #       label_gt=a+1 - 1
        # else:
        #     label_gt=label_bearing #ground truth label
            
        label_gt=label_bearing #ground truth label
        
        
    elif dataset=='bearing_unprocessed':
                
        "Bearing data"
        data = pd.read_csv("bearing_notprocessed.csv",header=None)        
        data=data.to_numpy()
        
        label_bearing = pd.read_csv("bearingdata_label.csv", header=None)
        label_bearing=label_bearing.to_numpy().T
        label_bearing=label_bearing.reshape(len(label_bearing))
        
        # if nClusters==2:
        #       a=label_bearing >1
        #       label_gt=a+1 - 1
        # else:
        #     label_gt=label_bearing #ground truth label
            
        label_gt=label_bearing #ground truth label
                
   
    
    elif dataset=='wt_test':
        "WT data"
        data = pd.read_csv("wt_data_test.csv",header=None)        
        data=data.to_numpy()
        
        label_wt = pd.read_csv("wt_labels_test.csv", header=None)
        label_wt=label_wt.to_numpy()
        label_wt=label_wt.reshape(len(label_wt))
        label_gt=label_wt+1

    elif dataset=='wt_clean':
        "WT data"
        data = pd.read_csv("data_WT_clean.csv",header=None)        
        data=data.to_numpy()
        
        label_wt = pd.read_csv("label_WT_clean.csv", header=None)
        label_wt=label_wt.to_numpy()
        label_gt=label_wt.reshape(len(label_wt))
        label_gt=np.array(label_gt, dtype=int)        
    elif dataset=='wt_tr':
        "WT data"
        data = pd.read_csv("wt_data_training.csv",header=None)        
        data=data.to_numpy()
        
        label_wt = pd.read_csv("wt_labels_training.csv", header=None)
        label_wt=label_wt.to_numpy()
        label_wt=label_wt.reshape(len(label_wt))
        label_gt=label_wt+1
        
        
    return data , label_gt

if __name__ == '__main__':

    dataset='wt_tr'
    data, label_gt =load_data(dataset)


