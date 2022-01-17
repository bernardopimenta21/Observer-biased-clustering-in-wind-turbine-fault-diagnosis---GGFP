# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:04:49 2021

@author: berna
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from load_data import load_data
import umap   # pip install umap-learn
import seaborn as sns
from sklearn.metrics.cluster import adjusted_rand_score
from rand_index import rand_index_score



def hierarc(data,n_clusters, max_iter=100,min_impro=1e-6):


    cluster = AgglomerativeClustering(n_clusters, affinity='euclidean')
    cluster.fit_predict(data)


    return  cluster




# plt.scatter(y[:,0],y[:,1], c=cluster.labels_, cmap='rainbow')


if __name__ == '__main__':
    dataset='wt_test'              #dataset='iris','bearing','wt_test'
    data, label_gt =load_data(dataset)


    n_clusters=4
    
    cluster = hierarc(data,n_clusters)
    
    y = umap.UMAP(n_neighbors=10,min_dist=0, random_state=10).fit_transform(data)

    
    "Seaborn plot"
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set_style("white")
    
    # palette = sns.color_palette("bright",n_clusters)
    palette = sns.color_palette([(1.0, 0.48627450980392156, 0.0),(0.10196078431372549, 0.788235294117647, 0.2196078431372549),(0.9450980392156862, 0.2980392156862745, 0.7568627450980392),(0.00784313725490196, 0.24313725490196078, 1.0)])
    # palette = sns.color_palette([(1.0, 0.48627450980392156, 0.0),(0.10196078431372549, 0.788235294117647, 0.2196078431372549)])

    ax=sns.scatterplot(y[:, 0], y[:, 1], hue=cluster.labels_, legend='full',palette=palette)
    
    
    if n_clusters==2:
          aux=label_gt >1
          label_gt=aux+1 -1
    else:
        label_gt=label_gt  #ground truth label
    print('Hierarchical Rand Index :', rand_index_score(label_gt, cluster.labels_))
    
    print('Hierarchical Adjusted Rand Index :', adjusted_rand_score(label_gt, cluster.labels_))
