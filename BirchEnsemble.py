# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:58:17 2020

@author: Theresa
"""
from sklearn.cluster import Birch
import numpy as np

class BirchEnsemble:
   
    
    def __init__(self, T_arr, mode='median', descr=None):
        self.T_arr = T_arr
        self.mode = mode
        self.e = len(T_arr)
        if descr is None:
            self.description = str(self.e)+' '+str(T_arr[0])+' '+mode
        else:
            self.description = descr
        self.cspp = None
        self.cluster_score = None
    
    def fit(self, X: np.ndarray):
        '''Creates ensemble of size "e" and calculates the final cluster size for each point according to the mode
        and from that the outlier score
        arguments:
            X: data array
            
        returns nothing
        '''
        self.cspp = np.array([self.getClusterSizesInTree(X, np.random.permutation(X.shape[0]), tree) for tree in np.arange(self.e)])
        if self.mode=="max":
            cs = np.max(self.cspp, axis=0)
        elif self.mode=="mean":
            cs = np.mean(self.cspp, axis=0)
        else:
            cs = np.median(self.cspp, axis=0)
        
        self.cluster_score = cs
        #print(np.sort(cs))
        #print(np.unique(cs, return_counts=True))
        #self.y_score = self.getOutlierScore(cs)
        #print(np.sort(self.y_score))
        
    def score_samples(self, X: np.ndarray):
        '''Returns y_score for fitted data, if not fitted yet raise error 
        
        arguments:
            X: data array
            
        returns y_score for the fitted data
        '''
        if self.cluster_score is not None:
            return self.getOutlierScore()
        else:
            raise ValueError("BIRCH ensemble must be fitted first")
    
    def calcBirch(self, T: float, Y: np.ndarray) -> Birch:
        '''Shuffles data array and builds a Birch tree with threshold T
        
        arguments: 
            T: T value for the Birch tree
            Y: shuffled data array
            
        returns:
            1. Birch tree build with shuffled data in Y and threshold T
        '''
        brc = Birch(threshold=T, n_clusters=None)
        brc.fit(Y)
        return brc
    
    def getClusterSizesOfPoints(self, ind: np.ndarray, labels: np.ndarray, subclusterLabels: np.ndarray) -> np.ndarray:
        '''Maps data points to size of assigned cluster
        
        arguments:
            ind: indices of data points in shuffled array
            labels: cluster label of each data point
            subclusterLabels: array with cluster labels
        
        returns:
            size of cluster to which the datapoints have been assigned to by the Birch configuration
        '''
        clusterLabels, clusterSizesPerCluster = np.unique(labels, return_counts=True)
        
        # special case when a cluster is empty: cluster occurs in subclusterLabels but there are no data points assigned to it -> it does not appear in brc.labels_
        # -> it does not appear in clusterLabels and its size (0) does not appear in clusterSizesPerCluster -> clusterSizesPerCluster length is less than biggest element in brc.labels_
        # messes with clusterSizesPerCluster[labels]
        # rare case (mostly with small T's and lots of small clusters)
        while not len(clusterLabels) == len(subclusterLabels):
            skipped_ind = np.where(np.isin(subclusterLabels, clusterLabels) == False)[0][0]
            
            left = clusterLabels[:skipped_ind]
            right = clusterLabels[skipped_ind:]
            clusterLabels = np.concatenate((left, np.array([skipped_ind]), right))
            
            left = clusterSizesPerCluster[:skipped_ind]
            right = clusterSizesPerCluster[skipped_ind:]
            clusterSizesPerCluster = np.concatenate((left, np.array([0]), right))
                
        cs = np.zeros(len(ind))
        cs[ind] = clusterSizesPerCluster[labels]
        return cs
       
    def getClusterSizesInTree(self, X: np.ndarray, ind: np.ndarray, tree: int) -> np.ndarray:
        '''Creates new BIRCH tree within an ensemble and updates biggest cluster size per point accordingly
        
        arguments: 
            X: data array
            ind: indices of X in random order
            tree: integer representig the position of the tree within the examined ensemble (e.g. if it is the second tree in an ensemble of size 3 it would be 2)
        
        returns array with clustersize for each point
        '''
        T = self.T_arr[tree]
        brc = self.calcBirch(T, X[ind])
        cs = self.getClusterSizesOfPoints(ind, brc.labels_, brc.subcluster_labels_)
        return cs
    
    def getOutlierScore(self) -> np.ndarray:
        '''Calculates an outlier score for each data point based on its cluster score (attribute)
        
        arguments: None
            
        returns:
            negative(!) outlierscore for each data point
            (negative because other OD methods also give high scores for normal points and low for anomalous ones, this makes the generalisation with class Model.py easier)
        '''
        # linear score
        #return -(np.max(self.cluster_score) - self.cluster_score +1) / (np.max(self.cluster_score))
        return -1/self.cluster_score