# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:57:39 2020

@author: Theresa
"""

import numpy as np

class Dataset:
    
    def __init__(self, X, od_labels, labels=None):
        self.X = X
        self.od_labels = od_labels
        self.labels = labels
        self.contamination = None
     
    def getBigSampleSize(self, benign: int, malign: int, contamination_rate: float) -> (int, int, int):
        ''' Calculates very big sample size with the given amount of benign and malign cases such that a given part is anomalous
        
        arguments:
            benign: amount of benign cases in data
            malign: amount of malign cases in data
            contamination: what part of the data should be anomalous, 0.001<=contamination<1
            
        return: 
            size of OD dataset with this contamination rate, amount of anoms, amount of normal data
        '''
        permille = int(contamination_rate*1000)
        if int(benign/ (1000-permille)) <= int(malign / permille):
            num_anoms = int(benign/(1000-permille)) * permille
            num_normal = int(benign/(1000-permille)) * (1000-permille)
            sampleSize = num_anoms + num_normal
        else:
            num_anoms = int(malign/permille) * permille
            num_normal = int(malign/permille) * (1000-permille)
            sampleSize = num_anoms + num_normal
        return sampleSize, int(num_anoms), int(num_normal)
    
    def makeODDataset(self, benign_labels: np.ndarray, malign_labels: np.ndarray, contamination_rate:float):
        '''Creates OD dataset which contains contamination_rate anomalies
        
        arguments:
            benign_labels: images with those labels are normal data
            malign_labels: images with those labels are anomalies
            contamination_rate: how much data in dataset is anomalous
            
        returns:
            Dataset where anomalies have label -1 and normal data has label 1
        '''
        
        X, labels = self.get_data(benign_labels, malign_labels, contamination_rate)
        
        od_labels = np.copy(labels)
        od_labels[np.where(np.isin(labels, malign_labels))] = 1
        od_labels[np.where(np.isin(labels, benign_labels))] = 0
        
        return Dataset(X, od_labels, labels)
    
    def get_data(self, benign_labels: np.ndarray, malign_labels: np.ndarray, contamination_rate: float) -> (np.ndarray, np.ndarray):
        '''Finds benign and malign instances and corresponding labels with specified contamination rate
        
        arguments: 
            benign_labels: labels of normal data
            malign_labels: labels of anomalies
            contamination_rate: what part of dataset should be anomalous
        
        returns:
            array with normal and anomalous instances
            array with labels according to parameters (not 0/1 yet)
        '''
        benign = np.where(np.isin(self.labels, benign_labels)==True) # labels in benign_labels
        malign = np.where(np.isin(self.labels, malign_labels)==True)
        
        sample_size, num_anoms, num_normal = self.getBigSampleSize(len(benign[0]), len(malign[0]), contamination_rate)
        print('Dataset size: ',sample_size, '( anoms:',num_anoms,'+ normal:', num_normal,')')
        self.contamination = contamination_rate
        
        idx = np.random.permutation(len(self.X))
        normal_im = self.X[idx][np.where(np.isin(self.labels[idx],benign_labels))][:num_normal]
        normal_labels = self.labels[idx][np.where(np.isin(self.labels[idx], benign_labels))][:num_normal]
        
        idx = np.random.permutation(len(self.X))
        anom_im = self.X[idx][np.where(np.isin(self.labels[idx],malign_labels))][:num_anoms]
        anom_labels = self.labels[idx][np.where(np.isin(self.labels[idx],malign_labels))][:num_anoms]
        
        X = np.concatenate((normal_im, anom_im))
        labels = np.concatenate((normal_labels, anom_labels))
        
        return X, labels
    
if __name__ == '__main__':
    print("Reload class Dataset")