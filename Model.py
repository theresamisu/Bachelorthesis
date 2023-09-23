# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:14:45 2020

@author: Theresa
"""

import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from BirchEnsemble import BirchEnsemble
from sklearn.neighbors import LocalOutlierFactor

class Model:
    
    
    def __init__(self, od_method, name, short=None):
        #, y_score=[], false_pos=[], time=0, fit_time=0, AP=[], roc_auc=[], prec_at_n=[], OSTM=[], OSTM_ratio=[]
        self.od_method = od_method
        self.name = name
        if isinstance(od_method, BirchEnsemble):
            self.type = 'be'
        elif isinstance(od_method, OneClassSVM):
            self.type = 'ocsvm'
        elif isinstance(od_method, IsolationForest):
            self.type = 'if'
        else:
            self.type = 'undefined'
        if short is None:
            self.short = name
        else:
            self.short = short
        self.y_score = []
        self.false_pos = []
        self.time = 0
        self.fit_time = 0
        self.AP = []
        self.roc_auc = []
        self.prec_at_n = []
        self.OSTM = []
        self.OSTM_ratio = []
    
        
    def fit(self, X: np.ndarray):
        self.od_method.fit(X)
        
    def score_samples(self, X:np.ndarray) -> np.ndarray:
        if isinstance(self.od_method, LocalOutlierFactor):
            y_score = -self.od_method.negative_outlier_factor_
        else:
            y_score = -self.od_method.score_samples(X)
        return y_score
        # svm, IF, the smaller score samples the better, need to transform to the bigger the better by using negative value
        