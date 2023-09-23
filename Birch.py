# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:12:32 2020

@author: Theresa

"""

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn import datasets
from sklearn import metrics

#import mnist
from tensorflow.keras.datasets import mnist

import numpy as np
import time
import itertools as it
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import random
import json

from utils import get_histogram_values, flatten_last_axes, get_subplots, plot_scatterplots, plot_images, plot_figure, partition
from BirchEnsemble import BirchEnsemble
from Model import Model
from Dataset import Dataset

#colors = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00', '#129900', '#993311'])

acc_metrics = ['AP', 'ROC_AUC', 'p@n','OSTM', 'OSTM ratio']#['prec', 'rec', 'p@n', 'f1']
    
def plot_birch_histogram(BE: Model, row: int, column: int, ax: np.array):
    '''Plots histogram of cluster sizes per point for specific Birch Tree on specific Dataset
    
    arguments:
        BE: Birch Tree as Birch ensemble of one tree
        row, column, index: where to plot
    
    returns nothing
    '''
    #BE.fit(data.X)
    cs = BE.od_method.cluster_score #'BE'+str(e)+' '+str(T_description[index])+' '+mode
    c,l = get_histogram_values(cs, BE.name, row, column, ax, "cluster score", "occurances")

def get_small_synthetic_data(sample_size: int) -> Dataset:
    '''Creates new data set with 4 anomalies added
    
    arguments: 
        sampleSize: number of data points
        
    returns:
        1. array containing normal data points first and anomalies after
        2. array with an entry for each datapoint in X. contains 1 if data point is anomaly, 0 if not
    '''
    anomalies = np.array([[1,1],[0,-6],[2.01,-2],[2.0,7]])
    X, y = datasets.make_blobs(n_samples=sample_size-len(anomalies), n_features=2, cluster_std=[1.0, 1.0, 1.0], centers=[[-4,-4],[4,3],[-3,5]])
    X = np.concatenate((X, anomalies), axis=0)
    od_labels = np.concatenate((np.zeros(sample_size-len(anomalies)), [1]*len(anomalies)), axis=0)
    return Dataset(X, od_labels)

def get_big_synthetic_data(sample_size: int) -> Dataset:
    '''Creates a big synthetic data set
    
    arguments: 
        sampleSize: number of data points
        
    returns:
        1. array containing normal data points first and anomalies after
        2. array with an entry for each datapoint in X. contains 1 if data point is anomaly, 0 if not
    '''
    #anomalies = np.array([[60,-80],[180,-10],[0,-200],[-50,-200]])
    anomalies = np.array([[-15,10], [-20,7], [-10,-25], [-12,-28], [-16,-23], [8,25], [10,26], [34,-30], [31,-6], [32.5, -5], [12,0], [8,20], [-15,-25], [-20,-22]])
    anomalous_cluster, y = datasets.make_blobs(n_samples=6, n_features=2, cluster_std=[2.0], centers=[[-15,-25]])
    #anomalous_cluster_2, y = datasets.make_blobs(n_samples=10, n_features=2, cluster_std=[2.0], centers=[[-20,35]])
    #anomalous_cluster_3, y = datasets.make_blobs(n_samples=10, n_features=2, cluster_std=[2.0], centers=[[10,35]])
    a = len(anomalies) + len(anomalous_cluster) #+ len(anomalous_cluster_2) + len(anomalous_cluster_3)
    
    normal_cluster, y = datasets.make_blobs(n_samples=sample_size-a, n_features=2, cluster_std=[3.0, 2.0, 1.80, 2.50, 2.70, 2.00], centers=[[-6,-6],[-5,23],[10,-20], [17,-20], [25,14],[30,30]])
    #normal_cluster, y = datasets.make_blobs(n_samples=sample_size-a, n_features=2, cluster_std=[1.4]*12, centers=[[-6,-6],[-5,23],[10,-20], [27,-20], [25,14],[30,30], [0,10],[-16,20], [-14,28], [20,-10],[10,10],[-17,-5]])
    X = np.concatenate((normal_cluster, anomalous_cluster), axis=0)
    #X = np.concatenate((X, anomalous_cluster_2), axis=0)
    #X = np.concatenate((X, anomalous_cluster_3), axis=0)
    X = np.concatenate((X, anomalies), axis=0)
    #od_labels = np.zeros(sample_size)
    od_labels = np.concatenate((np.zeros(sample_size-a), [1]*a), axis=0)
    return Dataset(X, od_labels)

def load_mnist() -> Dataset:
    '''Loads mnist data and returns as dataset
    
    arguments: none
    
    returns: 
        Dataset: whole mnist dataset with images and labels (NOT od labels yet) 
    '''
    (X_training, Y_training), (X_test, Y_test) = mnist.load_data()
    
    images = np.concatenate((X_training, X_test), axis=0)
    labels = np.concatenate((Y_training, Y_test), axis=0)    
    
    data = Dataset(images, None, labels)
    print(len(data.labels))
    return data

def get_complex_data(dataset: int, benign_labels: np.ndarray, malign_labels: np.ndarray, contamination_rate: float) -> Dataset:
    """Creates new data set of MNIST pictures with pictures that show numbers from 0-9
    normal data consists of 0's in the data set 
    anomalies are randomly chosen from the other classes and make up 1% of the data
        
    arguments:
        dataset: if 0 synthetic data, if 1 other (mnist, Fashion mnist)
        benign_labels: normal data labels
        malign_labels: anomalous data labels
        contamination: how much of the data should be anomalous
    
    returns:
        Dataset: od dataset with given distribution of normal and anomalous data
    """
    if dataset==1:
        data = load_mnist()
    else:
        raise ValueError('no valid dataset. 0=synthetic 1=mnist ...')
        return
    
    od_data = data.makeODDataset(benign_labels, malign_labels, contamination_rate)
    od_data.X = flatten_last_axes(od_data.X, 1)
    
    # by default int8 but overflow in sklearn.cluster.Birch implementation if smaller than 64
    od_data.X = np.array(od_data.X, dtype=np.int64)
    a, n = np.unique(od_data.labels[od_data.od_labels!=0], return_counts=True)
    print('Anomalies:',a,'occurances:',n)
        
    return od_data

def random_order(data: Dataset):
    """OD data contains normal data first and anomalies second. They are sorted by normality.
    To have a random order of normal data and anomalies, we randomize the order of data points (data.X), od_labels and, if available, true labels (digit in mnist data), accordingly insitu
    
    arguments:
        data: od dataset
        
    returns:
        nothing, data is shuffled insitu
    """
    r = np.random.permutation(len(data.od_labels))
    data.od_labels = data.od_labels[r]
    data.X = data.X[r]
    if data.labels is not None:
        data.labels = data.labels[r]

def get_od_data(dataset: int, benign_labels=np.array([0]), malign_labels=np.array([1,2,3,4,5,6,7,8,9]), contamination_rate=0.01, sample_size=500000) -> Dataset:
    """Creates OD Dataset object
    
    arguments:
        dataset: if 0 synthetic data, if 1 other (mnist, .. others can be added)
    optional:
        benign_labels: normal data labels (0)
        malign_labels: anomalous data labels (1,2,3,4,5,6,7,8,9)
        contamination: how much of the data should be anomalous
        
    returns: 
        OD Dataset with data points and labels: normal=0, anomalous=1
    """
    # 0 = synthetic, always 500
    if dataset==0:
        data = get_small_synthetic_data(500)
    # 1 complex data set (eg mnist)
    elif dataset==1:
        data = get_complex_data(dataset, benign_labels, malign_labels, contamination_rate)
    # 2 = big synthetic dataset
    else:
        data = get_big_synthetic_data(sample_size)
    
    random_order(data)
    return data
   
def get_top_anoms(od_values: np.ndarray, y_score: np.ndarray, n: int) -> np.ndarray:
    """Calculates od_labels of top n anomalies with highest OS
    
    arguments: 
        od_values: array with one value for each data point, sorted like y_score
        y_score: outlier score for each data point as calculated by some model
        n: of how many most outlying cases the od_values are desired
        
    returns:
        array with n od_values
    """
    idx = np.argsort(y_score)
    n_od_values = od_values[idx][::-1][:n]
    return n_od_values
           
def precision_at_n(od_labels: np.ndarray, y_score: np.ndarray) -> float:
    """Calculates precision at n where n is the number of true outliers
    prec@n = #(anoms in top n) / #(all anoms)
    
    arguments:
        od_labels: array with an entry for each datapoint in X. contains 1 if data point is anomaly, 0 if not
        y_score: outlier score for each datapoint, data points in same order as in od_labels
        
    returns
        precision at n value
    """
    n_od_labels = get_top_anoms(od_labels, y_score, int(np.sum(od_labels)))
    return np.sum(n_od_labels) / np.sum(od_labels)
  
def os_mean(y_score: np.ndarray, m: int) -> float:
    """Calculates average outlier score for top m outlying cases (only for BE's as other method's OS is not necessarily in [0,1])
    
    arguments:
        y_score: outlier score for each datapoint
        m: how many most outlying cases should be considered
        
    returns
        average outlier score of top m elements
    """
    most_out = np.sort(y_score)[::-1][:m]
    return np.mean(most_out)
    
def get_accuracy(od_labels: np.ndarray, y_score: np.ndarray, num_metrics = 3) -> list:
    """Finds points with clustersizes smaller than a threshold 'maxsize', evaluates the accuracy of the ensemble's results based on the metrics precision, recall, prec@n, f1
       also plots scatter plots to see which datapoints are labelled as outliers in the ensemble
       
    arguments:
        od_labels: array with an entry for each datapoint in X. contains 1 if data point is anomaly, 0 if not
        y_score: outlier score for each datapoint
    optional:
        num_metrics=3: how many of the metrics should be calculated (5 for BE (3+OSTM+OSTM ratio), 3 else)
        
    returns:
        list with calculated values for the chosen metrics 
    """
    AP = metrics.average_precision_score(od_labels, y_score)
    ROC_AUC= metrics.roc_auc_score(od_labels, y_score)
    prec_at_n = precision_at_n(od_labels, y_score)
    result = [AP, ROC_AUC, prec_at_n]
    
    if num_metrics == len(acc_metrics):
        m = int(np.sum(od_labels)) #m=number of true outliers
        OSTM = os_mean(y_score, m)
        OSTM_ratio = OSTM / os_mean(y_score, 2*m)
        result += [OSTM, OSTM_ratio]
    return result

def evaluate_model_accuracy(data: Dataset, M: Model, Value: list, num_metrics=3) -> np.ndarray:
    """Evaluates given Model on given Dataset
    
    arguments:
        data: Dataset to evaluate
        M: Model to evaluate
        Value: list, results for metrics are stored here
    optional:
        num_metrics=3: how many of the metrics should be calculated (5 for BE (3+OSTM+OSTM ratio), 3 else)
        
    returns:
        y_score: outlier scores for each value
    """
    start_time = time.perf_counter()
    M.fit(data.X)
    fit_time = time.perf_counter() - start_time
    y_score = M.score_samples(data.X)
    run_time = time.perf_counter() - start_time
    M.time += run_time
    M.fit_time += fit_time
    
    V = get_accuracy(data.od_labels, y_score, num_metrics)
    Value += V
    M.AP += [V[0]]
    M.roc_auc += [V[1]]
    M.prec_at_n += [V[2]]
    if num_metrics == 5:
        M.OSTM += [V[3]]
        M.OSTM_ratio += [V[4]]
    return y_score

def evaluate_model_acc_multiple_trials(data: Dataset, M: Model, trials: int, Value: list, num_metrics=3):
    """Evaluates accuracy for a given model on a given dataset multiple times
    
    arguments: 
        data: Dataset object containing X and od_labels
        M: Model to evaluate
        trials: how often the model should be evaluated
        Value: results for metrics
    optional:
        num_metrics=3: how many of the metrics should be calculated (5 for BE (3+OSTM+OSTM ratio), 3 else)
        
    returns 
        y_score of last trial
    """
    for i in range(trials):
        y_score = evaluate_model_accuracy(data, M, Value, num_metrics)
    M.time /= trials
    M.fit_time /= trials
    return y_score

def get_number_of_datapoints_to_show(od_labels: np.ndarray, y_score: np.ndarray):
    """Returns the number of datapoints that have an outlier score at least as big as the smallest outlier score of the true anomalies
    
    arguments:
        od_labels: array with an entry for each datapoint in X. contains 1 if data point is anomaly, 0 if not
        y_score: outlier scores of datapoints in the same order as in od_labels
    
    returns:
        n: number of data points
    """
    m = min(y_score[od_labels==1])
    n = len(y_score[y_score>=m])
    # to show the top n (=amount of true outliers) detected anomalies use the following:
    #n = int(np.sum(od_labels))
    return n

def plot_detected_anomalies(data: Dataset, y_score: np.ndarray, rows: int, columns: int, ax: np.array, caption: str):
    """Plots scatterplots where all data points with an outlier score at least as high as the smallest score of an anomaly are red, others blue
    
    arguments: 
        data: Dataset object containing X and od_labels
        y_score: outlier scores of datapoints in the same order as in data
        rows: number of rows of plot
        columns: number of columns of plot
        ax: array of subplots
        caption: descriptive text above the scatterplot
        
    returns nothing
    """
    idx = np.argsort(y_score)
    n = get_number_of_datapoints_to_show(data.od_labels, y_score)
    plot_scatterplots(data.X, idx, n, rows, columns, ax, caption)
    

def evaluate_and_plot(data: Dataset, M: Model, Value: list, trials: int, rows: int, column: int, ax: np.array, plot_detected: bool, num_metrics=3): #, index: int, configurations: int):
    """Creates ensemble of e trees, finds clustersizes for each data point and calculates metrics
    
    arguments:
        data: Dataset object containing X and od_labels
        M: Birch Ensemble to evaluate
        mode: OD method used to find the clustersize of each point
        T: array of T's used in ensemble
            if ensemble is a BEE it contains this T e times
            if it is a BEM it contains e randomly chosen T's
        Value: list of calculated metrics
        trials: number of runs per configuration
        rows, column: position to plot the results
        ax: array of subplots
        plot_detected: if true plots scatterplot with n anoms with highest outlier score marked 
    optional:
        num_metrics=3: how many of the metrics should be calculated (5 for BE (3+OSTM+OSTM ratio), 3 else)
        
    returns nothing
        calculated metrics for this ensemble are added to the list 'Value' 
    """
    y_score = evaluate_model_acc_multiple_trials(data, M, trials, Value, num_metrics)

    top_anoms = get_top_anoms(data.X, y_score, int(np.sum(data.od_labels)))
    top_od_labels = get_top_anoms(data.od_labels, y_score, int(np.sum(data.od_labels)))
    M.false_pos += [top_anoms[np.where(top_od_labels == 0)].tolist()]
    
    
    if data.X.shape[1] == 2 and plot_detected:
        plot_detected_anomalies(data, y_score, rows, column, ax, M.name)
    elif data.X.shape[1] != 2:
        #top_anoms = get_top_anoms(data.X, y_score, int(np.sum(data.od_labels)))
        #top_od_labels = get_top_anoms(data.od_labels, y_score, int(np.sum(data.od_labels)))
        #M.false_pos = top_anoms[np.where(top_od_labels == 0)]
        top_labels = get_top_anoms(data.labels, y_score, int(np.sum(data.od_labels)))
        if plot_detected:
            print(M.name+':',top_labels)

def plot_false_pos(M: Model, benign_labels=None, data=None, dim=(28,28)):
    """Plots name, amount of false positives and images of false positives for a Model trained on a datasets with images
    
    arguments: 
        M: Model
    optional:
        benign_labels=None: labels of benign classes
        data=None: Dataset
        dim=(28,28): dimensions of the picture
        
    returns nothing
    """
    
    print(M.name,'has',len(M.false_pos),'false positives')
    if len(M.false_pos) == 1:
        plot_images(np.array([M.false_pos]), 'false pos', benign_labels, data, dim)
    else:
        plot_images(M.false_pos, 'false pos', benign_labels, data, dim)
    
   
def evaluate_model(df: pd.DataFrame, data: Dataset, M: Model, trials: int, row: int, column: int, params: list, ax: np.array, plot_detected: bool) -> pd.DataFrame:
    """Calculates the accuracy of a given Model for multiple trials, adds results to dataframe and plots top n anomalies.
    
    arguments:
        df: Dataframe to add results
        data: Dataset to be examined
        M: Model
        trials: how often Model M should be fitted and evaluated
        row, column: in which subplot to plot the anomalies (only for the last trial)
        ax: array of subplots
        params: other values for the dataframe not dependant on the results and metrics
        plot_detected: if true top n anomalies of last trial are shown in scatterplot (only for 2d data)
        
    returns: 
        updated Dataframe
    """
    V = []
    # use all metrics if Model is a BE, use all except OSTM, OSTM ratio else
    if isinstance(M.od_method, BirchEnsemble):
        num_metrics = len(acc_metrics)
    else:
        num_metrics = 3
    evaluate_and_plot(data, M, V, trials, row, column, ax, plot_detected, num_metrics)
    df = update_df(df, trials, V, params, num_metrics)
    return df
        
def T_average_nn(X: np.ndarray, size: int) -> float:
    """Estimates the Birch T value based on a subsample of the data set
    
    threshold estimate: average distance to the closest neighbours of each sample point
    
    arguments:
        X: data array
        size: sample size
            
    returns
        T to use in Birch configurations
    """
    sample = X[np.random.choice(np.arange(len(X)), size, replace=False)]
    dist = metrics.pairwise.euclidean_distances(sample,sample)
    t = np.mean(np.sort(dist, axis=0)[1])
    return t

def T_average(X: np.ndarray, size: int) -> float:
    """Estimates the Birch T value based on a subsample of the data set
    
    threshold estimate: average distance to the neighbours of each sample point
    
    arguments:
        X: data array
        size: sample size
            
    returns
        T to use in Birch configurations
    """
    sample = X[np.random.choice(np.arange(len(X)), size, replace=False)]
    dist = metrics.pairwise.euclidean_distances(sample,sample)
    t = np.mean(np.sort(dist, axis=0)[1:])
    return t

def add_row_to_df(df: pd.DataFrame, params: list) -> pd.DataFrame:
    """Adds row with given values to dataframe
    
    arguments:
        df: Dataframe
        params: one value for each column in the dataframe
        
    returns:
        updated dataframe
    """
    row = {}
    for param in range(len(params)):
        row[df.columns.values[param]] = params[param]
    df = df.append(row, ignore_index=True)
    return df

def update_df(df: pd.DataFrame, trials: int, Value: list, params: list, num_metrics: int) -> pd.DataFrame:
    """Adds values for each trial and metric in V a row to the dataframe 
    
    arguments:
        df: Dataframe
        trials: how often each configuration was performed
        Value: list containing one value for each metrics and trial
        params: parameters of the configuration that are the same for each value
        num_metrics: how many of the metrics should be calculated (5 for BE (3+OSTM+OSTM ratio), 3 else)
    
    returns:
        updated dataframe
    """
    for i in range(trials):
        for j in range(num_metrics):
            p = list(params) + [acc_metrics[j]]
            p = p + [Value[i*num_metrics+j]]
            df = add_row_to_df(df, p)
    return df

def get_plot_data_birch(data: Dataset, T_values: np.ndarray, T_description: np.array, modes: list, ensembles: list, trials: int, differing_T: bool, df: pd.DataFrame, plot_detected: bool, plot_histograms: bool, pdf=None) -> (pd.DataFrame, list):
    """Evaluates different birch configurations 
    
    arguments:
        data: Dataset object containing X and od_labels (0 or 1)
        T_values: list of T's to comparemodes: od methods to compare
        T_description: list, BEE: T values if fixed, how it was calculated if heuristic. number if BEM,
        modes: how to calculate cluster size per point from the trees
        ensembles: ensemble sizes
        trials: how often each BE will be evaluated
        differing_T: true if thresholds within ensemble differ (BEM)
            false if thresholds within ensemble are the same (BEE)
        df: dataframe to store the data
        plot_detected: for 2D data: if true plots scatterplot with n anoms with highest outlier score marked 
        plot_histogram: if true plots histogram of clustersizes for each Birch Ensemble
    optional:
        pdf=None: pdf file to store the plots, if None plots are not stored
            
    returns:
        1. updated DataFrame
        2. list of Birch Ensembles as Models
    """
    #heuristic_T, T_description, T_values = calculate_T_values(data.X, T, heuristic, differing_T, configurations=configurations, n_heuristic=n_heuristic)
    print('T:',T_values)
    Models = []
    
    for e in ensembles:
         
        t = get_ensemble_T_values(T_values, e, differing_T)
        fig1, ax1 = get_subplots(plot_detected and data.X.shape[1]==2, len(modes), len(t), 'Ensemble size '+str(e))
        fig2, ax2 = get_subplots(plot_histograms, len(modes), len(t), 'Ensemble size '+str(e))
        
        for mode in modes:
            for index in range(len(t)):
                name = 'BE'+str(e)+' '+str(T_description[index])+' '+mode
                M = Model(BirchEnsemble(t[index],mode, str(T_description[index])), name)
                df = evaluate_model(df, data, M, trials, modes.index(mode), index, [e, mode, str(T_description[index])], ax1, plot_detected)
                if plot_histograms:
                    get_histogram_values(M.od_method.cluster_score, M.name, modes.index(mode), index, ax2, xlabel='cluster score',ylabel='amount') #M.y_score*100.0
                Models += [M]
                
        plot_figure(plot_histograms, fig2, pdf)
        plot_figure(plot_detected and data.X.shape[1]==2, fig1, pdf)
        plt.show()
            
            
    return df, Models
        
def calculate_T_values(X: np.ndarray, heuristic, differing_T: bool,  T=None, n_heuristic=5, n=10) -> (np.ndarray, np.ndarray):
    """Find T values to compare based on whether we want to compare heuristics or fixed values and BEM or BEE
    
    arguments: 
        X: data array
        heuristic: function(X, n), how to calculate heuristical T value based on subset of the data
        differing_T: true if BEM, false if BEE
    optional:
        T=None: contains fixed values or 'None' if T-values should be based on the heuristic
        n_heuristic=5: how many T values should be created from the heuristic
        n=10: sample size for heuristic function
        
    returns:
        T_description: list, BEE: T values if fixed, how it was calculated if heuristic. number if BEM,
        T_values: list of T's to compare
    """
    # find T values with heuristic
    if T is None:
        #heuristic_T = True
        T_description, T_values = get_heuristic_values(X, heuristic, differing_T, n_heuristic, n)
    # use given T values
    elif type(T) is np.ndarray: 
        #heuristic_T = False
        T_values = T
        T_description = T
    else:
        raise ValueError('invalid argument. T must be either None or array with T-values')
    
    if differing_T:
        configurations=5
        T_description = np.arange(configurations)+1#np.array(['0', '1', '2', '3', '4'])
        T_description = np.array([str(item) for item in T_description])
    
    return T_description, T_values


def get_heuristic_values(X: np.ndarray, heuristic, differing_T: bool, n_heuristic: int, n: int) -> (np.ndarray, np.ndarray):
    """Calculates heuristical T values, first value is calculated with specified heuristic function, 
    then this value is divided by two until 'n_heuristic' values are obtained
    
    arguments:
        X: data array
        heuristic: function, how to calculate heuristical T value based on a randomly chosen subset of X
        differing_T: true if BEM, false if BEE
        n_heuristic: how many T values should be created
        n: sample size for heuristic function
        
    returns:
        T_description: array, how T values were calculated, None if BEM
        T_values: array of heuristic T_values
    """
    heuristic = heuristic(X, n)#int(np.sqrt(len(X))/4))
    T_description = None
    if not differing_T:
        #T_description = np.array(['avNN', '/2', '/4', '/8', '/16'])
        temp = 2**np.arange(n_heuristic)
        T_description = np.array(["T_av"])
        T_description = np.concatenate((T_description, np.array(['/'+str(item) for item in temp[1:]])))
        #np.array(['avNN', '/2', '/4', '/8', '/16'])
    T_values = (2**np.arange(n_heuristic)).astype(np.float32)
    T_values = np.array([heuristic/item for item in T_values])    
    return T_description, T_values

def get_ensemble_T_values(T_values: np.ndarray, e: int, differing_T: bool) -> np.ndarray:
    """Chooses T values for ensemble and transforms in to matrix of shape e times configurations for comparison
    
    arguments:
        T_values: T's to choose from 
        e: ensemble size
        differing_T: true if T values within an ensemble differ
            false if T values within an ensemble are the same
        
    returns:
        matrix of shape e x configurations with e BIRCH thresholds for c ensembles
    """
    configurations=5
    # if T values within ensemble differ
    if differing_T:
        #t = np.array([np.random.permutation(x)[:e] for x in np.tile(T_values, (configurations,1))])
        # more T values than trees -> use all possible combinations of T values but maximum 'configurations'
        if len(T_values) >= e:
            t = np.array([np.array(x) for x in it.combinations(T_values, e)])
            np.random.shuffle(t)
            t = t[:min(len(t), configurations)]
        # else create 5 arrays of len e with ratio 1:1, 1:4,4:1,2:3,3:2, since we use e = 5*k (except for e=1) these ratios work for all e's except for 1:1, then we use the second T once more
        elif len(T_values) ==2:
            t = np.array([np.array([T_values[0]]*int(e/2.0) + [T_values[1]]*(e-int(e/2.0)))])
            t = np.concatenate((t, np.array([[T_values[0]]*1*int(e/5.0) + [T_values[1]]*4*int(e/5.0)])))        
            t = np.concatenate((t, np.array([[T_values[0]]*4*int(e/5.0) + [T_values[1]]*1*int(e/5.0)])))        
            t = np.concatenate((t, np.array([[T_values[0]]*2*int(e/5.0) + [T_values[1]]*3*int(e/5.0)])))        
            t = np.concatenate((t, np.array([[T_values[0]]*3*int(e/5.0) + [T_values[1]]*2*int(e/5.0)])))        
            '''
            partitions = np.array(partition(len(T_values), e))
            T_bem = []
            for i in range(len(partitions)):
                temp_T = []
                for j in range(len(T_values)):
                    temp_T += [T_values[j]]*partitions[i,j]
                T_bem += [temp_T]
            T_bem = np.array(T_bem)
            # if 3 T values (small, middle and big) are  provided, create combinations such that combinations of middle, small and high value is balanced
            if len(T_values) == 3:
                # smallest and biggest T value equally often
                m = T_bem[np.where((partitions[:,0] == partitions[:,2]) * (partitions[:,0]%(e/5) == 0))]
                # smallest and middle equally often
                b = T_bem[np.where((partitions[:,0] == partitions[:,1]) * (partitions[:,1] >= e/5))]
                # middle and biggest equally often
                s = T_bem[np.where((partitions[:,1] == partitions[:,2]) * (partitions[:,1] >= e/5))]
                # only smallest and biggest value, equally often
                t = np.array([np.array([T_values[0]]*int(e/2.0) + [T_values[2]]*(e-int(e/2.0)))])
                #if int(e/2.0) != e-int(e/2.0):
                #    t2 = np.array([np.array([T_values[0]]*(e-int(e/2.0)) + [T_values[2]]*int(e/2.0))])
                #    t = np.concatenate((t, t2))
                c = configurations-1
                # take equally many cases from m,b,s if possible
                if c%3==0:
                    t = np.concatenate((t, m[:min(len(m), int(c/3.0))]))
                    t = np.concatenate((t, s[len(s)-min(len(s), int(c/3.0)):]))
                    t = np.concatenate((t, b[:min(len(b), int(c/3.0))]))
                # else split c in two, one half for m and other half for s and b.  
                # if one of the quotients of floor(c/2), ceil(c/2) is even, use it to split into one part s and one part b, use the other one for m
                else:
                    if int(c/2.0)%2==0:
                        x = int(c/2.0)
                    else:
                        x = c - int(c/2.0)
                    y = c-x
                    t = np.concatenate((t, m[:min(len(m), y)]))
                    t = np.concatenate((t, s[len(s)-min(len(s), int(x/2.0)):]))
                    t = np.concatenate((t, b[:min(len(b), x-int(x/2.0))]))
            # not 3 T values, create distinct ensemble T values.
            '''
        # if more than 2 T-values are given, we use 'configurations' combinations of the T-values such that each combination occurs at most once
        else:
            partitions = np.array(partition(len(T_values), e))
            T_bem = []
            for i in range(len(partitions)):
                temp_T = []
                for j in range(len(T_values)):
                    temp_T += [T_values[j]]*partitions[i,j]
                T_bem += [temp_T]
            T_bem = np.array(T_bem)
            indices = np.random.permutation(np.arange(len(T_bem)))[:configurations]
            t = T_bem[indices]
        print('e=',e,'T=',t)
    else:
        t = np.repeat(T_values, e)
        t = t.reshape((len(T_values),e))
        print('T=', t[:,0],'x'+str(e))
    
    return t

def get_birch_configs(data: Dataset, heuristic=T_average_nn, T=None, modes=['mean', 'median', 'max'], ensembles=[1,2,3], differing_T=False, n_heuristic=5) -> list:
    """Creates Birch ensembles and returns them in a list
    
    arguments:
        data: Dataset 
        heuristic: function, how to calculate heuristical T value
        T: T values, if None chosen heuristically based on NN
        modes: modes of BE's
        ensembles: ensemble sizes
        differing_T: if true BEM else BEE
        n_heuristic: how many T values should be created from the heuristic
        
    returns:
        list of BE's
    """
    T_description, T_values = calculate_T_values(data.X, heuristic=heuristic, differing_T=differing_T, T=T, n_heuristic=n_heuristic)
    print(T_values)
    BE = []
    for e in ensembles:
        t = get_ensemble_T_values(T_values, e, differing_T)
        for mode in modes:
            for index in range(len(t)):
                name = str(T_description[index])
                BE += [BirchEnsemble(t[index],mode, name)]
    return BE  

def get_birch_models(BE: list) -> list:
    """Creates Birch ensemble models and returns them in a list
    
    arguments:
        BE: list of Birch Ensembles
        
    returns:
        list of BE's as Models
    """
    mode_names = {'mean': 'mn', 'max': 'mx', 'median': 'md'}
    return [Model(B, str(B.e)+' '+B.description+' '+mode_names[B.mode], short=B.description) for B in BE]
 
def get_average_pairwise_distance(X:np.ndarray, a: np.ndarray, labels: np.ndarray) -> float:
    """Calculates average distance between 2 instances of the same class for each class
    
    arguments: 
        X: Matrix with data points (eg data.X for Dataset instance)
        a: labels of data points in X
        labels: 2 labels to calculate average distance of instances
    
    returns
        average of distances between all pairs of instances of different classes
    """
    l1 = labels[0]
    l2 = labels[1]
    s1 = X[np.where(a==l1)]
    s2 = X[np.where(a==l2)]
    dist = metrics.pairwise.euclidean_distances(s1,s2)
    average_distance= np.mean(dist)
    return average_distance

def store_models(Models: list, file: str):
    """Stores each Model in list with all parameters in JSON format to the file
    
    arguments:
        Models: list of Models
        file: txt file to store the JSON to 
        
    returns nothing
    """
    print('storing',len(Models),'Models')
    data = []
    for M in Models:
        od_method = []
        if isinstance(M.od_method, OneClassSVM):
            od_method = {'type': 'ocsvm', 'kernel': M.od_method.kernel, 'nu': M.od_method.nu, 'gamma': M.od_method.gamma, 'degree': M.od_method.degree}
        elif isinstance(M.od_method, IsolationForest):
            od_method = {'type': 'if', 'n_estimators': int(M.od_method.n_estimators), 'max_samples': int(M.od_method.max_samples), 'contamination': M.od_method.contamination}
        elif isinstance(M.od_method, BirchEnsemble):
            if isinstance(M.od_method.T_arr[0], np.int32):
                print('convert to int')
                T = list(map(int, M.od_method.T_arr))
            else:
                T = list(M.od_method.T_arr)
            od_method = {'type': 'be', 'T_arr': T, 'e': M.od_method.e, 'mode': M.od_method.mode}
        else:
            raise ValueError('od method is not defined')
        model_data = {'od_method': od_method, 'name': M.name, 'short': M.short, 'time': M.time, 'fit_time': M.fit_time, 'AP': M.AP, 'roc_auc': M.roc_auc, 'prec_at_n': M.prec_at_n, 'OSTM': M.OSTM, 'OSTM_ratio': M.OSTM_ratio}
        data += [model_data]
    with open(file, 'w') as fp:
        json.dump(data, fp)
    
def load_models(file: str) -> Model:
    """Loads Models from file where they have been stored in JSON format
    
    arguments:
        file: txt file where the JSON is stored
        
    returns:
        list of Model objects from file
    """
    Models = []
    with open(file, "r") as read_file:
        print("load models from",file)
        data = json.load(read_file)
        for d in data:
            method = d['od_method']
            if method['type'] == 'be':
                od_method = BirchEnsemble(T_arr = method['T_arr'], mode=method['mode'])
            elif method['type'] == 'ocsvm':
                od_method = OneClassSVM(kernel=method['kernel'], nu=method['nu'], gamma=method['gamma'], degree=method['degree'])
            elif method['type'] == 'if':
                od_method = IsolationForest(n_estimators=method['n_estimators'], max_samples=method['max_samples'], contamination=method['contamination'])
            M = Model(od_method=od_method, name=d['name'], short=d['short'])
            # M.y_score=d['y_score']
            # M.false_pos=d['false_pos']
            M.AP=d['AP']
            M.roc_auc=d['roc_auc']
            M.prec_at_n=d['prec_at_n']
            M.OSTM=d['OSTM']
            M.OSTM_ratio=d['OSTM_ratio']
            M.time = d['time']
            try:
                M.fit_time = d['fit_time']
            except:
                M.fit_time =0
            Models += [M]
    return Models


if __name__ == '__main__':

    T = 2000
    print('T=', T)
    
    '''
    data_r = np.loadtxt('syn_data_shuffled.dat')
    x = data_r[:,0:2]
    y = data_r[:,2]
    data = Dataset(X=x,od_labels=y)
    isoFor = IsolationForest()
    isoFor.fit(data.X)
    print(metrics.roc_auc_score(data.od_labels, -isoFor.score_samples(data.X)))
    data_r = np.loadtxt('mnist_data_shuffled.dat')
    x = data_r[:,0:784]
    y = data_r[:,784]
    z = data_r[:,785]
    data = Dataset(X=x,od_labels=y, labels=z)#x = data_r[:,0:2]
    '''
    #y = data_r[:,2]
    #z = data_r[:,785]
    #    y = data[:, 1]
    #data = load_mnist_from_file()
    #print(len(data.X))
    #brc = Birch(threshold = T, n_clusters = None)
    #brc.fit(data.X)
    #print(len(np.unique(brc.subcluster_labels_)), 'clusters')