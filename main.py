# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:15:05 2020

@author: Theresa
"""

import numpy as np
import time
#import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn import metrics

from BirchEnsemble import BirchEnsemble
from Model import Model
from Dataset import Dataset
from utils import get_histogram_values, get_subplots, plot_scatterplots, plot_figure
from Birch import get_od_data, store_models, get_plot_data_birch, evaluate_and_plot, plot_birch_histogram, calculate_T_values, get_heuristic_values, get_ensemble_T_values, evaluate_model, T_average_nn, T_average, get_birch_models, get_birch_configs
from os.path import join as p_join
from pathlib import Path 
PROJDIR = str(Path(__file__).resolve().parents[0])
    
def plot_histograms(Models: list):
    '''Plots a histogram for each of the BIRCH ensembles showing how many data points have each cluster score
    
    arguments:
        Models: list of fitted(cluster_score not empty) BIRCH ensembles as Model instances
            
    returns nothing
    '''
    od_methods = [isinstance(M.od_method,BirchEnsemble) for M in Models]
    BE = list(np.array(Models)[np.array(od_methods)==True])
    print(len(BE),'Models')
    fig, ax = get_subplots(True, int((len(BE)-1)/5)+1, min(len(BE), 5), 'Occurances of Cluster Scores')
    for B in BE:
        plot_birch_histogram(B, int(BE.index(B)/5), BE.index(B)%5, ax)
    
    plot_figure(True, fig, None)    
    
def od_and_plot(data: Dataset, Models: list, trials=3, plot_detected=False, pdf=None) -> pd.DataFrame:
    '''Compares accuracies of given models on given datasets
    
    arguments:
        data: Dataset
        Models: list of Models
    optional:
        trials: how often each model should be evaluated
        plot_detected: whether to show the top n detected anoms of last trial for each model (n = number of true outliers) scatterplot for 2D data, labels of top n for higher dimensions
        pdf: pdf file to store the plots, if None plots are not stored
        
    returns 
        Dataframe with results
    '''
    trials = max(1, trials)
    df = pd.DataFrame(columns = ['Method','Metric', 'Value'])
    #find oscvm best
    fig, ax = get_subplots(plot_detected and data.X.shape[1]==2, int((len(Models)-1)/5)+1, min(len(Models), 5),'top n anoms shown in red')
    
    for M in Models:
        df = evaluate_model(df, data, M, trials, int(Models.index(M)/5), Models.index(M)%5, [M.name], ax, plot_detected)
        
    plot_figure(plot_detected and data.X.shape[1]==2, fig, pdf)
    
    return df

def compare_birch_configs(data: Dataset, T_values: np.ndarray, T_description: np.array, trials: int, differing_T: bool, modes=['median', 'max', 'mean'], ensembles=[1,2,3], plot_detected=False, plot_histograms=False, pdf=None) -> (pd.DataFrame, list):
    '''Plots accuracies of different Birch configurations to compare their performance. It produces one diagram for each ensemble size and OD method
    
    the catplot contains one diagram for each OD method
    each diagram contains a section for one ensemble which shows its accuracy based on metrics
    if the ensembles contain Birch configurations with the same Birch threshold (BEE) this value will be displayed on the x-Axis
    if they contain configurations with differing thresholds (BEM) the ensembles in one diagram will be numbered serially. The number will be shown on the x-Axis
    
    arguments:
        data: 0 if synthetic, 1 if mnist
        T_values: list of T's to compare
        T_description: list, BEE: T values if fixed, how it was calculated if heuristic. number if BEM
        trials: how often each setting is evaluated
        differing_T: true if T values within an ensemble differ
            false if T values within an ensemble are the same
    
    optional:
        modes: how to calculate cluster size per point from the trees
        ensembles: ensemble sizes to compare
        plot_detected: whether to show the top n detected anoms of last trial for each model (n = number of true outliers) scatterplot for 2D data, labels of top n for higher dimensions
        plot_histogram: if true plots histogram of clustersizes for each Birch Ensemble
        configurations: how many BEM's of same size
        pdf: pdf file to store the plots, if None plots are not stored
        
    returns
        1. Dataframe with results
        2. list of BE's as Model instances
    '''  
    trials = max(1, trials)
    df = pd.DataFrame(columns = ['Ens size', 'Mode', 'Method', 'Metric', 'Value'])
    df, Models = get_plot_data_birch(data, T_values, T_description, modes, ensembles, trials, differing_T, df, plot_detected, plot_histograms, pdf)
    
    with plt.style.context("seaborn-darkgrid"):
        sns.catplot(x="Method", y="Value", row='Mode', col='Ens size', hue="Metric", kind="bar",height=4, data=df)
    return df, Models
    

if __name__ == '__main__':
    
    # 1. evaluating several BIRCH configurations:
    
    # path to folder of data set
    DATA = p_join(PROJDIR, 'data')
    DATA_S = p_join(DATA, 'synthetic_small')
    
    # add name of data file to path
    data_path = p_join(DATA_S,'syn_data_shuffled.dat')
    
    # load data from file (synthetic data)
    data_r = np.loadtxt(data_path)
    X = data_r[:,0:2]
    od_labels = data_r[:,2]
    data = Dataset(X, od_labels)
    
    # or create new data set:
    #data = get_od_data(1, benign_labels=[1,7], malign_labels=[0]) # see documentation of function for other kinds of data sets
    
    print('data shape:', data.X.shape, '#anomalies:', len(np.where(data.od_labels==1)[0]))
    
    differing_T = False # False->BEE, True->BEM
    T= np.array([0.4,0.6,0.8,1.0,1.2]) # array of fixed values or None if heuristic values are wanted
    T_description, T_values = calculate_T_values(X=data.X, heuristic=T_average,T=T,differing_T=differing_T,n_heuristic=3, n=100)
    
    # store plots to pdf:
    out_pdf = p_join(DATA_S,'small_synthetic_dummy.pdf')
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
    
    trials = 10 # how often each configuration (T-arr, e, mode) should be evaluated
    start_time = time.perf_counter()
    df_fixed, Models_fixed = compare_birch_configs(data, T_values, T_description, trials=trials, modes=['median', 'max', 'mean'], differing_T=differing_T, 
                                       ensembles=[1,5,10,15], plot_detected=True, plot_histograms=False, pdf=pdf) # store scatterplots(if plot_detected=True) and histograms of clusterscores(if plot_histograms=True) for last trial of each configuration to pdf
    print (time.perf_counter() - start_time, "seconds")
    pdf.savefig() # store bar plot to pdf
    pdf.close()
    plt.show()
    
    # store results in dataframe to csv file
    #file = 'results.csv'
    #df_fixed.to_csv(p_join(DATA_S,file), encoding='utf-8', index=False)
    # store models in json format to file
    #models_fixed_file = p_join(DATA_S, 'models.txt')
    #store_models(Models_fixed, models_fixed_file)
    
    # plot histograms for each model
    # plot_histograms(Models_fixed)

    # 2. compare different algorithms:
    
    Models = [Model(BirchEnsemble(T_arr=np.array([0.8]*10), mode='median'), 'BE10 0.8 md'),
              Model(OneClassSVM(nu=0.18, gamma=0.5), 'OneClassSVM'),
              Model(IsolationForest(max_samples=300, n_estimators=200, contamination=4/500.0), 'isoFor')]
    trials=10
    # metric results in df_compare, training time in Models
    df_compare = od_and_plot(data, Models, trials=trials)
    with plt.style.context("seaborn-darkgrid"):
        sns.catplot(x="Method", y="Value", hue="Metric", kind="bar",height=4, data=df_compare)
    plt.show()
    
    for M in Models:
        print(M.name, M.fit_time)
    