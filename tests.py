# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:18:44 2020

@author: Theresa

"""

from sklearn.cluster import Birch
from sklearn.svm import OneClassSVM
from sklearn import datasets
from sklearn import metrics
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import unittest
from BirchEnsemble import BirchEnsemble
from Model import Model
from Dataset import Dataset
from utils import flatten_last_axes, partition, get_subplots, get_histogram_values
from Birch import get_small_synthetic_data, get_big_synthetic_data, load_mnist, get_complex_data, get_od_data, get_top_anoms, precision_at_n, os_mean, get_accuracy, evaluate_model_accuracy, evaluate_model_acc_multiple_trials, evaluate_model, evaluate_and_plot, get_number_of_datapoints_to_show, plot_detected_anomalies, update_df, add_row_to_df, get_plot_data_birch, calculate_T_values, get_heuristic_values, get_ensemble_T_values, get_birch_configs, get_birch_models, T_average, T_average_nn, random_order
acc_metrics = ['AP', 'ROC_AUC', 'p@n', 'OSTM', 'OSTM ratio']

class Tests(unittest.TestCase):
    
    
    def test_get_subplots(self):
        plot = False
        rows = 2
        columns = 3
        title = 'test get_subplots'
        fig, ax = get_subplots(plot, rows, columns, title)
        self.assertTrue(fig is None)
        self.assertEqual(len(ax),0)
        
        plot = True
        fig, ax = get_subplots(plot, rows, columns, title)
        fig.tight_layout()
        plt.show()
        self.assertEqual(ax.shape,(rows, columns))
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertEqual(fig._suptitle.get_text(), title)
        
    def test_flatten_last_axes_2(self):
        data = load_mnist()
        self.assertTrue(data.X.shape == (70000, 28, 28))
        
        flat_X = flatten_last_axes(data.X, 2)
        self.assertTrue(flat_X.shape == (70000, 28, 28))
    
    def test_flatten_last_axes_1(self): 
        data = load_mnist()
        flat_X = flatten_last_axes(data.X, 1)
        self.assertTrue(flat_X.shape == (70000, 784))
        
    def test_flatten_last_axes_0(self): 
        data = load_mnist()
        flat_X = flatten_last_axes(data.X, 0)
        print(flat_X.shape)
        self.assertTrue(flat_X.shape == (70000*784,))
    
    def test_get_plot_data_birch_BEE_fixed(self):
        data = get_od_data(0)
        modes = ['mean', 'max']
        ensembles = [2,4]
        trials = 3
        
        differing_T = False
        T = np.array([0.8,0.9])
        
        T_values, T_description = calculate_T_values(data.X, T_average_nn, differing_T, T=T)
        
        df = pd.DataFrame(columns=['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        df, M = get_plot_data_birch(data, T_values, T_description, modes, ensembles, trials, differing_T, df, False, False)
        
        length = len(modes)*len(ensembles)*trials*len(acc_metrics)*len(T)
        expected_En = list(np.sort(ensembles * len(modes)*len(T)*len(acc_metrics)*trials))
        expected_Mo = list(np.repeat(modes, len(acc_metrics)*trials *len(T))) *len(ensembles)
        expected_Th = list(np.repeat(T, len(acc_metrics)*trials ).astype(str)) *len(modes) *len(ensembles)
        expected_Me = acc_metrics*len(T) *len(modes) *len(ensembles) *trials
        
        self.assertTrue(length == len(df))
        self.assertTrue(np.all(expected_En == df['Ens size']))
        self.assertTrue(np.all(expected_Mo == df['Mode']))
        self.assertTrue(np.all(expected_Th == df['T']))
        self.assertTrue(np.all(expected_Me == df['Metric']))
    
    def test_partition(self):
        c = 2
        n = 5
        result = np.array(partition(c, n))
        correct = np.array([[1,4],[4,1],[2,3],[3,2]])
        self.assertEqual(result.shape, correct.shape)
        self.assertTrue(np.all(np.isin(result , correct)))
        for i in range(len(result)):
            self.assertEqual(np.sum(result[i]), n)
            
        c = 5
        n = 2
        result = partition(c, n)
        correct = []
        self.assertEqual(result, correct)
        
    def test_get_big_synthetic_data(self):
        n=200000
        data = get_big_synthetic_data(n)
        self.assertEqual(data.X.shape, (n,2))
        self.assertEqual(data.od_labels.shape, (n,))
        self.assertTrue(isinstance(data, Dataset))
        self.assertEqual(len(np.where(data.od_labels==1)[0]), 20)
        self.assertTrue(np.all(data.od_labels[n-20:] == [1]*20))
        self.assertTrue(np.all(data.od_labels[:n-20] == [0]*(n-20)))
        
    def test_get_od_data(self):
        data = get_od_data(0)
        self.assertTrue(isinstance(data, Dataset))
        self.assertEqual(data.X.shape, (500,2))
        self.assertEqual(data.od_labels.shape, (500,))
        self.assertEqual(len(np.where(data.od_labels==1)[0]), 4)
        self.assertTrue(data.labels is None)
        self.assertTrue(np.all(np.isin(data.od_labels, [0,1])))
        
        data = get_od_data(1)
        self.assertTrue(isinstance(data, Dataset))
        self.assertEqual(data.X.shape, (6000,784))
        self.assertEqual(data.od_labels.shape, (6000,))
        self.assertEqual(data.labels.shape, (6000,))
        self.assertTrue(np.all(np.isin(data.od_labels, [0,1])))
        self.assertTrue(np.all(np.isin(data.labels, [0,1,2,3,4,5,6,7,8,9])))
        self.assertEqual(len(np.where(data.od_labels==1)[0]), 60)
        
        data = get_od_data(1,  benign_labels=np.array([1,7]), malign_labels=np.array([0]))
        self.assertTrue(isinstance(data, Dataset))
        self.assertEqual(data.X.shape, (15000,784))
        self.assertEqual(data.od_labels.shape, (15000,))
        self.assertEqual(data.labels.shape, (15000,))
        self.assertTrue(np.all(np.isin(data.od_labels, [0,1])))
        self.assertTrue(np.all(np.isin(data.labels, [0,1,7])))
        self.assertEqual(len(np.where(data.od_labels==1)[0]), 150)
        
        data = get_od_data(1,  benign_labels=np.array([1,7]), malign_labels=np.array([0]), contamination_rate=0.4)
        self.assertTrue(isinstance(data, Dataset))
        self.assertEqual(data.X.shape, (17000,784))
        self.assertEqual(data.od_labels.shape, (17000,))
        self.assertEqual(data.labels.shape, (17000,))
        self.assertTrue(np.all(np.isin(data.od_labels, [0,1])))
        self.assertTrue(np.all(np.isin(data.labels, [0,1,7])))
        self.assertEqual(len(np.where(data.od_labels==1)[0]), 17000*0.4)
        
        n=10000
        data = get_od_data(2, sample_size=n)
        self.assertTrue(isinstance(data, Dataset))
        self.assertEqual(data.X.shape, (n,2))
        self.assertEqual(data.od_labels.shape, (n,))
        self.assertTrue(data.labels is None)
        self.assertTrue(np.all(np.isin(data.od_labels, [0,1])))
        self.assertEqual(len(np.where(data.od_labels==1)[0]), 20)
        
        
    def test_random_order(self):
        data = get_od_data(0)
        data_copy = Dataset(data.X, data.od_labels)
        random_order(data)
        # at least one anomaly has a different index
        self.assertFalse(np.all(np.isin(np.where(data.od_labels)[0], np.where(data_copy.od_labels))))
        
        u=data.X[np.where(data.od_labels)[0]]
        v=data_copy.X[np.where(data_copy.od_labels)[0]]
        # anomalies are the same
        self.assertTrue(np.all(np.isin(u,v)))
        
    def test_os_mean(self):
        y_score = np.array([0.1,0.5,0.1,0.2,0.7,0.8,0.1])
        m = 2
        correct = 0.75
        result = os_mean(y_score, m)
        self.assertEqual(result, correct)
    
        m = 5
        correct = (0.5+0.2+0.8+0.7+0.1)/m
        result = os_mean(y_score, m)
        self.assertEqual(result, correct)
    
        
    def test_evaluate_model_accuracy(self):
        data = get_od_data(0)
        M = Model(OneClassSVM(nu=0.01, gamma=0.2), 'OCSVM')
        Value = []
        num_metrics=3
        y_score = evaluate_model_accuracy(data, M, Value, num_metrics)
        
        self.assertEqual(len(Value), num_metrics)
        self.assertEqual(len(y_score),len(data.od_labels))
        self.assertEqual([M.AP[0], M.roc_auc[0], M.prec_at_n[0]] , Value[0:3])
        
        M = Model(BirchEnsemble(T_arr=[0.6]*10, mode='median'), 'BE15')
        Value = []
        num_metrics=5
        y_score = evaluate_model_accuracy(data, M, Value, num_metrics)
        
        self.assertEqual(len(Value), num_metrics)
        self.assertEqual(len(y_score),len(data.od_labels))
        self.assertEqual([M.AP[0], M.roc_auc[0], M.prec_at_n[0], M.OSTM, M.OSTM_ratio] , Value)
        self.assertTrue(np.all(y_score <=1.0))
        self.assertTrue(np.all(y_score >=0.0))
        
    
    def test_get_number_of_datapoints_to_show(self):
        od_labels = np.array([0,1,1,0,0,0,0,1])
        y_score = np.array([0.1,0.7,0.7,0.5,0.8,0.1,0.2,0.8])
        correct = 4
        result = get_number_of_datapoints_to_show(od_labels, y_score)
        self.assertEqual(correct, result)
    
        od_labels = np.array([0,1,1,0,1,0,0,1])
        y_score = np.array([0.1,0.7,0.7,0.5,0.8,0.1,0.2,0.8])
        correct = 4
        result = get_number_of_datapoints_to_show(od_labels, y_score)
        self.assertEqual(correct, result)
        
        od_labels = np.array([1,0,0,1,0,0,0,1])
        y_score = np.array([0.1,0.7,0.7,0.5,0.8,0.1,0.2,0.8])
        correct = len(od_labels)
        result = get_number_of_datapoints_to_show(od_labels, y_score)
        self.assertEqual(correct, result)
    
    
    def test_plot_detected_anomalies(self):
         #(data: Dataset, idx: np.ndarray, rows: int, columns: int, ax: np.array, caption: str
         data = get_od_data(0)
         M = Model(BirchEnsemble(T_arr=np.array([0.4]*10), mode='median'), name='Model')
         M.fit(data.X)
         y_score = M.score_samples(data.X)
         #idx = np.argsort(y_score)
         fig, ax = get_subplots(True, 1, 1, 'test plot detected anoms')
         plot_detected_anomalies(data, y_score, 0,0,ax,'detected')
         
         fig.tight_layout()
         plt.show()
        
    def test_get_plot_data_birch_BEM_fixed(self):
        data = get_od_data(1)
        modes = ["median"]
        e = [1,2]
        T = np.array([1500,1600])
        trials = 2
        differing_T = True
        
        T_description, T_values = calculate_T_values(data.X, T_average_nn, differing_T, T=T)
        
        df = pd.DataFrame(columns=['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        df, M = get_plot_data_birch(data, T_values, T_description, modes, e, trials, differing_T, df, False, False)
        
        expected_En = list(np.sort([e[0]] * len(modes)*2*len(acc_metrics)*trials)) + list(np.sort([e[1]] * len(modes)*1*len(acc_metrics)*trials))
        expected_Mo = list(np.repeat(modes, len(acc_metrics)*trials *3))
        expected_Th = list(np.repeat((np.arange(2)+1).astype(str), len(acc_metrics)*trials )) *len(modes) + list(np.repeat((np.arange(1)+1).astype(str), len(acc_metrics)*trials )) *len(modes)
        expected_Me = acc_metrics* 3*len(modes) *trials
        
        self.assertTrue(len(df) == trials*3*len(modes)*len(acc_metrics))
        self.assertTrue(np.all(df['Ens size'] == expected_En))
        self.assertTrue(np.all(df['Mode'] == expected_Mo))
        self.assertTrue(np.all(df['T'] == expected_Th))
        self.assertTrue(np.all(df['Metric'] == expected_Me))
    
    def test_get_plot_data_birch_BEE_heuristic(self):    
        data = get_od_data(0)
        modes = ["median"]
        differing_T = False
        e = [1,2,3]
        trials = 3
        T_description, T_values = calculate_T_values(data.X, T_average_nn, differing_T, n_heuristic=5)
        
        df = pd.DataFrame(columns=['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        df, M = get_plot_data_birch(data, T_values, T_description, modes, e, trials, differing_T, df, False, False)
        
        expected_En = list(np.sort(e * len(modes)*len(T_description)*len(acc_metrics)*trials))
        expected_Mo = list(np.repeat(modes, len(acc_metrics)*trials *len(T_description))) *len(e)
        expected_Th = list(np.repeat(T_description, len(acc_metrics)*trials )) *len(modes) *len(e)
        expected_Me = acc_metrics* len(T_description)*len(modes) *len(e) *trials
        
        self.assertTrue(len(df) == len(e)*trials*len(T_description)*len(modes)*len(acc_metrics))
        self.assertTrue(np.all(df['Ens size'] == expected_En))
        self.assertTrue(np.all(df['Mode'] == expected_Mo))
        self.assertTrue(np.all(df['T'] == expected_Th))
        self.assertTrue(np.all(df['Metric'] == expected_Me))

    def test_get_plot_data_birch_BEM_heuristic(self):    
        data = get_od_data(0)
        modes = ["median"]
        e = [1,2,3]
        differing_T = True
        trials = 3
        configurations = 5
        T_description, T_values = calculate_T_values(data.X, T_average_nn, differing_T)
       
        df = pd.DataFrame(columns=['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        df, M = get_plot_data_birch(data, T_values, T_description, modes, e, trials, differing_T, df, False, False)
        expected_En = list(np.sort(e * len(modes)*configurations*len(acc_metrics)*trials))
        expected_Mo = list(np.repeat(modes, len(acc_metrics)*trials *configurations)) *len(e)
        expected_Th = list(np.repeat((np.arange(configurations)+1).astype(str), len(acc_metrics)*trials )) *len(modes) *len(e)
        expected_Me = acc_metrics* configurations*len(modes) *len(e) *trials
        
        self.assertTrue(len(df) == len(e)*trials*configurations*len(modes)*len(acc_metrics))
        self.assertTrue(np.all(df['Ens size'] == expected_En))
        self.assertTrue(np.all(df['Mode'] == expected_Mo))
        self.assertTrue(np.all(df['T'] == expected_Th))
        self.assertTrue(np.all(df['Metric'] == expected_Me))
        
    def test_calculate_T_values_mnist(self):
        data = get_od_data(1)
        
        # not heuristic
        T = np.array([1500,1600,1700])
        n_heuristic = 3
        configurations = 5
        heuristic = T_average_nn
        # BEM
        T_description, T_values = calculate_T_values(data.X, heuristic, True, T=T, n_heuristic=n_heuristic)
        self.assertTrue(len(T_description) == configurations)
        self.assertTrue(np.all(np.isin(T_values, T)))
        self.assertTrue(np.all(T_values == T))
        self.assertTrue(np.all(T_description==(np.arange(configurations)+1).astype(str) ))
        
        #BEE
        T_description, T_values = calculate_T_values(data.X, heuristic, False, T=T, n_heuristic=n_heuristic)
        self.assertTrue(len(T_description) == len(T))
        self.assertTrue(np.all(T_values==T))
        self.assertTrue(np.all(T_description==T ))
        
        # heuristic
        #BEM
        T_description, T_values = calculate_T_values(data.X, heuristic, True, n_heuristic=n_heuristic)
        self.assertTrue(np.all(T_values < 2300) and np.all(T_values>0))
        self.assertTrue(len(T_values) == n_heuristic)
        self.assertTrue(np.all(T_description== (np.arange(configurations)+1).astype(str) ))
        
        #BEE
        T_description, T_values = calculate_T_values(data.X, heuristic, False, n_heuristic=n_heuristic)
        self.assertTrue(np.all(T_values < 2300) and np.all(T_values>0))
        self.assertTrue(len(T_values) == n_heuristic)
        self.assertTrue(len(T_description) == n_heuristic)
        self.assertTrue(np.all(T_description== np.array(['T_av', '/2', '/4'])))
        
    def test_calculate_T_values_synthetic(self):
        data = get_od_data(0)
        
        T = np.array([0.7,0.8,0.9])
        configurations = 5
        n_heuristic = 4
        heuristic = T_average
        
        # not heuristic
        # BEM
        T_description, T_values = calculate_T_values(data.X, heuristic, True, T=T, n_heuristic=n_heuristic)
        self.assertTrue(len(T_description) == configurations)
        self.assertTrue(np.all(np.isin(T_values, T)))
        self.assertTrue(np.all(T_values == T))
        self.assertTrue(np.all(T_description==(np.arange(configurations)+1).astype(str) ))
        
        # BEE
        T_description, T_values = calculate_T_values(data.X, heuristic, False, T=T, n_heuristic=n_heuristic)
        self.assertTrue(len(T_description) == len(T))
        self.assertTrue(np.all(T_values==T))
        self.assertTrue(np.all(T_description==T ))
        
        # heuristic
        # BEM
        T_description, T_values = calculate_T_values(data.X, heuristic, True, n_heuristic=n_heuristic)
        self.assertTrue(np.all(T_values < 2300) and np.all(T_values>0))
        self.assertTrue(len(T_values) == n_heuristic)
        self.assertTrue(np.all(T_description== (np.arange(configurations)+1).astype(str) ))
        
        # BEE
        T_description, T_values = calculate_T_values(data.X, heuristic, False, n_heuristic=n_heuristic)
        self.assertTrue(np.all(T_values < 2300) and np.all(T_values>0))
        self.assertTrue(len(T_values) == n_heuristic)
        self.assertTrue(len(T_description) == n_heuristic)
        self.assertTrue(np.all(T_description== np.array(['T_av', '/2', '/4', '/8'])))
        
    def test_get_ensemble_T_values(self):
        e = 5
        T_values = np.array([0.5,0.6,0.7,0.8,0.9])
        
        differing_T = False
        result = get_ensemble_T_values(T_values, e, differing_T)
        self.assertTrue(result.shape == (len(T_values),e))
        self.assertTrue(np.all( result == np.repeat(T_values, e).reshape((len(T_values),e)) ))
        
        # only one configuration possible for 5 T values and e=5
        differing_T = True
        result = get_ensemble_T_values(T_values, e, differing_T)
        self.assertTrue(result.shape == (1,e))
        self.assertTrue(np.all(np.isin( result, T_values )))
        
        e=4
        configurations=5 # always configurations=5 in get_ensemble_T_values
        result = get_ensemble_T_values(T_values, e, differing_T)
        self.assertTrue(result.shape == (configurations,e))
        self.assertTrue(np.all(np.isin( result, T_values )))
    
    def test_get_ensemble_T_values_bem(self):
        T_values = np.array([0.4,1.2])
        e = 15
        differing_T = True
        configurations = 5
        
        t = get_ensemble_T_values(T_values, e, differing_T)
        t_correct = np.array([[0.4]*7+[1.2]*8,
                             #[0.4]*8+[1.2]*7,
                             [0.4]*3+[1.2]*12,
                             [0.4]*12+[1.2]*3,
                             [0.4]*6+[1.2]*9,
                             [0.4]*9+[1.2]*6])
                             
        self.assertTrue(t.shape, (e,configurations))
        self.assertTrue(np.all(t_correct== np.array(t)))
        
        e = 10
        t = get_ensemble_T_values(T_values, e, differing_T)
        t_correct = np.array([[0.4]*5+[1.2]*5,
                             [0.4]*2+[1.2]*8,
                             [0.4]*8+[1.2]*2,
                             [0.4]*4+[1.2]*6,
                             [0.4]*6+[1.2]*4])
        
        self.assertTrue(t.shape, (e,configurations))
        self.assertTrue(np.all(t_correct== np.array(t)))
        
        e=1
        t = get_ensemble_T_values(T_values, e, differing_T)
        t_correct = np.array([[0.4], [1.2]])
        self.assertTrue(t.shape, (e,len(T_values)))
        self.assertTrue(np.all(np.isin(t_correct, np.array(t))))
        
        T_values = np.array([0.4,0.8,1.2])
        e = 1
        t = get_ensemble_T_values(T_values, e, differing_T)
        t_correct = np.array([[0.4], [0.8], [1.2]])
        self.assertTrue(t.shape, (e,len(T_values)))
        self.assertTrue(np.all(np.isin(t_correct, np.array(t))))
        
        e = 10
        t = get_ensemble_T_values(T_values, e, differing_T)
        self.assertTrue(t.shape, (e,len(T_values)))
        self.assertTrue(np.all(np.isin(t_correct, np.array(t))))
        
        
    def test_get_histogram_values(self):
        data = get_od_data(0)
        fig, ax = get_subplots(True, 1, 3, 'Test Histogram plot')
        
        T = 0.7
        BE = BirchEnsemble(np.array([T]), mode='median')
        BE.fit(data.X)
        cs = BE.cluster_score
        
        c, b = get_histogram_values(cs, str(T),0,0,ax)
        self.assertTrue(int(max(cs)-min(cs))+1 == len(b))
        self.assertTrue(np.sum(c) == 500)
        
        T = 5
        BE = BirchEnsemble(np.array([T]), mode='median')
        BE.fit(data.X)
        cs = BE.cluster_score
        
        c, b = get_histogram_values(cs, str(T),0,1,ax)
        count, labels = np.unique(cs, return_counts=True)
        self.assertTrue(np.sum(c) == len(cs))
                
        data = get_od_data(1)
        T = 1600
        BE = BirchEnsemble(np.array([T]), mode='median')
        BE.fit(data.X)
        cs = BE.cluster_score
        
        c_a, b_a = get_histogram_values(cs, str(T),0,2,ax)
        u_a, count_a = np.unique(cs, return_counts=True)
        self.assertTrue(np.sum(c_a) == len(cs))
        fig.tight_layout()
        plt.show()
        
        fig, axes = get_subplots(True, 1,1, 'Test single subplot')
        c_b, b_b = get_histogram_values(cs, str(T),0,0,axes)
        u_b, count_b = np.unique(cs, return_counts=True)
        self.assertTrue(np.sum(c_b) == len(cs))
        fig.tight_layout()
        plt.show()
        
        self.assertTrue(np.all(c_a==c_b))
        
     
    def test_load_mnist(self):
        data = load_mnist()
        contamination_rate = 0.001
        benign_labels = np.array([1,2,3,4,5,6,7,8,9])
        malign_labels = np.array([0])
        size = 63000
        
        self.assertEqual(data.X.shape, (70000, 28, 28))
        self.assertTrue(np.all(np.isin(data.labels, np.arange(0,10))))
        self.assertTrue(len(data.labels) == 70000)
        
        self.assertTrue(data.od_labels is None)
        
        od_data = data.makeODDataset(benign_labels, malign_labels, contamination_rate)
        self.assertTrue(len(od_data.X)==size)
        self.assertTrue(np.all(np.isin(od_data.od_labels, np.array([0,1]))))
        self.assertEqual(np.sum(od_data.od_labels), int(size*contamination_rate))
        
        self.assertTrue(np.all(np.isin(od_data.labels, np.arange(10))))
        
        
    def test_getSyntheticData(self):
        sampleSize = 100
        data = get_small_synthetic_data(sampleSize)
        self.assertEqual(data.X.shape, (sampleSize,2))
        self.assertEqual(len(data.od_labels), len(data.X))
        
    def test_get_complex_data_mnist_0_123456789(self):
        data = load_mnist()
        
        benign_labels = np.array([0])
        malign_labels = np.array([1,2,3,4,5,6,7,8,9])
        contamination = 0.01
        size = 6000
        
        benign = len(np.where(np.isin(data.labels, benign_labels))[0])
        malign = len(np.where(np.isin(data.labels, malign_labels))[0])
        
        sample_size, num_anoms, num_normal = data.getBigSampleSize(benign, malign, contamination)
        self.assertTrue(sample_size*contamination == num_anoms)
        self.assertTrue(sample_size == num_anoms + num_normal)
        self.assertTrue(sample_size == size)
        
        # get correct malign-benign-ratio
        X, labels = data.get_data(benign_labels, malign_labels, contamination)
        self.assertTrue(X.shape[0] == size)
        self.assertTrue(len(labels) == size)
        self.assertTrue(size*contamination == len(np.where(np.isin(labels, malign_labels))[0]))
        self.assertTrue(len(labels)-(size*contamination) == len(np.where(np.isin(labels, benign_labels))[0]))
        
        correct_od = np.array( [0]*num_normal+ [1]*num_anoms )
        compl_data = get_complex_data(1, benign_labels, malign_labels, contamination)
        
        self.assertEqual(len(compl_data.X), sample_size)
        self.assertEqual(len(correct_od), len(compl_data.od_labels))
        self.assertTrue(np.all( compl_data.od_labels[np.where(correct_od == 0)] == 0 ))
        self.assertTrue(np.all( correct_od[np.where(compl_data.od_labels == 1)] == 1 ))
        self.assertTrue(np.all(np.isin(compl_data.labels[np.where(compl_data.od_labels==0)], benign_labels)))
        self.assertTrue(np.all(np.isin(compl_data.labels[np.where(compl_data.od_labels==1)], malign_labels)))
        
        od_data = data.makeODDataset(benign_labels, malign_labels, contamination)
        self.assertTrue(len(od_data.X)==size)
        self.assertTrue(np.all(np.isin(od_data.od_labels, np.array([0,1]))))
        self.assertEqual(np.sum(od_data.od_labels), int(size*contamination))
        self.assertTrue(np.all(np.isin(od_data.labels[np.where(od_data.od_labels==0)], benign_labels)))
        self.assertTrue(np.all(np.isin(od_data.labels[np.where(od_data.od_labels==1)], malign_labels)))
        
    def test_get_complex_data_mnist_08_1(self):
        data = load_mnist()
        benign_labels = np.array([0,8])
        malign_labels = np.array([1])
        contamination = 0.001
        size = 13000
        
        benign = len(np.where(np.isin(data.labels, benign_labels))[0])
        malign = len(np.where(np.isin(data.labels, malign_labels))[0])
        sample_size, num_anoms, num_normal = data.getBigSampleSize(benign, malign, contamination)
        self.assertTrue(sample_size*contamination == num_anoms)
        self.assertTrue(sample_size == num_anoms + num_normal)
        self.assertTrue(sample_size == size)
        
        X, labels = data.get_data(benign_labels, malign_labels, contamination)
        self.assertTrue(X.shape[0] == sample_size)
        self.assertTrue(len(labels) == sample_size)
        self.assertTrue(num_anoms == len(np.where(np.isin(labels, malign_labels))[0]))
        self.assertTrue(len(labels)-num_anoms == len(np.where(np.isin(labels, benign_labels))[0]))
        
        correct_od = np.array( [0]*num_normal+ [1]*num_anoms )
        data = get_complex_data(1, benign_labels, malign_labels, contamination)
        
        self.assertEqual(len(data.X), sample_size)
        self.assertEqual(len(correct_od), len(data.od_labels))
        self.assertTrue(np.all( data.od_labels[np.where(correct_od == 0)] == 0 ))
        self.assertTrue(np.all( correct_od[np.where(data.od_labels == 1)] == 1 ))
        self.assertTrue(np.all(np.isin(data.labels[np.where(data.od_labels==0)], benign_labels)))
        self.assertTrue(np.all(np.isin(data.labels[np.where(data.od_labels==1)], malign_labels)))
        
        od_data = data.makeODDataset(benign_labels, malign_labels, contamination)
        self.assertTrue(len(od_data.X)==size)
        self.assertTrue(np.all(np.isin(od_data.od_labels, np.array([0,1]))))
        self.assertEqual(np.sum(od_data.od_labels), int(size*contamination))
        self.assertTrue(np.all(np.isin(od_data.labels[np.where(od_data.od_labels==0)], benign_labels)))
        self.assertTrue(np.all(np.isin(od_data.labels[np.where(od_data.od_labels==1)], malign_labels)))
        
    def test_get_heuristic_values(self):
        data = get_od_data(0)
        heuristic = T_average
        differing_T = False
        n_heuristic = 5
        n = 100
        
        d_result, t_result = get_heuristic_values(data.X, heuristic, differing_T, n_heuristic, n)
        correct = []
        aprox = heuristic(data.X, n)
        for i in range(n_heuristic):
            correct += [t_result[0]/2**i]
        
        self.assertEqual(len(t_result), n_heuristic)
        self.assertTrue(np.all(d_result== np.array(['T_av', '/2', '/4', '/8','/16'])))
        self.assertTrue(np.all(correct==t_result))
        self.assertTrue(t_result[0] <= aprox +5)
        self.assertTrue(t_result[0] >= aprox -5)
        
        differing_T = True
        
        d_result, t_result = get_heuristic_values(data.X, heuristic, differing_T, n_heuristic, n)
        correct = []
        aprox = heuristic(data.X, n)
        for i in range(n_heuristic):
            correct += [t_result[0]/2**i]
        
        self.assertEqual(len(t_result), n_heuristic)
        self.assertTrue(d_result is None)
        self.assertTrue(np.all(correct==t_result))
        self.assertTrue(t_result[0] <= aprox +5)
        self.assertTrue(t_result[0] >= aprox -5)
        
        
    def test_calcBirch(self):
        t = 0.7
        BE = BirchEnsemble(np.array([t,t]), 'median')
        X = np.array([[1.2, 1.4], [1.3, 1.2], [1.1, 1.2], [2.1, 2.3]])
        brc = BE.calcBirch(t, X)
        self.assertEqual(brc.get_params()['threshold'], 0.7)
        self.assertEqual(brc.get_params()['n_clusters'], None)
    
    def test_getClusterSizesOfPoint_1(self):
        X = np.array([[1.2, 1.4], [1.3, 1.2], [1.1, 1.2], [2.1, 2.3]])
        ind = np.array([0,3,1,2])
        labels = np.array([0, 1, 0, 0])
        clusterLabels = np.array([0,1])
        
        BE = BirchEnsemble(np.array([1.0,1.0]), 'mean')
        result = BE.getClusterSizesOfPoints(ind, labels, clusterLabels)
        correct = np.array([3,3,3,1])
        self.assertTrue(np.all(correct == result))
        
        X = np.array([[1.1,1.0], [-5,2], [3,-2], [7,1], [-2,-1]])
        ind = np.array([2,0,4,3,1])
        labels = np.array([0,1,0,0,1])
        clusterLabels = np.array([0,1])
        
        BE = BirchEnsemble(np.array([1.0,1.0]), 'mean')
        result = BE.getClusterSizesOfPoints(ind, labels, clusterLabels)
        correct = np.array([2,2,3,3,3])
        self.assertTrue(np.all(correct == result))
        
    def test_getClusterSizesOfPoint_oneEmptyCluster(self):
        labels = np.array([1,2,2,3,4,3,3,1,0,2,6,4,7,2,3,3,3]) # cluster 5 empty
        clusterLabels = np.array([0,1,2,3,4,5,6,7])
        ind = np.arange(len(labels))
        
        BE = BirchEnsemble(np.array([1.0,1.0]), 'mean')
        result = BE.getClusterSizesOfPoints(ind, labels, clusterLabels)
        correct = np.array([2,4,4,6,2,6,6,2,1,4,1,2,1,4,6,6,6])
        self.assertTrue(np.all(result == correct))
    
    def test_getClusterSizesOfPoint_twoEmptyClusters(self):
        labels = np.array([2,2,3,4,3,8,3,0,8,2,6,4,7,2,3,3,3]) # cluster 1 & 5 empty
        clusterLabels = np.array([0,1,2,3,4,5,6,7,8])
        ind = np.arange(len(labels))
        
        BE = BirchEnsemble(np.array([1.0,1.0]), 'mean')
        result = BE.getClusterSizesOfPoints(ind, labels, clusterLabels)
        correct = np.array([4,4,6,2,6,2,6,1,2,4,1,2,1,4,6,6,6])
        self.assertTrue(np.all(result == correct))
    
    def test_getClusterSizesInTree(self):
        t = 0.2
        X = np.array([[1.2, 1.4], [1.3, 1.2], [1.1, 1.2], [2.1, 2.3]])
        ind = np.random.permutation(X.shape[0])
    
        mode = "max"
        BE = BirchEnsemble(np.array([t]),mode)
        cspp = BE.getClusterSizesInTree(X, ind, 0)
        self.assertTrue(np.all(cspp != np.zeros(4)))
        self.assertTrue(np.all(np.in1d(cspp, np.array([1,2,3,4]))))
        
        mode = "mean"
        BE = BirchEnsemble(np.array([t]),mode)
        cspp = BE.getClusterSizesInTree(X,ind, 0)
        self.assertTrue(np.all(cspp != np.zeros(4)))
        self.assertTrue(np.all(np.in1d(cspp, np.array([1,2,3,4]))))
        
        mode = "median"
        BE = BirchEnsemble(np.array([t]),mode)
        cspp = BE.getClusterSizesInTree(X,ind, 0)
        self.assertTrue(np.all(np.in1d(cspp, np.array([1,2,3,4]))))
        
    
    def test_getClusterSizesInTree_advanced(self):
        BE = BirchEnsemble(np.array([10,3]), 'median')
        X1 = np.array([[1.1,1.0], [-5,2], [3,-2], [7,1], [-2,-1]])
        cs = BE.getClusterSizesInTree(X1, np.array([2,0,4,3,1]), 1)
        cs_correct = np.array([3.0,1.,3.,1.,3.])
        self.assertTrue(np.all(cs == cs_correct))
        
    def test_calculate_BEE(self):
        X = np.array([[1.2, 1.4], [1.15,1.3], [1.25,1.3], [1.3,1.4], [1.2,1.2], [1.3, 1.2], [1.1, 1.2], [2.1, 2.3]])
        ind = np.random.permutation(X.shape[0])
        T = np.array([0.3, 0.3])
        
        mode = 'median'
        BE = BirchEnsemble(T, mode)
        cspp = BE.getClusterSizesInTree(X, ind, 0)
        correct = np.array([7,7,7,7,7,7,7,1])
        self.assertTrue(np.all(cspp==correct))
        
        BE.fit(X)
        y_score = BE.score_samples(X)
        self.assertTrue(np.all(y_score == BE.getOutlierScore()))
        
    def test_getClusterSizesInTree_and_calculate_BEM(self):
        X = np.array([[1.2, 1.4], [1.15,1.3], [1.25,1.3], [1.3,1.4], [1.2,1.2], [1.3, 1.2], [1.1, 1.2], [2.1, 2.3]])
        ind = np.random.permutation(X.shape[0])
        T = np.array([0.3, 1.5])
        
        mode = 'max'
        BE = BirchEnsemble(T, mode)
        ind = np.random.permutation(X.shape[0])
        cspp = BE.getClusterSizesInTree(X, ind, 0)
        correct =  np.array([7,7,7,7,7,7,7,1])
        self.assertTrue(np.all(cspp==correct))
        
        ind = np.random.permutation(X.shape[0])
        cspp = BE.getClusterSizesInTree(X,ind, 1)
        correct = np.array([8,8,8,8,8,8,8,8])
        self.assertTrue(np.all(cspp==correct))
        
        BE.fit(X)
        y_score = BE.score_samples(X)
        correct = BE.getOutlierScore()
        self.assertTrue(np.all(correct==y_score))
        
        mode = 'mean'
        BE = BirchEnsemble(T, mode)
        BE.fit(X)
        y_score = BE.score_samples(X)
        correct = np.mean([[7,7,7,7,7,7,7,1], [8,8,8,8,8,8,8,8]], axis=0)
        correct_score = BE.getOutlierScore()
        self.assertTrue( np.all(correct_score==y_score) )
        
        mode = 'median'
        BE = BirchEnsemble(T, mode)
        BE.fit(X)
        y_score = BE.score_samples(X)
        correct = np.median([[7,7,7,7,7,7,7,1], [8,8,8,8,8,8,8,8]], axis=0)
        correct_score = BE.getOutlierScore()
        self.assertTrue( np.all(correct_score==y_score) )
       
    def test_fit_ensemble_synthetic_data(self):
        data = get_od_data(0)
        mode = 'median'
        e = 4
        T = [0.7]*e
        BE = BirchEnsemble(T, mode)
        BE.fit(data.X)
        y_score = BE.score_samples(data.X)
        self.assertEqual(len(y_score), len(data.od_labels))
        self.assertTrue((y_score >= -1.0).all() and (y_score < 0).all())
        
    def test_fit_ensemble_mnist_data(self):
        benign = np.random.choice(np.arange(10), 3)
        malign = np.where(np.invert(np.isin(np.arange(10), benign)))
        data = get_od_data(1, benign, malign)
        mode = 'median'
        e = 4
        T = [1700]*e
        BE = BirchEnsemble(T, mode)
        BE.fit(data.X)
        y_score = BE.score_samples(data.X)
        self.assertEqual(len(y_score), len(data.od_labels))
        self.assertTrue((y_score >= -1.0).all() and (y_score < 0).all())
        
    def test_evaluate_and_plot(self):
        data = get_od_data(0)
        mode = 'median'
        e = 4
        num_metrics = len(acc_metrics)
        T = [0.8]*e
        V = []
        M = Model(BirchEnsemble(T, mode), 'BE')
        trials = 1
        fig, ax = get_subplots(True, 1, 2, 'Test evaluate_and_plot')
        
        evaluate_and_plot(data, M, V, trials, 0, 0, ax, True, num_metrics)
        self.assertEqual(len(V), num_metrics)
        
        V = []
        trials = 3
        evaluate_and_plot(data, M, V, trials, 0, 1, ax, True, num_metrics)
        self.assertEqual(len(V), len(acc_metrics)*trials)
        
        fig.tight_layout()
        plt.show()
    
    def test_evaluate_model_acc_multiple_trials(self):
        data = get_od_data(0)
        M = Model(BirchEnsemble(T_arr=np.array([0.9,0.9,0.9]), mode='median'), 'BE3 0.9 md')
        trials= 3
        m = int(np.sum(data.od_labels))
        Value = []
        num_metrics=len(acc_metrics)
        y_score = evaluate_model_acc_multiple_trials(data, M, trials, Value, num_metrics)
    
        self.assertEqual(len(Value), trials*num_metrics)
        self.assertEqual(Value[trials*num_metrics-1], os_mean(y_score, m)/os_mean(y_score, 2*m))
    
    def test_getOutlierScore(self):
        cspp = np.array([3,2,3,3,2,1])
        correct = -1 / cspp
        BE = BirchEnsemble(np.array([1.0,1.0]), 'mean')
        BE.cluster_score = cspp
        result = BE.getOutlierScore()
        self.assertTrue(np.all(correct == result))


    def test_precision_at_n_0_Of_2(self):
        y_score = np.array([0.2,0.5,0.7,0.1,0.9,0.3,0.8,0.1,0.1,0.5])
        trueAnoms = np.array([0,1,1,0,0,0,0,0,0,0])
        n = 2
        correct = 0.0 / n
        result = precision_at_n(trueAnoms, y_score)
        self.assertTrue(np.isclose(correct, result))
        
    def test_precision_at_n_1_Of_2(self):
        y_score = np.array([0.2,0.5,0.7,0.1,0.9,0.3,0.8,0.1,0.1,0.5])
        trueAnoms = np.array([0,0,1,0,1,0,0,0,0,0])
        n = 2
        correct = 1.0 / n
        result = precision_at_n(trueAnoms, y_score)
        self.assertTrue(np.isclose(correct, result))
    
    def test_precision_at_n_2_Of_2(self):
        y_score = np.array([0.2,0.5,0.7,0.1,0.9,0.3,0.8,0.1,0.1,0.5])
        trueAnoms = np.array([0,0,0,0,1,0,1,0,0,0])
        n = 2
        correct = 2.0 / n
        result = precision_at_n(trueAnoms, y_score)
        self.assertTrue(np.isclose(correct, result))
     
           
    def test_get_accuracy_E4_Med_BEE(self):
        data = get_od_data(0)
        mode = 'median'
        num_metrics = len(acc_metrics)
        m = int(np.sum(data.od_labels))
        e = 4
        
        T = np.repeat(np.array([0.8]), e)
        BE = BirchEnsemble(T, mode)
        BE.fit(data.X)
        y_score = BE.score_samples(data.X)
        
        correct = [metrics.average_precision_score(data.od_labels, y_score), metrics.roc_auc_score(data.od_labels, y_score), precision_at_n(data.od_labels, y_score), os_mean(y_score, m), os_mean(y_score, m)/os_mean(y_score, 2*m)]
        result = get_accuracy(data.od_labels, y_score, num_metrics)
        self.assertTrue(np.all(correct == result))
    
    def test_get_accuracy_E2_Max_BME(self):
        data = get_od_data(0)
        mode = 'max'
        num_metrics = len(acc_metrics)
        m = int(np.sum(data.od_labels))
        T = np.array([0.8, 0.7])
        BE = BirchEnsemble(T, mode)
        BE.fit(data.X)
        y_score = BE.score_samples(data.X)
        
        correct = [metrics.average_precision_score(data.od_labels, y_score), metrics.roc_auc_score(data.od_labels, y_score), precision_at_n(data.od_labels, y_score), os_mean(y_score, m), os_mean(y_score, m)/os_mean(y_score, 2*m)]
        result = get_accuracy(data.od_labels, y_score, num_metrics) 
        self.assertTrue(np.all(correct == result ))
     
    def test_get_accuracy_3metrics(self):
        data = get_od_data(0)
        mode = 'mean'
        T = np.array([0.2])
        BE = BirchEnsemble(T, mode)
        BE.fit(data.X)
        y_score = BE.score_samples(data.X)
        
        correct = [metrics.average_precision_score(data.od_labels, y_score), metrics.roc_auc_score(data.od_labels, y_score), precision_at_n(data.od_labels, y_score)]
        result = get_accuracy(data.od_labels, y_score) 
        self.assertTrue(np.all(correct == result ))
    
    def test_evaluate_model(self):
        df = pd.DataFrame(columns = ['Method','Metric', 'Value'])
        data = get_od_data(0)
        M = Model(BirchEnsemble(np.array([0.8,0.8]), 'median'), 'BE')
        trials = 3
        
        fig, ax = get_subplots(False, 1,2, 'test evaluate Model')
        params = [M.name]
        
        self.assertTrue(isinstance(M.od_method, BirchEnsemble))
        df = evaluate_model(df, data, M, trials, 0, 0, params, ax, False)
        self.assertTrue(len(df) == trials*len(acc_metrics))
        self.assertTrue(np.all(df['Metric']==acc_metrics*trials))
        self.assertTrue(np.all(df['Method']==np.array(['BE']*trials*len(acc_metrics))))
        self.assertTrue(isinstance(M.od_method, BirchEnsemble))
        
        df = pd.DataFrame(columns = ['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        params = [M.od_method.e, M.od_method.mode, str(M.od_method.T_arr[0])]
        df = evaluate_model(df, data, M, trials, 0, 1, params, ax, False)
        self.assertTrue(len(df) == trials*len(acc_metrics))
        self.assertTrue(np.all(df['Metric']==acc_metrics*trials))
        self.assertTrue(np.all(df['T']==np.array(['0.8']*trials*len(acc_metrics))))
        self.assertTrue(np.all(df['Mode']==np.array(['median']*trials*len(acc_metrics))))
        self.assertTrue(isinstance(M.od_method, BirchEnsemble))
        
    def test_add_row_to_df(self):
        df = pd.DataFrame(columns=['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        params = [2, 'mean', '0.8', 'p@n', 0.712]
        
        df = add_row_to_df(df, params)
        df_corr = pd.DataFrame({'Ens size': [2], 'Mode': 'mean', 'T': ['0.8'], 'Metric': ['p@n'], 'Value': [0.712]})
        for i in range(len(df.columns)):
            self.assertTrue(np.all(list(df[df.columns[i]]) == list(df_corr[df_corr.columns[i]])))
        
        params = [3, 'max', '0.7', 'AP', 0.5]
        df = add_row_to_df(df, params)
        
        df_corr = df_corr.append({'Ens size': 3, 'Mode': 'max', 'T': '0.7', 'Metric': 'AP', 'Value': 0.5}, ignore_index=True)
        for i in range(len(df.columns)):
            self.assertTrue(np.all(list(df[df.columns[i]]) == list(df_corr[df_corr.columns[i]])))
        
    def test_update_df_3metrics(self):
        e = 2
        T = 0.8
        trials = 2
        modes = ['mean', 'max']
        V = [[0.123,0.45,0.5,0.22,0.66,0.75],[ 0.6,0.1,0.7,0.5,0.2,1.0]]
        num_metrics = 3
        
        df_corr = pd.DataFrame(columns = ['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        df = pd.DataFrame(columns = ['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        
        for mode in modes:
            params = [int(e), mode, str(T)]
            df = update_df(df, trials, V[modes.index(mode)], params, num_metrics)
            for i in range(trials):
                for j in range(num_metrics):
                    df_corr = df_corr.append({'Ens size': e, 'Mode': mode, 'T': str(T), 'Metric': acc_metrics[j], 'Value': V[modes.index(mode)][i*num_metrics+j]}, ignore_index=True)
        
        self.assertTrue(len(df) == len(df_corr))
        for i in range(len(df.columns)):
            self.assertTrue(np.all(list(df[df.columns[i]]) == list(df_corr[df_corr.columns[i]])))
    
    def test_update_df_4metrics(self):
        e = 2
        T = 0.8
        trials = 2
        num_metrics = len(acc_metrics)
        modes = ['mean', 'max']
        V = [[0.123,0.45,0.5,0.7,0.22,0.66,0.75,0.7,0.1,0.5],[ 0.6,0.1,0.75,0.8,0.5,0.2,1.0,0.8,0.5,0.4]]
        
        df_corr = pd.DataFrame(columns = ['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        df = pd.DataFrame(columns = ['Ens size', 'Mode', 'T', 'Metric', 'Value'])
        
        for mode in modes:
            params = [int(e), mode, str(T)]
            df = update_df(df, trials, V[modes.index(mode)], params, num_metrics)
            for i in range(trials):
                for j in range(len(acc_metrics)):
                    df_corr = df_corr.append({'Ens size': e, 'Mode': mode, 'T': str(T), 'Metric': acc_metrics[j], 'Value': V[modes.index(mode)][i*len(acc_metrics)+j]}, ignore_index=True)
        
        self.assertTrue(len(df) == len(df_corr))
        for i in range(len(df.columns)):
            self.assertTrue(np.all(list(df[df.columns[i]]) == list(df_corr[df_corr.columns[i]])))
    
    def test_get_birch_models(self):
        modes = ['mean', 'median']
        ensembles = [1,3,5]
        data = get_od_data(1)
        
        Models = get_birch_models(get_birch_configs(data, modes=modes, ensembles=ensembles, differing_T=False))
        
        self.assertTrue(len(Models) == len(modes)*len(ensembles)*5)
        self.assertTrue(all(isinstance(M.od_method), BirchEnsemble) for M in Models)
        
        modes_models = [M.od_method.mode for M in Models]
        for mode in modes:
            self.assertTrue(modes_models.count(mode) == len(ensembles)*5)
        
        ensembles_models = [M.od_method.e for M in Models]
        T_models = [len(M.od_method.T_arr) for M in Models]
        self.assertTrue(np.all(np.isin(T_models, ensembles)))
        for e in ensembles:
            self.assertTrue(ensembles_models.count(e) == len(modes)*5)
            self.assertTrue(T_models.count(e) == len(modes)*5)
        
        T_models = np.array([M.od_method.T_arr for M in Models])
        for t in T_models:
            self.assertTrue(np.all(t <= 3000) and np.all(t >= 90))
      
    def test_T_average_nn(self):
        data = np.reshape(np.arange(8), (4,2))
        correct = 2.8284271247461903
        result = T_average_nn(data, 4)
        self.assertTrue(np.isclose(result, correct))
        
        result = T_average_nn(data, 3)
        print(result)
        correct2 = 3.771236166328254
        self.assertTrue(np.isclose(result, correct) or np.isclose(result, correct2))
        
    def test_T_average(self):
        data = np.reshape(np.arange(8), (4,2))
        correct = 4.714045207910317
        result = T_average(data, 4)
        self.assertTrue(np.isclose(result, correct))
        
        result = T_average(data, 3)
        correct1 = 5.65685424949238
        correct2 = 3.771236166328254
        self.assertTrue(np.isclose(result, correct1) or np.isclose(result, correct2))
        
    def test_get_top_anoms(self):
        od_labels = np.array([0,0,1,0,0,1,0,1,1,0,1])
        y_score = np.array([0.62,0.11,0.56,0.54,0.45,0.41,0.22,0.48,0.72,0.15,0.66])
        n = np.sum(od_labels)
        result = get_top_anoms(od_labels, y_score, n)
        correct = np.array([1,1,0,1,0])
        self.assertTrue(np.all(result==correct))
        
if __name__ == '__main__':
    unittest.main()