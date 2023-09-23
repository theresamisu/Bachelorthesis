# Bachelor Thesis

This git contains the source code and data for my bachelor thesis

the folder "data" contains a subfolder for each dataset.
These contain the results as .csv files.
The notebook may help visualizing the results.

## main.py
contains functions for comparing the different BIRCH ensemble configurations and comparing different Models in general and examples for the usage.

A configuration corresponds to one parameter setting with size e, an array of T values and a mode (mean, max, median). 

#### compare_birch_configs:
This function can be used to evaluate several BIRCH ensemble configurations multiple times.
It returns a dataframe with the results and a list of Models. Each configuration is represented by one Model.
The dataframe contains ensemble size, description of T values (BEE->T value except if heuristic->T_av, /2..., BEM->number), mode, metric results

1. mandatory arguments: 
	- data: Dataset instance
	- T_values: list of T values to compare
	- T description: list, BEE: T values if fixed, how it was calculated if heuristic. number if BEM
	- trials: how often each configuration is evaluated
	- differing_T: true if the ensembles are supposed to be BEMs, false if BEEs
	
2. optional:
	- modes=['median','max','mean']: list of modes as strings (how to calculate cluster size per point in ensemble)
	- ensembles=[1,2,3]: list of ensemble sizes
	- plot_detected=False: if true plots scatterplot with data points that have a OS bigger or equal to the smallest OS of an anomaly highlighted. only for 2D dataset and for the last trial of each configuration
	- plot_histograms=False: if true plots histograms of cluster sizes per point for the last trial of each configuration
	- pdf=None: where to store the histograms/detected anomaly plots, if None they are not stored
	

#### od_and_plot:

1. mandatory arguments: 
	- dataset, trials, plot_detected, pdf: as above
	- Models: list of models to be compared

2. optional:
	- trials=3 how often each Model is evaluated
	- plot_detected=False: if true plots scatterplot with data points that have a OS bigger or equal to the smallest OS of an anomaly highlighted. only for 2D dataset and for the last trial of each configuration
	- pdf=None: where to store the histograms/detected anomaly plots, if None they are not stored
	
This function can be used to compare different methods (OCSVM, isoForest, BE). Methods are given as a list of Models.
It evaluates each model multiple times and stores metric results and training times for given models. The funciton returns a dataframe with metric results. The (average) training times are stored in the Model instances.

#### plot_histograms:

1. arguments:
	- M: list of fitted Birch Ensembles as Models
	
plots how often each cluster score appears for each Birch Ensemble Model (in last trial if Model has been fitted multiple times)

## utils.py
library with non clustering/anomaly detection related useful functions

## Birch.py
library with functions to evaluate the BIRCH ensembles and other useful functions

important function:
- od_and_plot: creates data sets, see documentation for details 

- load_models: load model results from file (training time is stored there)

## BirchEnsemble.py
class representing one ensemble configuration with array of T-values, ensemble size e (number of values in T array) and mode.
Main method 'fit' takes array of data points X and calculates y_score for each data point
negative(!) of y_score can be retrieved from function 'score_samples' (to allow similar usage in Model class as for isolation forest and OCSVM)
y_score is negative because the other two models return high scores for normal data and low scores for outliers as well. This is compensated in the fit function of the Model class.

## Model.py
class representing a model.

important attributes: 
- od_method: instance of a specific OD_method (OCSVM, BE, isoForest)
- name: name of OD method
- fit_time: training time (average with multiple trials)

## Dataset.py
class representing an (OD) dataset.

attributes:
- X: data
- od_labels: {0,1}
- labels: only for mnist, stores digits based on original dataset

## tests.py
unit tests

### metrics
ROC_AUC

AP

prec@n for n = #outliers

OSTM(m) = average OS of top m outliers for m = #outliers

OSTM ratio = OSTM(m)/OSTM(2m) for m = #outliers