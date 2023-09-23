# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:45:26 2020

@author: Theresa
"""

import numpy as np
import matplotlib.pyplot as plt

def get_histogram_values(data: np.ndarray, caption: str, row: int, column: int, ax: np.array, xlabel='', ylabel='') -> (np.ndarray, np.ndarray):
    """Plots histogram of occurances for values in given data array
    
    arguments:
        data: data to be displayed in histogram
        caption: explanation for what is shown
        row, column, index: where to plot
    
    returns:
        c: counts of occurances
        l: values of occurances
    """
    ax[row, column].title.set_text(caption)
    if row == ax.shape[0]-1:
        ax[row, column].set_xlabel(xlabel)
    if column == 0:
        ax[row, column].set_ylabel(ylabel)
    ax[row, column].grid(True)
        
    bins = int(min((int(max(data))-int(min(data))), 120))
    
    if int(min(data)) == int(max(data)):
        r = (int(min(data))-1, int(max(data))+1)
        bins+=2
    else:
        r = (int(min(data)), int(max(data)))
    c, l, p =  ax[row, column].hist(data, bins=bins, range=r, color='#377eb8') #ax[row, column].bar(l.astype(str), c)
    return c,l

def flatten_last_axes(X: np.ndarray, num_first_kept_axes: int) -> np.ndarray:
    """This function creates a new tensor that keeps the
    num_first_kept_axes of X and flattens all the remaining ones.
    
    arguments:
        X: np.array: a dataset to partially flatten
        num_first_kept_axes: the number of leading axes that will not be
                       flattened
    returns: a copy of X with the flattened shape
    """
    Y = np.copy(X)
    if  num_first_kept_axes not in range(len(Y.shape)):
        raise ValueError("argument must be in range(len(X.shape))")
    if num_first_kept_axes == 0:
        return Y.flatten()
  
    new_shape = np.r_[Y.shape[:num_first_kept_axes], np.prod(Y.shape[num_first_kept_axes:])]
    return Y.reshape(new_shape)

def get_subplots(plot: bool, rows: int, columns: int, title: str) -> (plt.Figure, np.array):
    """Creates figure with specified amount of subplots
    
    arguments:
        plot: if true figure and ax is created, else ax is an empty array
        rows, columns: shape of figure
        title: figure title
        
    returns:
        figure
        ax: array of subplots
    """
    if plot:
        fig, ax = plt.subplots(rows, columns, figsize=(2.5*columns,2.5*rows))
        fig.suptitle(title, y=1.05, fontsize='x-large')
        if rows==1 and columns==1:
            ax = np.array(ax).reshape(1,1)
        elif rows==1:
            ax = np.array([ax])
        elif columns == 1:
            ax = ax.reshape(rows, columns)
        return fig, ax
    else:
        return None, np.array([])

def plot_scatterplots(X: np.ndarray, idx: np.ndarray, n: int, rows=0, columns=0, ax=None, caption='plot'):
    """Plots scatterplots with the n data points with the highest index marked as orange
    
    arguments:
        X: unshuffled array with datapoints (normal and anormal)
        idx: indices of the elements in X sorted by outlier score (high index value = high outlier score)
        n: how many of the elements with the highest index are shown in orange
        rows, column: position of the subplot to plot the results
        ax: array of subplots, if none plot in a new figure
        caption: descriptive text above of scatterplot
        
    returns nothing    
    """
    colors = np.array(['#317eb8'] * len(X))
    colors[idx[max(len(idx)-n,0):]] = '#ff2200'
    if ax is None:
        plt.scatter(X[:,0], X[:,1], s=5, c=colors)
        plt.title(caption)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    else:
        ax[rows, columns].title.set_text(caption)
        if rows == ax.shape[0]-1:
            ax[rows, columns].set_xlabel("detected anomalies")
        ax[rows, columns].scatter(X[:,0], X[:,1], s=1.5, color=colors)

def plot_images(X: np.ndarray, caption, benign_labels=None, data=None, dim=(28,28)):
    """plots images given in X and ,if specified, centroids of benign class labels of whole data set
    
    arguments: 
        X: 2-dimensional array where each row is a image, subset of data
        caption: caption of the plot
        benign_labels: labels of benign classes
        data: Dataset
        dim: dimensions of the picture
        
    returns nothing
    """
    fig, ax = get_subplots(True, int((len(X)-1)/5)+1, 5, caption)
    for i in range(len(X)):
        ax[int(i/5), i%5].imshow(X[i].reshape(dim))
    plot_figure(True, fig, None)
    
    if not benign_labels is None and not data is None:
        fig, ax = get_subplots(True, int(len(benign_labels)/5)+1, min(len(benign_labels), 5), 'centroids of benign classes')
        for l in range(len(benign_labels)):
            ct = get_centroid_of_class(data.X, data.labels, benign_labels[l])
            ax[int(l/5), l%5].imshow(ct.reshape(dim))
        plot_figure(True, fig, None)

def get_centroid_of_class(X: np.array, labels: np.array, label: int) -> np.array:
    """calculates and returns centroid of cluster containing all data points with specified label
    
    arguments:
        X: data array
        labels: labels of data (NOT od_labels)
        label: label of class whose centroid is to be determined, must be in 'labels'
    
    returns:
        centroid 
    """
    label_data = X[np.where(labels==label)]
    centroid = np.average(label_data, axis=0)
    return centroid

   
def plot_figure(plot: bool, fig: plt.figure, pdf=None):
    """Plots figure if condition is true and saves figure to pdf

    arguments: 
        plot: if true fig is plotted
        fig: Figure
        pdf: pdfPages object of pdf document where the figure should be plotted
    
    returns nothing
    """
    if plot:
        fig.tight_layout()
        #plt.show()
        if pdf is not None:
            pdf.savefig(fig)

def partition(c: int, n: int) -> list:
    """finds all ordered c partitions of the number n
    (all combinations of c numbers that add up to n where [1,2] != [2,1])
    
    arguments:
        c: how many summands
        n: number to partition
        
    returns:
        list containing each partition in a separate list 
    """
    if c==1:
        return [[n]]
    p = []
    for i in np.arange(1,n-c+2):
        subpartitions = partition(c-1, n-i)
        for s in subpartitions:
            p += [[i]+s]
    return p
