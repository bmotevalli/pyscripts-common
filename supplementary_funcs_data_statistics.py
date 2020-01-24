# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:04:29 2019

@author: mot032
"""

import numpy as np     
from pylab import *
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.matlib import repmat
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist, pdist

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import shutil

import os


def cluster_quality(X, centroids):

    D_k = cdist(X, centroids, 'euclidean')
    cIdx = np.argmin(D_k,axis=1)
    dist = np.min(D_k,axis=1)

    tot_withinss = sum(dist**2)  # Total within-cluster sum of squares
    totss = sum(pdist(X)**2)/X.shape[0]	   # The total sum of squares
    betweenss = totss - tot_withinss		  # The between-cluster sum of squares

    variance_retain = betweenss/totss*100
	
    return variance_retain


def get_centroids(arr_clusters, arr_X):
    
    unqvals = np.unique(arr_clusters)
    
    centroids = []
    for val in unqvals:
        
        if val > -1:
            centroids.append(arr_X[np.where(arr_clusters == val)].mean(axis = 0))
    
    return np.array(centroids)


def vis_xyz(file_path):
    import py3Dmol
    with open(file_path) as f:
        xyz = f.read()
        
    xyzview = py3Dmol.view(width=400,height=400)
    xyzview.addModel(xyz,'xyz')
    xyzview.setStyle({'stick':{}})
    xyzview.setBackgroundColor('0xeeeeee')
    xyzview.zoomTo()
    xyzview.show()

    
def kmean_elbow_percentange(X, lst_kms, figname):
    
    from scipy.spatial.distance import cdist
    from scipy.spatial.distance import pdist
    ##### cluster data into K=1..20 clusters #####
    
    K_MAX = len(lst_kms)
    KK = range(1,K_MAX+1)
    
    centroids = [km.cluster_centers_ for km in lst_kms]
    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]

    tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
    totss = sum(pdist(X)**2)/X.shape[0]   # The total sum of squares
    betweenss = totss - tot_withinss  # The between-cluster sum of squares

    ##### plots ##### 
    kIdx = 4  # Elbow
    clr = cm.nipy_spectral( np.linspace(0,1,20) ).tolist()
    mrk = 'os^p<dvh8>+x.'
    variance_retain = betweenss/totss*100
    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(KK, betweenss/totss*100, marker='o', color='#212F3D')
#     ax.plot(KK[kIdx], betweenss[kIdx]/totss*100, marker='o', markersize=12, 
#         markeredgewidth=5, markeredgecolor='r', markerfacecolor='None')
#     ax.set_ylim((0,100))
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')
#     plt.title('Elbow for KMeans clustering')
    plt.savefig(figname, dpi=300, bbox_inch = 'tight')
    plt.show()
    
    
def kmean_elbow(X, K_MAX = 20):
    
    distortions = []
    kmeans = []
    for i in range(1, K_MAX + 1):
        km = KMeans(n_clusters=i,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=0)
        
        km.fit(X)
        distortions.append(km.inertia_)        
        kmeans.append(km)        
    
    distortions_norm = [dist / distortions[0] for dist in distortions]
    plt.plot(range(1,K_MAX + 1), distortions_norm, marker='o', color='black')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.savefig('KMeans_Elbow_all_ex_99.png', dpi = 300)
    plt.show()
        
    return kmeans
    
    
def kmean_elbow_distortion(lst_kms, figname):
    
    distortions = []
    K_MAX = len(lst_kms)
    for km in lst_kms:        
        distortions.append(km.inertia_)            
    
    distortions_norm = [dist / distortions[0] for dist in distortions]
    plt.plot(range(1,K_MAX + 1), distortions_norm, marker='o', color='#212F3D')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.show()    



def extract_closes_match(X_ref, X_data):
    
    """
    X_ref:   The reference point from which the closest point is going to be 
             extracted. (1D-Array)
    
    X_data:  The dataset. (2D-Array) (n_points x n_dim)
    
    """
    
    n_dim = len(X_ref)
    r = cdist(X_data, X_ref.reshape(1, n_dim))
    i_min = np.argmin(r)
    
    return i_min, r[i_min], X_data[i_min, :]


def ecdf(X, x):
    
    """Emperical Cumulative Distribution Function
    
    X: 
        1-D array. Vector of data points per each feature (dimension), defining
        the distribution of data along that specific dimension.
        
    x:
        Value. It is the value of the corresponding dimension. 
        
    P(X <= x):
        The cumulative distribution of data points with respect to the archetype
        (the probablity or how much of data in a specific dimension is covered
        by the archetype).
    
    """
    
    return float(len(X[X < x]) / len(X))


def calc_SSE(X_act, X_appr):
    
    """
    This function returns the Sum of Square Errors.
    """
    
    return ((X_act - X_appr) ** 2).sum()

def calc_SST(X_act):
    
    """
    This function returns the Sum of Square of actual values.
    """
    
    return (X_act ** 2).sum()




def explained_variance(X_act, X_appr, method = 'sklearn'):
    
    
    if (method.lower == 'sklearn'):
            
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(X_act.T, X_appr.T)  
        
    else:
        
        SSE = calc_SSE(X_act, X_appr)
        SST = calc_SST(X_act)
        
        return (SST - SSE) / SST
    

def ternaryPlot(data, scaling=True, start_angle=90, rotate_labels=True,
                labels=('one','two','three'), sides=3, label_offset=0.10,
                edge_args={'color':'black','linewidth':1},
                fig_args = {'figsize':(8,8),'facecolor':'white','edgecolor':'white'},
                grid_on = True
        ):
    '''
    source: https://stackoverflow.com/questions/701429/library-tool-for-drawing-ternary-triangle-plots
    
    This will create a basic "ternary" plot (or quaternary, etc.)
    
    DATA:           The dataset to plot. To show data-points in terms of archetypes
                    the alfa matrix should be provided.
    
    SCALING:        Scales the data for ternary plot such that the components along
                    each axis dimension sums to 1. This conditions is already imposed 
                    on alfas for archetypal analysis.
    
    start_angle:    Direction of first vertex.
    
    rotate_labels:  Orient labels perpendicular to vertices.
    
    labels:         Labels for vertices.
    
    sides:          Can accomodate more than 3 dimensions if desired.
    
    label_offset:   Offset for label from vertex (percent of distance from origin).
    
    edge_args:      Any matplotlib keyword args for plots.
    
    fig_args:       Any matplotlib keyword args for figures.
    
    '''
    basis = np.array(
                    [
                        [
                            np.cos(2*_*pi/sides + start_angle*pi/180),
                            np.sin(2*_*pi/sides + start_angle*pi/180)
                        ] 
                        for _ in range(sides)
                    ]
                )

    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data for you.
        newdata = np.dot((data.T / data.sum(-1)).T,basis)
    else:
        # Assumes data already sums to 1.
        newdata = np.dot(data,basis)

#    fig = plt.figure(**fig_args)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    for i,l in enumerate(labels):
        if i >= sides:
            break
        x = basis[i,0]
        y = basis[i,1]
        if rotate_labels:
            angle = 180*np.arctan(y/x)/pi + 90
            if angle > 90 and angle <= 270:
                angle = mod(angle + 180,360)
        else:
            angle = 0
        ax.text(
                x*(1 + label_offset),
                y*(1 + label_offset),
                l,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=angle
            )

    # Clear normal matplotlib axes graphics.
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)
    
    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    ignore = False
    for i in range(sides):
        for j in range(i + 2, sides):
            if (i == 0 & j == sides):
                ignore = True
            else:
                ignore = False                        
#                
            if not (ignore):
#            if (j!=i & j!=i+1 & j != i-1):                        
                lst_ax_0.append(basis[i,0] + [0,])
                lst_ax_1.append(basis[i,1] + [0,])
                lst_ax_0.append(basis[j,0] + [0,])
                lst_ax_1.append(basis[j,1] + [0,])

#    lst_ax_0.append(basis[0,0] + [0,])
#    lst_ax_1.append(basis[0,1] + [0,])
    
    ax.plot(lst_ax_0,lst_ax_1, color='#FFFFFF',linewidth=1, alpha = 0.5)
    
    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    for _ in range(sides):
        lst_ax_0.append(basis[_,0] + [0,])
        lst_ax_1.append(basis[_,1] + [0,])

    lst_ax_0.append(basis[0,0] + [0,])
    lst_ax_1.append(basis[0,1] + [0,])
#    ax.plot(
#        [basis[_,0] for _ in range(sides) + [0,]],
#        [basis[_,1] for _ in range(sides) + [0,]],
#        **edge_args
#    )
#    
    ax.plot(lst_ax_0,lst_ax_1,linewidth=1) #, **edge_args ) 
    

    return newdata,ax 


def compare_profile(prof_ref, prof_2, feature_cols, direction = 'h'):
    """
    This function compares the profile of two data points. Note, this two data
    points could be archetypes and prototypes as well.
    
    feature_cols:
        Optional input. list of feature names to use to label x-axis.
    """               
    
    plt.style.use('ggplot')
    
    n_dim = len(feature_cols)   
       
    fig = plt.figure()
    
    if (direction == 'h'):
    
        x_vals = np.arange(1, n_dim + 1)
        plt.bar(x_vals, prof_ref * 100.0, color = '#273746', label='Minimum Case')
        plt.bar(x_vals, prof_2 * 100.0, color = '#D81B60', alpha = 0.5, label='Maximum Case')
        plt.xticks(x_vals, feature_cols, rotation='vertical')
        plt.ylim([0,100])
    #    plt.ylabel('A' + str(i + 1))
        plt.rcParams.update({'font.size': 10})
        plt.tight_layout()
        plt.legend(loc='upper left') 
        
    elif (direction == 'v'):
        
        y_vals = np.arange(1, n_dim + 1)    
    
        plt.barh(y_vals, prof_ref * 100.0, color = '#273746', label='Archetype')
        plt.barh(y_vals, prof_2 * 100.0, color = '#D81B60', alpha = 0.5, label='Closet Data')  
        plt.yticks(y_vals, feature_cols)
        plt.xlim([0,100])
        plt.rcParams.update({'font.size': 10})
        plt.tight_layout()
        plt.legend(loc='upper left') 
        
    else:
        
        raise ValueError('acceptable direction values are "h" and "v"!')

    return fig


def datapoint_profile(x_point, x_data):
    
    point_profile = []
    
    for i, p in enumerate(x_point):
        
        d = x_data[i, :]
        
        point_profile.append(ecdf(d, p))
        
    return np.array(point_profile)


def list_missing_data_cols(df):
    lst_cols_missing = list(df.columns[df.isnull().any()])
    lst_cols_miss_data = [(col, df[col].isnull().sum()) for col in lst_cols_missing]
    lst_cols_miss_data = sorted(lst_cols_miss_data, reverse = True, key=lambda x: x[1])

    print('')
    print('List of columns with missing data:')
    print('==================================\n')
    print('')
    print('Total Number of Data (Rows):                 ', len(df))
    print('Total Number of Columns with Missing Data:   ', len(lst_cols_miss_data))
    print('Total Number of Columns:                     ', len(df.columns))
    print('')
    for item in lst_cols_miss_data: print(item[0],": ", item[1])
    print('')

    return lst_cols_miss_data


def show_balance(col_feat, df, label = 'Cluster No.', show_pie = True, 
                 title = '', show_percent = True,
                 colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']):
    
    lst_share = []
    lst_labels = []
    for col_cat in df[col_feat].unique():
        
        share = len(df[df[col_feat] == col_cat]) / len(df) * 100
        print(f'{label} = {col_cat}: {share: 5.2f} %')
        lst_share.append(share)
        if 'km' in col_feat:
            lst_labels.append(f'C{col_cat + 1}')
        else:
            lst_labels.append(col_cat)
    
    if (show_pie):
        
        plt.style.use('default')
        
        #colors       

        fig1, ax1 = plt.subplots(figsize=(3, 3))
        if show_percent:
            ax1.pie(lst_share, colors = colors, labels=lst_labels, autopct='%1.1f%%', startangle=90)
        else:
            ax1.pie(lst_share, colors = colors, labels=lst_labels, startangle=90)
        
        #draw circle
        centre_circle = plt.Circle((0,0),0.8,fc='white')
        fig1 = plt.gcf()
        fig1.gca().add_artist(centre_circle)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')  
        plt.tight_layout()
        
        return fig1


def share_percentage(df, columns):
    """
   This function shows the balance in each column of dataframe
    """
    import numpy as np
    shares_cols = dict()
    
    n_rows = len(df)

    for col in columns:
        unq_vals = df[col].unique()
        shares = dict()
        for val in unq_vals:
            shares[val] = round(len(df[df[col] == val]) / n_rows * 100.0, 3)
            
        shares_cols[col] = shares

    return shares_cols  


def report_unbalance_cols(df, columns, threshold = 90):
    
    import numpy as np
    
    n_rows = len(df)
    
    report_cols = []
    for col in columns:
        unq_vals = df[col].unique()
        if (len(unq_vals) < 50):
            for val in unq_vals:
                share = (len(df[df[col] == val]) / n_rows * 100.0)
                if ((len(df[df[col] == val]) / n_rows * 100.0) > threshold):
                    report_cols.append(col)

    return report_cols



def greedy_elimination(corr, col_list, main_label = None, thresh_val = 75):
    
    from copy import copy
    import random   
    
    random.seed(1001)
    
    df = pd.DataFrame(columns=['feat_1', 'feat_2', 'corr', 'deleted'])
    
    arr_corr = np.array(corr)
    
    # Getting the upper triangle of the matrix
    up_tri_inx = np.triu_indices(len(arr_corr),1)     
    
    # Creating a tuple of upper-triangle indexes and corresponding values, and sorting.
    lst_tpls = sorted(list(zip(up_tri_inx[0],up_tri_inx[1],arr_corr[up_tri_inx])), reverse = True, key=lambda x: x[2])
    
    cols_to_keep = copy(col_list)
    
    for i,j,v in lst_tpls:
        
        if (v < thresh_val):
            break
        
        col_i = col_list[i]
        col_j = col_list[j]
        
        if (main_label == None):
            
            if ((col_i in cols_to_keep) & (col_j in cols_to_keep)):
                
                col_to_del = random.choice([col_i, col_j])
                cols_to_keep.remove(col_to_del)
                
                #col_i = '$' + col_i + '$'
                #col_j = '$' + col_j + '$'
                #col_to_del = '$' + col_to_del + '$'
                
                #print(f'{col_i} | {col_j} = {v: 5.2f} Deleted Feat: {col_to_del}')     
                
                df = df.append({'feat_1': col_i, 'feat_2': col_j, 'corr': v, 'deleted': col_to_del}, ignore_index=True)
        else:
        
            if ((col_i != main_label) & (col_j != main_label) & (col_i in cols_to_keep) & (col_j in cols_to_keep)):                  
                corr_i = corr[main_label][col_i]
                corr_j = corr[main_label][col_j]

                # print_line = '('+col_i + ', ' + col_j + ')= ' + str(v) + '\tCorr. With ' + main_label + '(' + '{:0.4f}, {:0.4f})\tDeleted Feat.: '.format(corr_i, corr_j)
                print_line = '('+col_i + ', ' + col_j + ')= ' + str(v) + '\tDeleted Feat.: '.format(corr_i, corr_j)
                if (corr_i < corr_j):
                    print_line = print_line + col_i
                    cols_to_keep.remove(col_i)
                    df = df.append({'feat_1': col_i, 'feat_2': col_j, 'corr': v, 'deleted': col_i}, ignore_index=True)
                else:
                    print_line = print_line + col_j
                    cols_to_keep.remove(col_j)
                    df = df.append({'feat_1': col_i, 'feat_2': col_j, 'corr': v, 'deleted': col_j}, ignore_index=True)

                # print(print_line)
    
    print('\n\nOriginal Number of Columns: ', len(col_list))
    print('\n\nNumber of Columns To Keep: ', len(cols_to_keep))
    print('\n\nNumber of Columns To Delete: ', len(col_list) - len(cols_to_keep))
    
    return cols_to_keep, df


def scatter_label_vs_features(df, lst_x_cols, main_label, lst_tags = None, lst_colors = None, n_row=3, n_col=3, num_fig_to_show = None):

    count = 1
    tot_subs = n_row * n_col
    for i, x_col in enumerate(lst_x_cols):
        
        if not (num_fig_to_show == None):
            if (i == num_fig_to_show):
                break
        
        if(i % tot_subs == 0):
            count = 1
            plt.figure(figsize=(20,12))

        plt.subplot(n_row,n_col,count)
        
        if (lst_tags == None):
            plt.scatter(df[x_col],df[main_label])    
        else:
            for i, tag in enumerate(lst_tags):
                plt.scatter(df[x_col],df[main_label], color = lst_colors[i])
                
        plt.xlabel(x_col)
        plt.tight_layout()

        count += 1
        
        
def correlation_matrix(df, show_matrix = True):
    
    corr = df.corr(method='spearman').abs().mul(100).astype(float)
    
    fig = plt.figure()
    cmap = sns.diverging_palette(h_neg = 210, h_pos=350, s=90, l=30, as_cmap=True)
    cg = sns.clustermap(data = corr, cmap='Blues',  metric='correlation', figsize=(8,8))
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    #plt.savefig('paper_images/Fig1-correlation_matrix.png', dpi=300, bbox_inches='tight')
    
    
    return corr, fig


def show_EV_PCA(pca):
        
    var_exp = pca.explained_variance_ratio_ * 100
    cum_var_exp = np.cumsum(var_exp)

    num_comp = len(pca.explained_variance_ratio_)

    fig = plt.figure()
    plt.bar(range(1,num_comp + 1), var_exp, alpha=0.5, align='center',
    label='individual explained variance')
    plt.step(range(1,num_comp + 1), cum_var_exp, where='mid',
    label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
#    plt.show()
        
    return fig
        
    
def show_PCA_in_Feat(lst_Feat, res, shape = [3,3]):

    n = shape[0]
    m = shape[1]
    tot_plot = m * n

    x = np.arange(1, len(lst_Feat) + 1)

    count = 1
    lst_figs = []
    for i in range(res.shape[1]):            
     #   if (i % tot_plot == 0):
      #      count = 1
       #     plt.figure(figsize=(20,12))
        
        fig = plt.figure(figsize=(20,12))
#        plt.subplot(n,m,count)
        plt.bar(x, res[:,i])
        plt.xticks(x, lst_Feat, rotation='vertical')
#        plt.show()

        count += 1 
        
        lst_figs.append(fig)
        
    
    return lst_figs
    
    
    
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    fig = plt.figure(figsize=(6,4))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return fig    

def funcLinear(x, a, b):
    return a*x + b

def plot_gth_pre(Y_label, Y_pre, range_set = True, tag='Train'):
    from scipy.optimize import curve_fit
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.preprocessing import MinMaxScaler

    min_glob = min(Y_label.min(), Y_pre.min())
    max_glob = max(Y_label.max(), Y_pre.max())

    Y_label = (Y_label - min_glob) / (max_glob - min_glob).flatten()
    Y_pre = (Y_pre - min_glob) / (max_glob - min_glob).flatten()

    # Y_label = sc.fit_transform(Y_label.reshape(-1,1)).flatten()
    # Y_pre = sc.fit_transform(Y_pre.reshape(-1,1)).flatten()

    fig = plt.figure(figsize=(6,6))
    # Fitting best line:
    # ==================
    pre_Y = Y_pre
    parameter, covariance_matrix = curve_fit(funcLinear,
    Y_label.astype(float),
    pre_Y.flatten().astype(float))
    xx = np.linspace(-0.1, 1.1, 30)
    plt.plot(xx, funcLinear(xx, *parameter), color='#52BE80', linewidth = 2, label='fit')
    axes = plt.gca()
    if range_set:
        axes.set_xlim([-0.1,1.1])
        axes.set_ylim([-0.1,1.1])
    else:
        # min_true =  np.min(Y_label) - 0.1 * np.min(Y_label)
        # max_true =  np.max(Y_label) + 0.1 * np.max(Y_label)
        # min_pre =  np.min(Y_pre) - 0.1 * np.min(Y_pre)
        # max_pre =  np.max(Y_pre) + 0.1 * np.max(Y_pre)

        axes.set_xlim([min_glob,max_glob])
        axes.set_ylim([min_glob,max_glob])

    lims = [
    np.min([axes.get_xlim(), axes.get_ylim()]), # min of both axes
    np.max([axes.get_xlim(), axes.get_ylim()]), # max of both axes
    ]

    # 45 degree line:
    # ===============
    plt.plot(lims, lims, '--', color='#A569BD', linewidth=5, alpha=0.75, zorder=0)
    
    # Scattered Data:
    # ===============
    plt.scatter(Y_label,pre_Y,
    marker='o',
    s=20,
    facecolors='#273746',
    edgecolors='#273746')
    plt.legend(['Best Fit','Perfect Fit', 'Data'], loc='lower right')
    plt.text(0, 0.9, r'slope: %.2f'%parameter[0])
    plt.text(0, 0.85, r'interception: %.2f'%parameter[1])
    plt.xlabel('Normalized '+ tag +' Value')
    plt.ylabel('Normalized Prediction Value')
    from sklearn.metrics import r2_score
    plt.title("$R^2 = %.5f$"%r2_score(Y_label,pre_Y.flatten()))
    
    return fig

    
def plot_it(y_true, y_pred, filename=None):
    fig = plt.figure(figsize=(6,6))

    y_min = min(y_true.min(), y_pred.min())
    y_max = min(y_true.max(), y_pred.max())
    xx = np.linspace(y_min, y_max, 30)

    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1,1), y_true)

    y_line = lr.predict(y_pred.reshape(-1,1))

    # 45 degree line:
    # ===============
    plt.plot(xx,xx, '--', color='#A569BD', linewidth=5, label = 'Perfect fit')

    # Best straigth line:
    # ===================
    # plt.plot(np.unique(y_pred), np.poly1d(np.polyfit(y_pred, y_true, 1))(np.unique(y_pred)), color='#52BE80', linewidth = 2, label= 'Best fit')
    # plt.plot(y_pred.flatten(), y_line, color='#52BE80', linewidth = 2, label= 'Best fit')
    # plt.plot(y_line, y_pred.flatten(), color='#52BE80', linewidth = 2, label= 'Best fit')

    # scattered y-true and y-pred:
    # ============================
    # plt.scatter(y_pred, y_true, color = '#273746', label = 'Data')
    plt.scatter(y_true, y_pred, color = '#273746', label = 'Data', s = 20)    
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.legend(loc='upper left')
    R2 = np.round(r2_score( y_true, y_pred ), 3)
    RMSE = np.round(mean_squared_error( y_true, y_pred )**0.5, 3)
    MAE = np.round(mean_absolute_error( y_true, y_pred ), 3)
    plt.title(f"$R^2$ = {R2}, RMSE = {RMSE}, MAE = {MAE}")
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig
    
    
def skf_cv_scores(model, metrics, x_train, y_train, n_splits = 5, bins = 12, print_scores = False):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=42)
    bins_   = np.linspace(y_train.min(), y_train.max(), bins)
    y_train_binned_cv = np.digitize(y_train, bins_) 
    
    CV_scores = []
    for cv_training_index, cv_testing_index in skf.split(x_train, y_train_binned_cv):
        X_training_cv = x_train[cv_training_index, :]
        X_testing_cv = x_train[cv_testing_index, :]

        Y_training_cv = y_train[cv_training_index]
        Y_testing_cv = y_train[cv_testing_index]

        # fit model to training dataset
        model.fit(X_training_cv, Y_training_cv)

        # Test
        CV_scores.append(metrics(model.predict(X_testing_cv), Y_testing_cv))

    accuracy = np.mean(CV_scores)
    uncertainty = np.std(CV_scores)*2

    if print_scores:
        print('CV Scores:', np.round(CV_scores, 3))
        print('Accuracy:',np.round(accuracy, 3),'+/-',np.round(uncertainty, 3))
    
    return CV_scores    
    
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds, **kwargs):
    filename = kwargs.get('filename', None)
    """
    reference:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """    
    fig = plt.figure()
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalised confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, round(cm[i,j],2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        if (i==1) and (j==1):
            plt.text(j, i, round(cm[i,j],2), horizontalalignment="center", color="white")
        elif (i==0) and (j==0):
            plt.text(j, i, round(cm[i,j],2), horizontalalignment="center", color="black")
        else:
            plt.text(j, i, round(cm[i,j],2), horizontalalignment="center", color="black")


    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    return fig


def plot_feature_importance(ft_set, feature_importance, show_cols = 30):
    
    fig = plt.figure(figsize=(8,6))
    w_lr_sort, ft_sorted, _ = return_feature_importance(ft_set, feature_importance, show_cols = show_cols)
    x_val = list(range(len(w_lr_sort)))

    plt.bar(x_val, w_lr_sort, color = '#212F3D')
    plt.xticks(x_val, ft_sorted, rotation='vertical')
    
    return fig

def return_feature_importance(ft_set, feature_importance, show_cols = 30):

    w_lr = copy(np.abs(feature_importance))
    w_lr = 100 * (w_lr / w_lr.max())
    sorted_index_pos = [index for index, num in sorted(enumerate(w_lr), key=lambda x: x[-1], 
                   reverse=True)]

    ft_sorted = []
    w_lr_sort = []
    for i, idx in enumerate(sorted_index_pos):
        if i > show_cols:
            break
        ft_sorted.append(ft_set[idx])
        w_lr_sort.append(w_lr[idx])

    return w_lr_sort, ft_sorted, sorted_index_pos



def del_important_feat_classifier(estimator, X, y):

    pass

def feature_importance_elimination(estimator, df, ft_init, ylabel, imp_type = 'coef', ft_del_num = 3, metric = f1_score, cv = 5, threshold = 0.99,
                                                    test_size = 0.25, random_state = 0, bins = 2):

    '''
    This function gradually removes the most important features, then refits the model with new set. The repeatition continues until a certain
    threshold is met.

        estimator:      classifier/regressor model

        df:             dataframe of dataset

        ft_init:        initial feature set (name of columns) (it should be a list)

        ylabel:         class label vector (1D-array)

        imp_type:       feature importance type of estimators. Parametric models have a .coef_ attributes. Non-parametric ones such as trees and random forest
                        have .feature_importance_ attribute. For .coef_ enter 'coef', for .feature_importance_ just enter any string.
         
        ft_del_num:     number of features to delet in each iteration. e.g. 3 means the three most important features are deleted.

        metric:         is the metric to evaluate the model.

        cv:             number of cross-validations

        threshold:      accuracy threshold to stop the iterations.

        test_size:      test-size

        random_state:   stating the random seed number
        bins:           is used for cross-validation
    '''

    bins_   = np.linspace(ylabel.min() - 0.01 * ylabel.min(), ylabel.max() + 0.01 * ylabel.max(), bins)
    y_binned = np.digitize(ylabel, bins_) 

    sc = StandardScaler()
    thresh = 2.0
    ft_curr = copy(ft_init)
    ft_del = []
    acc = []
    iteration = 1
    with open('log.txt', 'w') as f:
        f.write('Start feature importance elimination:\n')
        f.write('=====================================\n\n')
            
        while (thresh > threshold):

            X_sc = sc.fit_transform(df[ft_curr])
            X_train, X_test, y_train, y_test = train_test_split(X_sc, ylabel, stratify = y_binned, test_size= test_size, random_state = random_state)
            
            estimator.fit(X_train, y_train)
            thresh = np.array(skf_cv_scores(estimator, metric, X_train, y_train, n_splits = cv, bins = bins)).mean()
            
            if (imp_type == 'coef'):
                w_lr_sort, ft_sorted, _ = return_feature_importance(ft_curr, estimator.coef_.flatten(), show_cols = ft_del_num - 1)
            else:
                w_lr_sort, ft_sorted, _ = return_feature_importance(ft_curr, estimator.feature_importances_.flatten(), show_cols = ft_del_num - 1)

            ft_del.append(ft_sorted)
            acc.append(thresh)
            
            f.write(f'iteration {iteration}:'+'\n')
            f.write(f'------------'+'\n')
            f.write(f'Accuracy: {thresh}'+'\n')

            print('')
            print(f'iteration {iteration}:')
            print('=============')
            print(f'Accuracy: {thresh}')
            for col in ft_sorted:
                print(col)
                f.write(f'{col}'+'\n')
                ft_curr.remove(col)

            iteration += 1

    return acc, ft_del, ft_curr


#======================================================
# RNN
#======================================================
def get_normalization_parameters(traindf, features):
    dropCol=[]
    normalization_parameters = {}
    for column in features:       
        mean = traindf[column].mean()
        std = traindf[column].std()
        maxi = traindf[column].max()
        mini = traindf[column].min()
        if std!=0:
#            traindf[column] = (traindf[column] - mean) / (std)
             traindf[column] = (traindf[column] - mini) / (maxi-mini)

        else:
#            traindf[column] = 0 #traindf[column] 
            dropCol.append(column)
            
        normalization_parameters[column]={'mean': mean, 'std': std, 'max': maxi, 'min': mini}    
    traindf.drop(traindf[dropCol], axis=1,inplace=True)

    return normalization_parameters


def denormalization_parameters(columnVal, normalization_parameters):

    mean = normalization_parameters['mean']  
    std = normalization_parameters['std']    
    maxi = normalization_parameters['max']  
    mini = normalization_parameters['min']    
#    deNorm = (columnVal * (std) ) + mean
    deNorm = (columnVal * (maxi-mini) ) + mini

    return deNorm

def atom_type_to_one_hot(atom_type):
    if atom_type == 'C':
        return 1,0
    if atom_type == 'H':
        return 0,1


def main_extracting_features_per_atom(base_path, 
                                        xyz_file, 
                                        out_path, 
                                        out_file_name, 
                                        dic_encode_atom_type, 
                                        lst_atom_types, 
                                        max_vals = [1,1,1], 
                                        centered = True,
                                        ordering = 'sorted',
                                        df_bond = pd.DataFrame([1.7], columns = ['bonds'])
                                        ):

    """
        This function reads a xyz text file, then extracts the x, y, z positions and converts them to float.
        It also encodes the atoms type to binary codes (one-hot-encoding). The function can also
        normalize the x, y, z values by maximum x, y, z values that exist in the set. The maximum values should be
        provided by user. It also extracts other features such as Coordination, Smooth Density, Distance from a reference points.
        At last, the function writes a .csv file containing xyz-positions and encoded atom types.

        base_path:              the path where all xyz files are located.
        xyz_file:               the name of the file.
        out_path:               location where to write the output file.
        out_file_name:          the name of the output file.
        dic_encode_atom_type:   dictionary mapping atom type to binary one-hot-code (e.g. in case of C,H,O:  {'C': [1,0,0], 'O': [0,1,0], 'H': [0,0,1]})
        lst_atom_types:         list of atom types in a same order as in the dictionary (e.g. lst_atom_types = [C, O, H])
        max_vals:               values along x, y, z direction for normalization purposes. 
        centered:               True: geometrical center is set to origin, False: it would transform all the coordinates to positive region.
        ordering:               'org': keeps original ordering, 'sorted': sorts based on x, y, z values, 'org_aligned': original ordering but aligned
                                based on the dimensions of the particle, 'sorted_aligned': sorted order and aligned.
        df_bond:                Dataframe including bonding cut-off among the atoms.
        ref_point:              Reference point from which the distance of the atoms is measured. 
    """
    list_alfa = [0.05, 0.144, 0.5, 1]
    if not (os.path.isdir(out_path)):
        os.mkdir(out_path)

    atom_type_col = 'atom_type'
    xyz_cols = ['x', 'y', 'z']
    try:
        df_xyz_plain = xyz_to_df(base_path, xyz_file, dic_encode_atom_type, lst_atom_types, centered = True)
    except:
        print(f'{xyz_file}')
        return

    n_atom_types = len(lst_atom_types)

    df_xyz_plain['r_dist_min'] = np.linalg.norm((df_xyz_plain[xyz_cols] - df_xyz_plain[xyz_cols].min()).values, axis=1)
    df_xyz_plain['r_dist_max'] = np.linalg.norm((df_xyz_plain[xyz_cols] - df_xyz_plain[xyz_cols].max()).values, axis=1)
    df_xyz_plain['r_dist_mean'] = np.linalg.norm((df_xyz_plain[xyz_cols] - df_xyz_plain[xyz_cols].mean()).values, axis=1)

    for alfa in list_alfa:
        for i in range(n_atom_types):
            for j in range(i, n_atom_types):
                atom_1 = lst_atom_types[i]
                atom_2 = lst_atom_types[j]
                coord_1 = df_xyz_plain[xyz_cols][df_xyz_plain['atom_type'] == atom_1]
                coord_2 = df_xyz_plain[xyz_cols][df_xyz_plain['atom_type'] == atom_2]

                dist_mat = distance_matrix(coord_1, coord_2)
                density_loc = np.exp(-alfa * dist_mat**2).sum(axis=1)
                df_xyz_plain[f'{atom_1}_{atom_2}_{alfa}'] = density_loc
    # =================================================================
    # 1. Extracting XYZ positions, encoding atoms types, ordering atoms
    # =================================================================
    if ordering == 'org':
        df_xyz = xyz_normalize_centered(df_xyz_plain, max_vals = max_vals, centered = centered)

    elif ordering == 'sorted':        
        df_xyz = xyz_normalize_centered(df_xyz_plain, max_vals = max_vals, centered = centered)
        df_xyz = df_xyz.sort_values(['x', 'y', 'z'])

    elif ordering == 'org_aligned':
        df_xyz_plain[xyz_cols] = xyz_align_PCA(df_xyz_plain[xyz_cols].values)
        df_xyz = xyz_normalize_centered(df_xyz_plain, max_vals = max_vals, centered = centered)

    elif ordering == 'sorted_aligned':
        df_xyz_plain[xyz_cols] = xyz_align_PCA(df_xyz_plain[xyz_cols].values)
        df_xyz = xyz_normalize_centered(df_xyz_plain, max_vals = max_vals, centered = centered)
        df_xyz = df_xyz.sort_values(['x', 'y', 'z'])

    # ================================
    # 2. Adding coordination of atoms
    # ================================    
    _, coords = return_neighbour_list(df_xyz_plain, df_bond, atom_type_col = 'atom_type', xyz_cols = ['x', 'y', 'z'])
    df_xyz['coordination'] = coords

    df_xyz.to_csv(os.path.join(out_path, out_file_name), index=False)


def xyz_normalize_centered(df_xyz, max_vals = [1,1,1], centered = True):

    df = copy(df_xyz)
    if (centered):
            # NORMALISE
            df['x'] = (df['x'] - df['x'].min()) / max_vals[0]
            df['y'] = (df['y'] - df['y'].min()) / max_vals[1]
            df['z'] = (df['z'] - df['z'].min()) / max_vals[2]
            # CENTER
            df['x'] = (df['x'] - df['x'].mean())
            df['y'] = (df['y'] - df['y'].mean())
            df['z'] = (df['z'] - df['z'].mean())
    else:
        df['x'] = (df['x'] - df['x'].min()) / max_vals[0]
        df['y'] = (df['y'] - df['y'].min()) / max_vals[1]
        df['z'] = (df['z'] - df['z'].min()) / max_vals[2]
    
    return df

# def xyz_to_df_encoded(base_path, xyz_file, dic_encode_atom_type, lst_atom_types, max_vals = [1,1,1], centered = True):

    
    
#     with open(os.path.join(base_path, xyz_file)) as f:
#         df=pd.DataFrame(
#             list(map(lambda x: list(dic_encode_atom_type[x.split()[0]]) + \
#                        [float(x.split()[1]), float(x.split()[2]), float(x.split()[3]), x.split()[0]],
#             list(filter(lambda x: (len(x.split()) != 0) and (x.split()[0] in lst_atom_types),
#                           f.readlines()))
#                       )),
#         columns = lst_atom_types + ['x', 'y', 'z', 'atom_type']) 

#         if (centered):
#             # NORMALISE
#             df['x'] = (df['x'] - df['x'].min()) / max_vals[0]
#             df['y'] = (df['y'] - df['y'].min()) / max_vals[1]
#             df['z'] = (df['z'] - df['z'].min()) / max_vals[2]
#             # CENTER
#             df['x'] = (df['x'] - df['x'].mean())
#             df['y'] = (df['y'] - df['y'].mean())
#             df['z'] = (df['z'] - df['z'].mean())
#         else:
#             df['x'] = (df['x'] - df['x'].min()) / max_vals[0]
#             df['y'] = (df['y'] - df['y'].min()) / max_vals[1]
#             df['z'] = (df['z'] - df['z'].min()) / max_vals[2]
    
#     return df
        

# def xyz_to_df_encoded_sorted(base_path, xyz_file, dic_encode_atom_type, lst_atom_types, max_vals = [1,1,1], centered = True):

#     """
#         This function is similar to xyz_to_df. In addition, it sorts xyz by x, then y, then z.
#     """
    
#     with open(os.path.join(base_path, xyz_file)) as f:
#         df = pd.DataFrame(
#             list(map(lambda x: list(dic_encode_atom_type[x.split()[0]]) + \
#                        [float(x.split()[1]), float(x.split()[2]), float(x.split()[3]), x.split()[0]],
#             list(filter(lambda x: (len(x.split()) != 0) and (x.split()[0] in lst_atom_types),
#                           f.readlines()))
#                       )),
#         columns = lst_atom_types + ['x', 'y', 'z', 'atom_type']).sort_values(['x', 'y', 'z'])

#         if (centered):
#             # NORMALISE
#             df['x'] = (df['x'] - df['x'].min()) / max_vals[0]
#             df['y'] = (df['y'] - df['y'].min()) / max_vals[1]
#             df['z'] = (df['z'] - df['z'].min()) / max_vals[2]
#             # CENTER
#             df['x'] = (df['x'] - df['x'].mean())
#             df['y'] = (df['y'] - df['y'].mean())
#             df['z'] = (df['z'] - df['z'].mean())
#         else:
#             df['x'] = (df['x'] - df['x'].min()) / max_vals[0]
#             df['y'] = (df['y'] - df['y'].min()) / max_vals[1]
#             df['z'] = (df['z'] - df['z'].min()) / max_vals[2]

#     return df


def xyz_to_df(base_path, xyz_file, dic_encode_atom_type, lst_atom_types, centered = True):
    
    with open(os.path.join(base_path, xyz_file)) as f:
        df = pd.DataFrame(
            list(map(lambda x: list(dic_encode_atom_type[x.split()[0]]) + \
                       [float(x.split()[1]), float(x.split()[2]), float(x.split()[3]), x.split()[0]],
            list(filter(lambda x: (len(x.split()) != 0) and (x.split()[0] in lst_atom_types),
                          f.readlines()))
                      )),
        columns = lst_atom_types + ['x', 'y', 'z', 'atom_type']) 

        if (centered):
            df['x'] = (df['x'] - df['x'].mean())
            df['y'] = (df['y'] - df['y'].mean())
            df['z'] = (df['z'] - df['z'].mean())
        else:
            df['x'] = (df['x'] - df['x'].min())
            df['y'] = (df['y'] - df['y'].min())
            df['z'] = (df['z'] - df['z'].min())
    
    return df



def xyz_to_RDF(df, xmin, xmax, n_bins):

    from scipy.stats import gaussian_kde

    df['x'] = (df['x'] - df['x'].mean())
    df['y'] = (df['y'] - df['y'].mean())
    df['z'] = (df['z'] - df['z'].mean())

    xyz_arr = df[['x', 'y', 'z']].values
    r_dist = np.linalg.norm(xyz_arr, axis=1)

    bins = np.linspace(xmin, xmax, n_bins)
    r_hist, x_bins_ = np.histogram(r_dist, bins = bins, density = True)

    x_bins = x_bins_[:-1] 
    x_bins[0] = 0.01
    r_hist_r = r_hist / x_bins
    r_hist_r2 = r_hist_r / x_bins
    r_hist_r3 = r_hist_r2 / x_bins
    
    kde = gaussian_kde(r_dist)
    r_density = kde(x_bins)
    r_density_r = r_density  / x_bins
    r_density_r2 = r_density_r  / x_bins
    r_density_r3 = r_density_r2  / x_bins

    trunc = int(0.05 * n_bins)
    r_hist_r[:trunc] = r_hist_r2[:trunc] = r_hist_r3[:trunc] = r_density_r[:trunc] = r_density_r2[:trunc] = r_density_r3[:trunc] = 0.0

    arr_signals = np.vstack((x_bins, r_hist, r_hist_r, r_hist_r2, r_hist_r3, r_density, r_density_r, r_density_r2, r_density_r3)).T
    cols =  ['x_bins', 'hist', 'hist_r', 'hist_r2', 'hist_r3', 'dens', 'dens_r', 'dens_r2', 'dens_r3']

    return pd.DataFrame(arr_signals, columns = cols)
        

def pad_xyz(xyz_arr, max_length = 80, dim = 7):
    
    n,m = xyz_arr.shape
    if n < max_length:        
        zero_pad = np.zeros([max_length-n, dim])
        return np.concatenate((xyz_arr,zero_pad), axis = 0 )
    else:
        return xyz_arr
    

def next_batch(x, y, seq_len, batch_size):
    N = x.shape[0]
    batch_indeces = np.random.permutation(N)[:batch_size]
    x_batch = x[batch_indeces]
    y_batch = y[batch_indeces]
    seq_len_batch = seq_len[batch_indeces]
    return x_batch, y_batch, seq_len_batch


def rotate_xyz(xyz_arr, th_x, th_y, th_z):

    th_rad_x = th_x * pi / 180.0
    th_rad_y = th_y * pi / 180.0
    th_rad_z = th_z * pi / 180.0

    Rx = [[1, 0, 0],
    [0, cos(th_rad_x), -sin(th_rad_x)],
    [0, sin(th_rad_x), cos(th_rad_x)]]

    Ry = [[cos(th_rad_y), 0, sin(th_rad_y)],
    [0, 1, 0],
    [-sin(th_rad_y), 0, cos(th_rad_y)]]

    Rz = [[cos(th_rad_z), -sin(th_rad_z), 0], 
    [sin(th_rad_z), cos(th_rad_z), 0],
    [0, 0, 1]]

    Rxyz = np.matmul(np.matmul(Rx, Ry), Rz)

    x_bar = xyz_arr[:,0].mean()
    y_bar = xyz_arr[:,1].mean()
    z_bar = xyz_arr[:,2].mean()

    xyz_arr[:,0] -= x_bar
    xyz_arr[:,1] -= y_bar
    xyz_arr[:,2] -= z_bar

    return np.matmul(Rxyz, xyz_arr.T).T


def rotate_z(xyz_arr, theta_z):
    
    th_rad = theta_z * pi / 180.0
    Rz = np.array([[cos(th_rad), sin(th_rad)],
                    [-sin(th_rad), cos(th_rad)]])
        
    
    return np.matmul(Rz, xyz_arr.T).T


def expand_data_reg_sample_rand_rotate(scale_exp, df_org, labels, read_path, out_path_strct, out_path_base):
    
    shutil.rmtree(out_path_strct)
    os.mkdir(out_path_strct)
    lst_par_data = []
    
    for file_name in df_org.index:        
        for i in range(scale_exp):
            th_rot = randint(0,180)
            df_xyz = pd.read_csv(os.path.join(out_path_strct, 
                                              f'{file_name[:-4]}_encoded_sorted.csv'))
    
            df_xyz[['x', 'y']] = rotate_z(df_xyz[['x', 'y']].values, th_rot)
    
            file_name_exp = f'{file_name[:-4]}_encoded_sorted_th{th_rot}.csv'
            df_xyz.sort_values(['x', 'y']).to_csv(os.path.join(out_path_strct, file_name_exp),
                                                 index = False)
    
            lst_par_data.append([file_name_exp] + list(df_org[labels].loc[file_name].values))
                            
    pd.DataFrame(lst_par_data, columns=['file_name'] + labels).to_csv(out_path_base, index=False)


def xyz_align_PCA(xyz_arr):

    pca = PCA(n_components = 3)
    pca.fit(xyz_arr)
    return np.matmul(pca.components_, xyz_arr.T).T

def molecule_dimension(xyz_arr):

    pca = PCA(n_components = 3)
    pca.fit(xyz_arr)
    return pca.explained_variance_


def return_neighbour_list(df_xyz, df_bond, atom_type_col = 'atom_type', xyz_cols = ['x', 'y', 'z']):

    """
    df_xyz:         should include the xyz coordinates and a column including atomic type.

    df_bond:        should include the pairwise cut-off distances between pair of atoms. It is a dataframe with columns of atom type and 
                    indexes of atom type. 
                    NOTE: if df_bond contains 1 value (i.e. uniform cut-off is considered), the functions uses in-build numpy functions to search for
                    bonded atoms, which is very much efficient. If df_bond contains more than one value, then manual search is implemented which could be
                    slow for bigger molecules.
    
    atom_type_col:  the column name in df_xyz which defines the atom type.

    xyz_cols:       the column names in df_xyz which define the x, y, z atomic positions.
    """
    
    from scipy.spatial import distance_matrix
    
    num_atoms = len(df_xyz)
    
    xyz = df_xyz[xyz_cols].values
    r_mat = distance_matrix(xyz, xyz)
    adj_mat = np.zeros(r_mat.shape)

    if len(df_bond) == 1:

        rcut = df_bond.values[0,0]
        neigh_idx = np.where(r_mat < rcut)
        for item in zip(neigh_idx[0], neigh_idx[1]): adj_mat[item[0], item[1]] = 1
        
    else:
        for i in range(num_atoms - 1):
            for j in range(i+1, num_atoms):

                ai = df_xyz[atom_type_col].values[i]
                aj = df_xyz[atom_type_col].values[j]
                rcut = df_bond.loc[ai, aj]
                if r_mat[i,j] < rcut:
                    adj_mat[i, j] = 1
                    adj_mat[j, i] = 1
    
    coordination = adj_mat.sum(axis=1)
    return adj_mat, coordination

