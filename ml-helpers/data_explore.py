"""
Created on Wed Apr 10 10:04:29 2019

@author: Ben Motevalli (b.motevalli@gmail.com)
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

import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import shutil

import os


def list_missing_data_cols(df):
    """
    This function reports statistics on missing data in the datasets.
    """
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

    """
    This function show the balance in categorical columns. 
    One useful case is for checking the clusters balances in the data.
    """
    
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
    This function shows the balance in each column of dataframe.
    For each column, it returns the list of unique values together with
    the percentange of each value in the dataset.
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


def report_unbalance_cols(df, columns, thresh = 90):
    """
    This function reports columns that are highly unbalanced.
    One use case is for columns with lots of missing data.

    @thresh:    is in percentange
    """
    
    import numpy as np
    
    n_rows = len(df)
    
    report_cols = []
    for col in columns:
        unq_vals = df[col].unique()
        if (len(unq_vals) < 50):
            for val in unq_vals:
                share = (len(df[df[col] == val]) / n_rows * 100.0)
                if ((len(df[df[col] == val]) / n_rows * 100.0) > thresh):
                    report_cols.append(col)

    return report_cols


def identify_cat_cols(df, columns, thresh=2):
    """
    This function identifies which columns could be categorical.
    It calculates the ratio of number of unique values to total number of records.

    @thresh:    is in percentange.
    """
    shared = share_percentage(df, columns)
    cat_cols = []
    for k,v in shared.items():
        per = len(v.keys()) / len(df) * 100
        if (per <= thresh):
            cat_cols.append({'name': k, 'per': per})

    return cat_cols, shared


def greedy_elimination(corr, col_list, main_label = None, thresh_val = 75):
    """
    This function helps in reducing number of features by eliminating
    how highly they are correlated.

    @thresh_val:    defines the criteria above which correlated columns are 
                    eliminated. Value should be provided in percentange.
    """    
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
    """
    This function is a quick way to check label column against feature columns
    in a scatter plot.

    Note:   numerical columns and labels are suited for this purpose.
    """
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


def barChart_label_vs_cat_feats(df, cols_cat, label, lst_tags = None, lst_colors = None, n_row=3, n_col=3, num_fig_to_show = None):
    """
    This function plots a label distribution against different categorical features.
    """
    for i, c in enumerate(cols_cat):    
    
        df_agg = df_train.groupby(c).agg({label:lambda x: x.mean()}).reset_index()

        if not (num_fig_to_show == None):
            if (i == num_fig_to_show):
                break

        if(i % tot_subs == 0):
            count = 1
            plt.figure(figsize=figsize)

        plt.subplot(n_row,n_col,count)

        if (lst_tags == None):
            plt.bar(df_agg[c], df_agg[label])    
        else:
            for i, tag in enumerate(lst_tags):
                plt.bar(df_agg[c],df_agg[label], color = lst_colors[i])

        plt.xlabel(c)
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



