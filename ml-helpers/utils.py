import numpy as np
import matplotlib.pyplot as plt


def set_matplotlib_font_size(SMALL_SIZE = 8, MEDIUM_SIZE = 10, BIGGER_SIZE = 12):
    """
    Helper function to set matplotlib's fontsize
    """

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



"""
Below are mainly plotting helpers for archetype package.
"""
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

  
def funcLinear(x, a, b):
    return a*x + b


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
