# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:08:54 2020

@author: Kim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import exit
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import ListedColormap

dir_path = '.'
sep = ","

col = ['HCP_1.0_0.1', 'HCP_1.0_0.3', 'HCP_1.0_0.5',
       'HCP_2.5_0.1', 'HCP_2.5_0.3', 'HCP_2.5_0.5',
       'HCP_4.0_0.1', 'HCP_4.0_0.3', 'HCP_4.0_0.5',
       
       'VCP_1.0_0.1', 'VCP_1.0_0.3', 'VCP_1.0_0.5',
       'VCP_2.5_0.1', 'VCP_2.5_0.3', 'VCP_2.5_0.5',
       'VCP_4.0_0.1', 'VCP_4.0_0.3', 'VCP_4.0_0.5',
       
       'PRP_1.0_0.1', 'PRP_1.0_0.3', 'PRP_1.0_0.5',
       'PRP_2.5_0.1', 'PRP_2.5_0.3', 'PRP_2.5_0.5',
       'PRP_4.0_0.1', 'PRP_4.0_0.3', 'PRP_4.0_0.5',]

df = pd.DataFrame(columns=col)
cfg = df.columns

param = [0,1,2,3,4]
par_name = ['ECA', 'ThickA', 'ECB', 'ThickB', 'ECC']
idx = ['full','cen','sl','sh']


full = pd.concat(map(pd.read_csv, ['Mean_imp_ThickA_full.csv', 
                                 'Mean_imp_ThickA_full.csv',
                                 'Mean_imp_ThickA_full.csv',
                                 'Mean_imp_ThickA_full.csv',
                                 'Mean_imp_ThickA_full.csv']
                   ))
full = full.drop(columns='Unnamed: 0')
full = full.rename(index={0: 'ECA', 1: 'ThickA', 2: 'ECB', 3: 'ThickB', 4: 'ECC' })

cen = pd.concat(map(pd.read_csv, ['Mean_imp_ECA_cen.csv', 
                                 'Mean_imp_ThickA_cen.csv',
                                 'Mean_imp_ECB_cen.csv',
                                 'Mean_imp_ThickB_cen.csv',
                                 'Mean_imp_ECC_cen.csv']
                   ))
cen = cen.drop(columns='Unnamed: 0')
cen = cen.rename(index={0: 'ECA', 1: 'ThickA', 2: 'ECB', 3: 'ThickB', 4: 'ECC' })

sl_ = pd.concat(map(pd.read_csv, ['Mean_imp_ECA_skewed.csv', 
                                 'Mean_imp_ThickA_skewed.csv',
                                 'Mean_imp_ECB_skewed.csv',
                                 'Mean_imp_ThickB_skewed.csv',
                                 'Mean_imp_ECC_skewed.csv']
                   ))
sl_ = sl_.drop(columns='Unnamed: 0')
sl_ = sl_.rename(index={0: 'ECA', 1: 'ThickA', 2: 'ECB', 3: 'ThickB', 4: 'ECC' })

sh_ = pd.concat(map(pd.read_csv, ['Mean_imp_ECA_skew_high.csv', 
                                 'Mean_imp_ThickA_skew_high.csv',
                                 'Mean_imp_ECB_skew_high.csv',
                                 'Mean_imp_ThickB_skew_high.csv',
                                 'Mean_imp_ECC_skew_high.csv']
                   ))
sh_ = sh_.drop(columns='Unnamed: 0')
sh_ = sh_.rename(index={0: 'ECA', 1: 'ThickA', 2: 'ECB', 3: 'ThickB', 4: 'ECC' })

#cfg = full.columns #Get Cfg names from column

sps = [0,0,0,0,0,
       1,1,1,1,1,
       2,2,2,2,2,
       3,3,3,3,3,
       4,4,4,4,4]

sps2 = [0,1,2,3,4,
        0,1,2,3,4,
        0,1,2,3,4,
        0,1,2,3,4,
        0,1,2,3,4]

param = np.linspace(0,24,25,dtype=int)
names = ['ECA', 'ThickA', 'ECB', 'ThickB', 'ECC']
par_name = names * 5
idx = ['full','cen','sl','sh']

fig = plt.figure(figsize = (12,11))
spec = gridspec.GridSpec(ncols=5, nrows=5, wspace = 0.2, hspace=0.4)
axs = []

for ipar,sps,sps2 in zip(param,sps,sps2):
    par = ipar #0=ECA, 1=ThickA, 2=ECB, 3=ThickB, 4=ECC
    
    f =  full.iloc[par,:].values
    c =  cen.iloc[par,:].values
    sl = sl_.iloc[par,:].values
    sh = sh_.iloc[par,:].values
    
    imps = pd.DataFrame([f,c,sl,sh],columns = cfg, index = idx)
    mean = imps.mean() #Mean of each column
    
    p_val = 8      #Values to plot seperately
    o_val = 27-p_val #Values to aggregate into "others"
    # sum of p_val & o_val must be 27
    
    if p_val + o_val != 27:
        print("Values are not equal 27")
        exit(0)
    
    
    most_imp = np.argsort(-np.asarray(mean))[:p_val] # The # most important feats - based on mean imp over all 4 restriction patterns
    other = cfg[np.argsort(-np.asarray(mean))[-o_val:]] #Other bin
    
    #Re-sort the most imp feats based on the order the appear originally so the coil pos are next to each other 
    most_imp_sort = np.sort(most_imp) 
    
    #Full
    f_other_val = f[np.argsort(-np.asarray(mean))[-o_val:]].sum()
    full_val = f[most_imp_sort]
    full_val= np.append(full_val,f_other_val)
    
    #Centered
    c_other_val = c[np.argsort(-np.asarray(mean))[-o_val:]].sum()
    c_val = c[most_imp_sort]
    c_val= np.append(c_val,c_other_val)
    
    #Skew low
    sl_other_val = sl[np.argsort(-np.asarray(mean))[-o_val:]].sum()
    sl_val = sl[most_imp_sort]
    sl_val= np.append(sl_val,sl_other_val)
    
    #Skew high
    sh_other_val = sh[np.argsort(-np.asarray(mean))[-o_val:]].sum()
    sh_val = sh[most_imp_sort]
    sh_val= np.append(sh_val,sh_other_val)
    
    #Labels
    label = list(cfg[most_imp_sort])
    label.append('others')
    
    ##################################################################################
    #                       Custom colourmap                    
    ##################################################################################
    
    blue = cm.get_cmap('Blues',9)
    red = cm.get_cmap('Reds',9)
    green = cm.get_cmap('Greens',9)
    
    low, mid, high = 7,5,2
    white = (1.0, 1.0, 1.0, 1.0)
    hatches = [ "" , "//" , "O" , "" , "//" , "O" ,"" , "//" , "O" ,"" , "//" , "O" ,"" , "//" , "O" ,"" , "//" , "O" ,"" , "//" , "O" ,"" , "//" , "O","" , "//" , "O","" ]
    clr = [blue(low),blue(low),blue(low),blue(mid),blue(mid),blue(mid),blue(high),blue(high),blue(high),
           green(low),green(low),green(low),green(mid),green(mid),green(mid),green(high),green(high),green(high),
           red(low),red(low),red(low),red(mid),red(mid),red(mid),red(high),red(high),red(high),
           white]
   
    
    newcolor =  []
    for i in most_imp_sort:
        newcolor.append(clr[i])
    newcolor.append(white) #Other category is always white
    
    newhatches = []
    for j in most_imp_sort:
        newhatches.append(hatches[j])
    newhatches.append("")
    
    cmap = ListedColormap(newcolor,N=p_val)
    
    
    ############################# With hatches ######################################
    #fig1, f1_axes = plt.subplots(ncols=5, nrows=1, constrained_layout=True)
    
    vals = [sh_val,sl_val,c_val,full_val]
    size = 0.3
    radius = [1+size*2,1+size,1,1-size]
    
    axs.append(fig.add_subplot(spec[sps, sps2]))
    
    for ival,ir in zip(vals,radius):
        piechart = axs[-1].pie(ival, radius = ir, wedgeprops=dict(width=0.3, edgecolor='k'), colors=newcolor)
        for i in range(len(piechart[0])):
            piechart[0][i].set_hatch(newhatches[(i)%len(newhatches)])
    
    #axs[-1].text(-0.1,0.0,par_name[ipar])
    #ax.legend(loc='upper left', labels=label,prop={'size': 12}, bbox_to_anchor=(-0.1, 1), bbox_transform=fig.transFigure)
    if sps == 0:
        axs[-1].set_title(par_name[ipar],y=1.15)
    
    if sps2 == 0:
         axs[-1].text(x=-2.7, y=-0.1, s = par_name[ipar+sps]  )
    
    if sps == 0 and sps2 == 2:
        plt.text(x=0.0, y=2.4, s = 'Inferred parameters',horizontalalignment='center', fontsize = 23)
        
    if sps == 2 and sps2 == 0:
        plt.text(x=-3.4, y=0.0, s = 'Restricted parameters',verticalalignment='center', fontsize = 23,rotation=90)    
    #plt.show()


patches = []
conf = list(cfg)
conf.append('others')
for ic,icfg,ih in zip(clr,conf,hatches):
    patches.append(mpatches.Patch(facecolor=ic, hatch=ih, label=icfg))
    
plt.legend(handles=patches, loc='upper left',prop={'size': 12}, bbox_to_anchor=(-0.1, 0.91), bbox_transform = fig.transFigure)    
plt.show()    



