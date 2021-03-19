# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:57:56 2021

@author: Kim
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
#import scipy.stats

dir_path = '.' # Set path to directory
os.chdir(dir_path)


targets = [0,1,2,3,4]
param = ['ECA', 'ThickA', 'ECB', 'ThickB', 'ECC']
#############################################################################
#Choose between full range or restricted case
############################################################################
res = 0 # 1 means restricted and anything else is the full range

#for it,itar in zip(targets,param):
key = 0
trgt = key
tar = param[key]

if res == 1:
    full = pd.concat(map(pd.read_csv, ['res_TrueVsPre_Tar'  +str(trgt)+ '_R0.csv', 
                                       'res_TrueVsPre_Tar'  +str(trgt)+ '_R1.csv',
                                       'res_TrueVsPre_Tar'  +str(trgt)+ '_R2.csv',
                                       'res_TrueVsPre_Tar'  +str(trgt)+ '_R3.csv',
                                       'res_TrueVsPre_Tar'  +str(trgt)+ '_R4.csv']
                       ))
    full = full.drop(columns='Unnamed: 0')
else:
    full = pd.concat(map(pd.read_csv, ['TrueVsPre_Tar'  +str(trgt)+ '_R0.csv', 
                                       'TrueVsPre_Tar'  +str(trgt)+ '_R1.csv',
                                       'TrueVsPre_Tar'  +str(trgt)+ '_R2.csv',
                                       'TrueVsPre_Tar'  +str(trgt)+ '_R3.csv',
                                       'TrueVsPre_Tar'  +str(trgt)+ '_R4.csv']
                       ))
    full = full.drop(columns='Unnamed: 0')

full['Dif'] = full[tar] - full['predicted']

true = full[tar].values
predicted = full['predicted'].values

rms = sqrt(mean_squared_error(np.squeeze(true), predicted))
rms

values = np.sort(full[tar].unique())
mean_f = []
std_f = []

for i in values:
    tempvar = full.loc[full[tar] == i].iloc[:,-2].values.mean()
    mean_f.append(tempvar)
mean_f = np.array(mean_f)

for i in values:
    tempvar = full.loc[full[tar] == i].iloc[:,-2].values.std()
    std_f.append(tempvar)
std_f = np.array(std_f)


bad = full.loc[full['Dif'] > std_f.mean()]

#param = ['ECA', 'ThickA', 'ECB', 'ThickB', 'ECC']
x_unit = [' [mS/m]', ' [m]', ' [mS/m]', ' [m]', ' [mS/m]']
xlim = [100,1.5,100,2,100]


# quality of testing as 1:1 plot
textstr = 'RMSE: '+str(np.round(rms,2))

#### PLOT 1 ####

fig = plt.figure(figsize = (10,8))                                                             
ax0 = fig.add_subplot(111) 
ax0.plot(true,predicted, 'o', markerfacecolor='none' ,label='Test Cases') #Test cases

ax0.plot(true,true,label='1:1', color = 'k', linestyle = '--') #1:1 line

ax0.plot(values,mean_f, label = "Mean predicted value",color='k', lw=0.7)
ax0.fill_between(values,mean_f-std_f*2,mean_f+std_f*2,alpha=.3, label='2 std') #2 std fill
ax0.fill_between(values,mean_f-std_f,mean_f+std_f,alpha=.3, label='1 std') #1 std fill

ax0.plot([], [], ' ', label=textstr) #add RMSE to legend

handles,labels = ax0.get_legend_handles_labels()

#handles2 = [handles[0], handles[1], handles[3], handles[4], handles[2]]
#labels2 = [labels[0], labels[1], labels[3], labels[4], labels[2]]

handles2 = [handles[0], handles[1], handles[2], handles[4], handles[5], handles[3]]
labels2 = [labels[0], labels[1], labels[2], labels[4], labels[5], labels[3]]

#ax0.set_ylim(-5,105)
ax0.legend(handles2,labels2)
ax0.set_xlabel('True EC [mS/m]')
ax0.set_ylabel('Predicted EC [mS/m]')
#ax0.set_title('True vs predicted values of ECA for maximum range of all parameters')
#ax0.set_title('Predicting ECA with training on full parameter range')
plt.show()


#fig = plt.figure()
#ax1 = fig.add_subplot()
s1 = 18
i=0
fig, ax1 = plt.subplots(figsize=(8,6))
ax2 = ax1.twiny()

#EC
ax1.hist(bad[param[i]]   ,alpha=1, rwidth=0.85, histtype='step', label='ECA', color='blue')
ax1.hist(bad[param[i+2]] ,alpha=1, rwidth=0.85, histtype='step', label='ECB', color='red') 
ax1.hist(bad[param[i+4]] ,alpha=1, rwidth=0.85, histtype='step', label='ECC', color='magenta')
#Thickness
ax2.hist(bad[param[i+1]] ,alpha=1, rwidth=0.85, histtype='step', linestyle=('solid'), color='green')
ax2.hist(bad[param[i+3]] ,alpha=1, rwidth=0.85, histtype='step', linestyle=('solid'), color='gray')

#Create custom Legend
ax2.plot([], [], '-', label="ECA",    color='blue')
ax2.plot([], [], '-', label="ThickA", color='green')
ax2.plot([], [], '-', label="ECB",    color='red')
ax2.plot([], [], '-', label="ThickB", color='gray')
ax2.plot([], [], '-', label="ECC",    color='magenta')


ax2.legend(prop={'size': 14})
ax1.set_xlabel('Electrical conductivity [mS/m]',fontsize=(s1))
ax1.set_ylabel('Frequency',fontsize=(s1))
ax2.set_xlabel('Thickness [Meter]',fontsize=(s1))
#ax2.set_title('CS',fontsize=(s1))

ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()

