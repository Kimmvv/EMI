# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:35:24 2020

@author: au606808
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

parameters = ['ECA', 'ThickA', 'ECB', 'ThickB', 'ECC']
colors = ['blue','green','red','0.50','purple']
linecolor = 'k'
figsize = (8,6)
ylim = (0,0.3)


dir_path = '.'       #home
sep = ","

rmse_all = pd.read_csv(os.path.join(dir_path,'sunbursts.csv'),sep=';')
rmse_full = pd.read_csv(os.path.join(dir_path,'RMSE_test_Full_range.csv'),sep=',')

prange = [99, 1.45, 99, 1.9, 99]
full = rmse_full.values / prange

 
frac = [0.11]
rmse = rmse_all[rmse_all['Frac'].isin(frac)] 
#rmse = rmse_all[(rmse_all['Frac'] == frac)]
sl  = rmse[rmse['Pattern'].isin(['SL'])] 
cen = rmse[rmse['Pattern'].isin(['Cen'])] 
sh  = rmse[rmse['Pattern'].isin(['SH'])] 


#full =  rmse_full.iloc [0,:].values

x = [0.9,1.9,2.9,3.9,4.9,
     1.0,2.0,3.0,4.0,5.0,
     1.1,2.1,3.1,4.1,5.1]

xl = [0.8,1.8,2.8,3.8,4.8]
xc = [1.0,2.0,3.0,4.0,5.0]
xh = [1.2,2.2,3.2,4.2,5.2]

pattern = ['SL','Cen','SH']

yl = []
for iy in parameters:
    yl.append(sl.loc[(sl['Constrain'] == iy)]['rmse_range'].values)

yc = []
for iy in parameters:
    yc.append(cen.loc[(cen['Constrain'] == iy)]['rmse_range'].values)

yh = []
for iy in parameters:
    yh.append(sh.loc[(sh['Constrain'] == iy)]['rmse_range'].values)

fig = plt.figure(figsize=figsize)

for ip, ilbl in zip(range(len(parameters)),parameters):
    plt.scatter(xl,yl[ip],color = colors[ip] )

for ip, ilbl in zip(range(len(parameters)),parameters):
    plt.scatter(xc,yc[ip], color = colors[ip] )

for ip, ilbl in zip(range(len(parameters)),parameters):
    plt.scatter(xh,yh[ip],label =ilbl, color = colors[ip] )

for il in range(4):
    plt.plot([xl[il], xh[il]], [full[0][il], full[0][il]], color=linecolor, linestyle='-', linewidth=2) 
   
plt.plot([xl[4], xh[4]], [full[0][4], full[0][4]], color=linecolor, linestyle='-', linewidth=2, label = 'Full range')
   

plt.ylim(ylim)
plt.xticks(xc, parameters)
plt.legend(title = 'Constrained par')
plt.xlabel('Inferred parameter')
plt.ylabel('RMSE / full parameter range')
#plt.title(label = 'Fraction of constrained parameter range: ' + str(frac[0]))
plt.show()















