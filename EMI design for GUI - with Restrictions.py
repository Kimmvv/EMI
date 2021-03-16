#!/usr/bin/env python
# coding: utf-8


#from sys import exit
import time
start = time.time()
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
import random
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from math import sqrt
#from sklearn.preprocessing import normalize
import scipy.stats
import pandas as pd
from emagpy import invertHelper as ih

#Edit filenames and directory path
prefix = "_" #Added infront of filesname
suffix = "_" #Added behind filesname

dir_path="."      # Set path to directory

mean_rmse_train = []
mean_rmse_test = []
mean_R2_train = []
mean_R2_test = []
mean_imp = []
var_imp = []

    
tic=time.perf_counter()

for i in np.arange(10):
    print()
    
###############################################################################
# Allow user to define limits of each parameter in the EMImatrix
# Define parameter values by dividing range into numsteps equal steps
###############################################################################
numparamsteps=10

ECA_max=100
ECA_min=1
ECB_max=100
ECB_min=1
ECC_max=100
ECC_min=1
thickA_max=1.5   
thickA_min=0.05
thickB_max=2.0
thickB_min=0.1

ECA_step=(ECA_max-ECA_min)/(numparamsteps-1)
ECA=np.arange(ECA_min,ECA_max+ECA_step,ECA_step)
ECB_step=(ECB_max-ECB_min)/(numparamsteps-1)
ECB=np.arange(ECB_min,ECB_max+ECB_step,ECB_step)
ECC_step=(ECC_max-ECC_min)/(numparamsteps-1)
ECC=np.arange(ECC_min,ECC_max+ECC_step,ECC_step)

#thickA_step=(thickA_max-thickA_min)/(numparamsteps-1)
#thickA=np.arange(thickA_min,thickA_max+thickA_step,thickA_step)
thickA=np.linspace(thickA_min,thickA_max,numparamsteps)

#thickB_step=(thickB_max-thickB_min)/(numparamsteps-1)
#thickB=np.arange(thickB_min,thickB_max+thickB_step,thickB_step)

thickB = np.linspace(thickB_min,thickB_max,numparamsteps)
thickB_step = thickB[1] - thickB[0]

###############################################################################
# Allow user to choose which EMI setups to include in design 
# Allow user to define range of instrument heights
# Instrument heights vary linearly between limits with num_inst_heights values
###############################################################################
num_inst_height=3
inst_height_max=0.5
inst_height_min=0.1

inst_height_step=(inst_height_max-inst_height_min)/(num_inst_height-1)
inst_height=np.arange(inst_height_min,inst_height_max+inst_height_step,inst_height_step)

num_ant_sep=3
ant_sep_max=4
ant_sep_min=1

ant_sep_step=(ant_sep_max-ant_sep_min)/(num_ant_sep-1)
ant_sep=np.arange(ant_sep_min,ant_sep_max+ant_sep_step,ant_sep_step)

hcp=1                                                                          # 1=consider, 0=exclude
vcp=1
prp=1
ant_orients=['hcp','vcp','prp']
if prp==0:
    ant_orients.remove('prp')
if vcp==0:
    ant_orients.remove('vcp')
if hcp==0:
    ant_orients.remove('hcp')

###############################################################################
# Allow user to restrict the parameter values of EMI matrix AFTER it has been 
# created, but PRIOR to running the machine learning 
###############################################################################

ECA_high    = 0
ECA_low     = 0
ECB_high    = 0
ECB_low     = 0
ECC_high    = 0
ECC_low     = 0
thickA_high = 0
thickA_low  = 0
thickB_high = 0
thickB_low  = 0

###############################################################################
# Allow user to configure the machine learning parameters
###############################################################################

numrepeats=5                                                                   # number of times to repeat training to test stabilty of results
regressionmodel=1                                                              # 0=regression tree, 1=gradient boosting, 2=random forest
min_max_scaler = preprocessing.MinMaxScaler()
max_depth=10                                                                   # maximum number of levels of tree
min_samples_leaf=2                                                             # splits per node
importancethreshold=0  

###############################################################################
# Calculate EMI response for every instrument configuration and parameter combination
###############################################################################
print('Calculating EMI responses')
print()
startEMI = time.time()

arraysize1 = np.shape(ECA)[0]*np.shape(thickA)[0]*np.shape(ECB)[0]*np.shape(thickB)[0]*np.shape(ECC)[0]
arraysize2 = np.shape(inst_height)[0]*np.shape(ant_sep)[0]*np.shape(ant_orients)[0]
EMImatrix = np.empty((arraysize1, arraysize2+5))
obstype=[]
count0=-1
count3=-1
IDmatrix=np.zeros((np.shape(ant_orients)[0], np.shape(ant_sep)[0],np.shape(inst_height)[0]))
for orientation in ant_orients:
    count0+=1
    count1=-1
    for sep in ant_sep:
        count1+=1
        count2=-1        
        for height in inst_height:
            count2+=1
            count3+=1
            tempname=orientation+'_'+str(np.round(sep,3))+'_'+str(np.round(height,3))
            obstype.append(tempname)
            IDmatrix[count0,count1,count2]=count3

counter2=-1
count=0
for EC3 in ECC:
    for th2 in thickB:
        for EC2 in ECB:
            for th1 in thickA:   
                for EC1 in ECA:
                    counter2+=1
                    counter1=-1
                    for orientation in ant_orients:
                        for sep in ant_sep:
                            sep_list = [float(sep)]
                            for height in inst_height:
                                orientation_list = [str(orientation)]
                                counter1+=1  

                                cond = [EC1, EC2, EC3]
                                depths = [th1, th1+th2]                                   
                                count+=1
                                EMI = ih.fCS(np.array(cond), depths, s=sep_list, cpos=orientation_list, hx=height)

                                EMImatrix[counter2,0] = EC1                    # [mS/m]
                                EMImatrix[counter2,1] = th1                    # [m]
                                EMImatrix[counter2,2] = EC2                    # [mS/m]
                                EMImatrix[counter2,3] = th2                    # [m]
                                EMImatrix[counter2,4] = EC3                    # [mS/m]
                                EMImatrix[counter2,5+counter1] = EMI           # [mS/m]      
        
print('Forward model run time[s]: ' + str(np.round(time.time()- startEMI,2)))

############################################################################## 
# Allow user to export EMImatrix  
##############################################################################

xtext = ['ECA', 'ThickA', 'ECB', 'ThickB', 'ECC']
cols = xtext + obstype   
emi = pd.DataFrame(EMImatrix, columns = cols)
emi.to_csv('EMImatrix.csv')

###############################################################################
# Train ML to results
###############################################################################
start_ml = time.time()
mean_param = [] 

restrict_matrix = EMImatrix
if ECA_low > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,0] >= ECA_low] 
if ECA_high > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,0] <= ECA_high] 

if ECB_low > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,2] >= ECB_low] 
if ECB_high > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,2] <= ECB_high] 
   
if ECC_low > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,4] >= ECC_low] 
if ECC_high > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,4] <= ECC_high] 

if thickA_low > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,1] >= thickA_low] 
if thickA_high > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,1] <= thickA_high] 

if thickB_low > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,3] >= thickB_low] 
if thickB_high > 0:
   restrict_matrix = restrict_matrix[restrict_matrix[:,3] <= thickB_high] 

means = [np.mean(restrict_matrix[:,0]), np.mean(restrict_matrix[:,1]), np.mean(restrict_matrix[:,2]), np.mean(restrict_matrix[:,3]), np.mean(restrict_matrix[:,4])]
mean_param.append(np.round(means,3))
   
print('training ML')
print()


data3 = restrict_matrix[:,5:]

rmse_all_train_wrpt = np.zeros((numrepeats,5))
rmse_all_test_wrpt = np.zeros((numrepeats,5))
R2_all_train_wrpt=np.zeros((numrepeats,5))
R2_all_test_wrpt=np.zeros((numrepeats,5))
importance_all_wrpt=np.zeros((numrepeats,5,np.shape(obstype)[0]))

for rpt in np.arange(numrepeats):
    print()
    print('repeat ',rpt)
    for trgt in np.arange(5):
        trgt_list = [int(trgt)]
        print('target ',trgt)
        
        data2 = restrict_matrix[:,:5]                                                   # EC_A, Thick_A, EC_B, Thick_B and EC_C
        target=data2[:,trgt_list]                                                       # define specific target to predict and data to consider
        
        if np.ndim(target)==1:                                                          # do some gymnastics to concatentate features and targets allowing for only a single target
            addcols=1
        else:
            addcols=np.shape(target)[1]
        data=np.zeros((np.shape(data3)[0],np.shape(data3)[1]+addcols))
        data[:,:-addcols]=data3
        data[:,-addcols:]=target
        
        Predictions_use=data[:,np.shape(data)[1]-addcols:]                                                   # normalize target for ML analysis and save values to recover unnormalized values later
        Predictions_usemean=np.mean(Predictions_use,axis=0)
        Predictions_userange=np.max(Predictions_use,axis=0)-np.min(Predictions_use,axis=0)
        Predictions_usemin=np.min(Predictions_use,axis=0)
        Predictions_usestd=np.std(Predictions_use,axis=0)    
        Predictions_use_norm = min_max_scaler.fit_transform(Predictions_use)           # normalizing flux DOES appear to be necessary, have to remember to un-normalize for later plots
        
        
        #data2frame=np.hstack((data[:,:-1],Predictions_use_norm))                       # replace targets with normalized targets
        data2frame = data.copy()
        data2frame[:,np.shape(data)[1]-addcols:] = Predictions_use_norm
        
        
        fract_train=0.7                                                                # define fraction of data to use for training
        numtrain=int(fract_train*np.shape(data)[0])                                    # find number of training points
        train_t=np.sort(random.sample(range(np.shape(data)[0]), numtrain))             # find times of training points
        test_t=np.arange(np.shape(data)[0])                                            # find times of testing points
        test_t = [i for i in test_t if i not in train_t]
                
        training_data=data2frame[train_t,:]                                           # extract training data 
        testing_data=data2frame[test_t,:]                                             # extract testing data 
        
        training_data_pd = pd.DataFrame(training_data)                                 # put training data into pandas
        #testing_data_pd = pd.DataFrame(testing_data)                                   # put testing data into pandas
        
#        training_data_pd = training_data                                                 # put training data into pandas
#        testing_data_pd = testing_data
#        
#        rmse_train_hold=np.zeros(max_depth)                                            # set aside matrices to hold output
#        rmse_test_hold=np.zeros(max_depth)
        
        #        importance_hold=np.zeros((max_depth,max_depth))
            
        
        if regressionmodel==0:                                                         # run machine learning analysis
            regression_model = DecisionTreeRegressor(criterion="mse",max_depth=max_depth, min_samples_leaf=min_samples_leaf)                      # MSE == varince is spliting criteria, minimum instances per leaf = 5
            regression_model.fit(training_data[:,:-addcols],training_data[:,np.shape(data)[1]-addcols:])
        elif regressionmodel==1:    
            regression_model = ensemble.GradientBoostingRegressor(criterion="mse",max_depth=max_depth, min_samples_leaf=min_samples_leaf)                      # MSE == varince is spliting criteria, minimum instances per leaf = 5
            regression_model.fit(training_data[:,:-addcols],training_data[:,np.shape(data)[1]-addcols:])
        else:    
            regression_model = ensemble.RandomForestRegressor(criterion="mse",max_depth=max_depth, min_samples_leaf=min_samples_leaf)                      # MSE == varince is spliting criteria, minimum instances per leaf = 5
            regression_model.fit(training_data[:,:-addcols],training_data[:,np.shape(data)[1]-addcols:])
                
        predicted_train = regression_model.predict(training_data[:,:-addcols])        # calculate predictions for training
        predicted_train = predicted_train*Predictions_userange[0]+Predictions_usemin[0] # un-normalize predictions for training
        true_train=Predictions_use[train_t]                                        # store correct target values for training
        predicted = regression_model.predict(testing_data[:,:-addcols])               # calculate predictions for testing
        predicted = predicted*Predictions_userange[0]+Predictions_usemin[0]             # un-normalize predictions for testing
        true=Predictions_use[test_t]                                               # store correct target values for testing
        
        importance = regression_model.feature_importances_                              # find feature importance values
        
        slope, intercept, r_value_train, p_value, std_err = scipy.stats.linregress(np.squeeze(true_train), predicted_train)    
        R2_train=r_value_train**2
        rmse_train = sqrt(mean_squared_error(np.squeeze(true_train), predicted_train))
        
        slope, intercept, r_value_test, p_value, std_err = scipy.stats.linregress(np.squeeze(true), predicted)    
        R2_test=r_value_test**2
        rmse_test = sqrt(mean_squared_error(np.squeeze(true), predicted))
        
        #####################################################################
        # Saves the true and predicted values from each repitition
        #####################################################################
        temp_df = pd.DataFrame(EMImatrix[test_t],columns=cols)
        temp_df['predicted'] = predicted
        temp_df.to_csv(os.path.join(dir_path,'TrueVsPre_Tar'+str(trgt)+'_R'+str(rpt)+'.csv'))
        
        rmse_all_train_wrpt[rpt,trgt]=rmse_train 
        rmse_all_test_wrpt[rpt,trgt]=rmse_test
        R2_all_train_wrpt[rpt,trgt]=R2_train
        R2_all_test_wrpt[rpt,trgt]=R2_test
        importance_all_wrpt[rpt,trgt,:]=importance
  
mean_rmse_all_train=np.mean(rmse_all_train_wrpt,axis=0)
mean_rmse_all_test=np.mean(rmse_all_test_wrpt,axis=0)    
mean_R2_all_train=np.mean(R2_all_train_wrpt,axis=0)
mean_R2_all_test=np.mean(R2_all_test_wrpt,axis=0)
mean_importance_all=np.mean(importance_all_wrpt,axis=0)
var_importance_all=np.var(importance_all_wrpt,axis=0)


mean_rmse_train.append(mean_rmse_all_train)
mean_rmse_test.append(mean_rmse_all_test)

mean_R2_train.append(mean_R2_all_train)
mean_R2_test.append(mean_R2_all_test)
mean_imp.append(mean_importance_all)
var_imp.append(var_importance_all)
        
end = time.time()
print(np.round(end-start_ml,2))
##############################################################################
# Save outputs
##############################################################################
   
xtext = ['EC_A', 'Thick_A', 'EC_B', 'Thick_B', 'EC_C']

#Put results into dataframes
rms_test = pd.DataFrame(mean_rmse_test, columns = xtext)
rms_train = pd.DataFrame(mean_rmse_train, columns = xtext)
R2_test = pd.DataFrame(mean_R2_test, columns = xtext)
R2_train = pd.DataFrame(mean_R2_train, columns = xtext)
means = pd.DataFrame(mean_param, columns = xtext)

#Export dataframes to csv files
#R²
R2_train.to_csv(os.path.join(dir_path,prefix+'R2_train_'+suffix+'.csv'))
R2_test.to_csv(os.path.join(dir_path,prefix+'R2_test_'+suffix+'.csv'))

#Mean parameter values
means.to_csv(os.path.join(dir_path,prefix+'mean_'+suffix+'.csv'))

#RMSE
rms_train.to_csv(os.path.join(dir_path,prefix+'rms_train_'+suffix+'.csv'))
rms_test.to_csv(os.path.join(dir_path,prefix+'rms_test_'+suffix+'.csv'))

#Mean importance
temp_var = pd.DataFrame(mean_imp[0], xtext, columns = obstype)
temp_var.to_csv(os.path.join(dir_path,prefix+'Mean_imp_'+suffix+'.csv'))




