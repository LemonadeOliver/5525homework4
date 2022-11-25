# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 22:01:35 2022

@author: LemonadeOliver
"""
### (4) (ii)
import pandas as pd
import numpy as np
import math as ma
import matplotlib.pyplot as plt

# data cleaning
## detect missing values
data1=pd.read_csv("Boston.csv",header=None)
data1.isnull().sum()
## detect and delete outliers
from scipy import stats
# data0=data1[data1.columns[data1.dtypes=='float64' or data1.columns==9]]
data0=data1.drop(columns=[3,8])
data2=data1[(np.abs(stats.zscore(data0))<4).all(axis=1)]
data2=data2.reset_index(drop=True)


# Construct dataset needed
# Construct dataset Boston50
Thre1=data2[data2.columns[13]].median()
newy=[]
for i in range(len(data2.index)):
    if data2.iloc[i,13]>=Thre1:
        newy.append(1)
    else:
            newy.append(0)
Boston50=data2.drop(columns=13)
Boston50["t"]=newy

# Construct dataset Boston75
Thre2=np.percentile(data2[data2.columns[13]],75)
newy=[]
for i in range(len(data2.index)):
    if data2.iloc[i,13]>=Thre2:
        newy.append(1)
    else:
        newy.append(0)
Boston75=data2.drop(columns=13)
Boston75["t"]=newy

### 1) Naive Bayes Classifier
# For Boston50 and Boston75
# Split up the dataset according to the class Label
# Boston50
Boston50_1=Boston50[Boston50["t"]==1]
Boston50_0=Boston50[Boston50["t"]==0]
# Boston75
Boston75_1=Boston75[Boston75["t"]==1]
Boston75_0=Boston50[Boston75["t"]==0]


# Define error rate matrix
error_rate_Boston50_Bayes=np.zeros((10,5),dtype=float)
error_rate_Boston75_Bayes=np.zeros((10,5),dtype=float)

# randomly choose samples from each class based on 80-20 spliting rate for train set and test set for 10 times
for j in range(10):
    ## Boston50
    # For Class 1
    # Shuffle the dataset in the jth time
    Boston50_1=Boston50_1.sample(frac=1)
    Boston50_1=Boston50_1.reset_index(drop=True)
    # Choose the top 80% of the dataset as the train set
    Boston50_1_80=Boston50_1.iloc[range((np.ceil(len(Boston50_1)*0.8)).astype(int)),:]
    # Choose the rest 20% of the dataset as the test set
    Boston50_1_20=Boston50_1.iloc[Boston50_1.index.difference(Boston50_1_80.index),:]
    
    # For Class 0
    # Shuffle the dataset in the jth time
    Boston50_0=Boston50_0.sample(frac=1)
    Boston50_0=Boston50_0.reset_index(drop=True)
    # Choose the top 80% of the dataset as the train set
    Boston50_0_80=Boston50_0.iloc[range((np.ceil(len(Boston50_0)*0.8)).astype(int)),:]
    # Choose the rest 20% of the dataset as the test set
    Boston50_0_20=Boston50_0.iloc[Boston50_0.index.difference(Boston50_0_80.index),:]
    
    # integrate them to get the train set and test set
    Boston50_train=pd.concat([Boston50_1_80,Boston50_0_80])
    Boston50_train=Boston50_train.sample(frac=1)
    Boston50_train=Boston50_train.reset_index(drop=True)
    
    Boston50_test=pd.concat([Boston50_1_20,Boston50_0_20])
    Boston50_test=Boston50_test.sample(frac=1)
    Boston50_test=Boston50_test.reset_index(drop=True)
    
    
    ## Boston75
    # For Class 1
    # Shuffle the dataset in the jth time
    Boston75_1=Boston75_1.sample(frac=1)
    Boston75_1=Boston75_1.reset_index(drop=True)
    # Choose the top 80% of the dataset as the train set
    Boston75_1_80=Boston75_1.iloc[range((np.ceil(len(Boston75_1)*0.8)).astype(int)),:]
    # Choose the rest 20% of the dataset as the test set
    Boston75_1_20=Boston75_1.iloc[Boston75_1.index.difference(Boston75_1_80.index),:]
    
    # For Class 0
    # Shuffle the dataset in the jth time
    Boston75_0=Boston75_0.sample(frac=1)
    Boston75_0=Boston75_0.reset_index(drop=True)
    # Choose the top 80% of the dataset as the train set
    Boston75_0_80=Boston75_0.iloc[range((np.ceil(len(Boston75_0)*0.8)).astype(int)),:]
    # Choose the rest 20% of the dataset as the test set
    Boston75_0_20=Boston75_0.iloc[Boston75_0.index.difference(Boston75_0_80.index),:]
    
    # integrate them to get the train set and test set
    Boston75_train=pd.concat([Boston75_1_80,Boston75_0_80])
    Boston75_train=Boston75_train.sample(frac=1)
    Boston75_train=Boston75_train.reset_index(drop=True)
    
    Boston75_test=pd.concat([Boston75_1_20,Boston75_0_20])
    Boston75_test=Boston75_test.sample(frac=1)
    Boston75_test=Boston75_test.reset_index(drop=True)
    
    # Take 25%, 50%, 75% and 100% of the train set as the new train set respectively and remain the same test set
    for k in range(5):
        if k==0:
            k=0.4
        # Boston50_train
        Boston50_train_new=Boston50_train.iloc[range((np.ceil(len(Boston50_train)*k*0.25)).astype(int))]
        # Boston75_train
        Boston75_train_new=Boston75_train.iloc[range((np.ceil(len(Boston75_train)*k*0.25)).astype(int))]
        if k==0.4:
            k=0
        # Split the train set and test set into X and Y
        # Boston50
        X_train_50=Boston50_train_new.drop(columns="t").to_numpy()
        Y_train_50=Boston50_train_new["t"].to_numpy()
        X_test_50=Boston50_test.drop(columns="t").to_numpy()
        Y_test_50=Boston50_test["t"].to_numpy()
        # Boston 75
        X_train_75=Boston75_train_new.drop(columns="t").to_numpy()
        Y_train_75=Boston75_train_new["t"].to_numpy()
        X_test_75=Boston75_test.drop(columns="t").to_numpy()
        Y_test_75=Boston75_test["t"].to_numpy()
        
        # Naive Bayes Classifier for Boston50
        # Split the train set into two classes according to the sample labels
        Boston50_train_new_1=Boston50_train_new[Boston50_train_new["t"]==1]
        Boston50_train_new_0=Boston50_train_new[Boston50_train_new["t"]==0]
        # Obtain X from Boston50_train_new_1
        X_train_50_1=Boston50_train_new_1.drop(columns="t").to_numpy()    
        # Obtain X from Boston50_train_new_0
        X_train_50_0=Boston50_train_new_0.drop(columns="t").to_numpy() 
        
        # Compute Class one and Class zero posterior probabilities by Naive Bayes Classifier for each case of Boston50_test
        Y_hat=[]
        for i in range(len(Boston50_test)):
            # Compute the estimate for uj1
            u1=np.mean(X_train_50_1,axis=0)
            # Compute the estimate for sigma^2j1
            si1=np.var(X_train_50_1,axis=0)
            # Compute the estimate for P(C1)
            P1=len(Boston50_train_new_1)/len(Boston50_train_new)
            # Compute the posterior probability of C1 for xi
            si1_multi=1
            H=0
            for x in range(len(si1)):
                si1_multi=si1_multi*si1[x]
                H=H+(X_train_50_1[i,x]-u1[x])**2/(2*si1[x]**2)
            Pos1=P1*1/(((2*ma.pi)**(len(si1)/2))*si1_multi)*ma.e**(-H)
            
            # Compute the estimate for uj0
            u0=np.mean(X_train_50_0,axis=0)
            # Compute the estimate for sigma^2j0
            si0=np.var(X_train_50_0,axis=0)
            # Compute the estimate for P(C0)
            P0=len(Boston50_train_new_0)/len(Boston50_train_new)
            # Compute the posterior probability of C0 for xi
            si0_multi=1
            H=0
            for x in range(len(si1)):
                si0_multi=si0_multi*si0[x]
                H=H+(X_train_50_0[i,x]-u0[x])**2/(2*si0[x]**2)
            Pos0=P0*1/(((2*ma.pi)**(len(si0)/2))*si0_multi)*ma.e**(-H)
            
            # Classify xi into 0 or 1
            if Pos1>Pos0:
                Y_hat.append(1)
            else:
                Y_hat.append(0)
        Y_hat=np.array(Y_hat)
            
        # compute the error rate
        error_num=0
        for i in range(len(X_test_50)):
            if Y_hat[i]!=Y_test_50[i]:
                    error_num=error_num+1
        error_rate_test=error_num/len(X_test_50)
        error_rate_Boston50_Bayes[j,k]=error_rate_test
        
        # Naive Bayes Classifier for Boston75
        # Split the train set into two classes according to the sample labels
        Boston75_train_new_1=Boston75_train_new[Boston75_train_new["t"]==1]
        Boston75_train_new_0=Boston75_train_new[Boston75_train_new["t"]==0]
        # Obtain X from Boston75_train_new_1
        X_train_75_1=Boston75_train_new_1.drop(columns="t").to_numpy()    
        # Obtain X from Boston75_train_new_0
        X_train_75_0=Boston75_train_new_0.drop(columns="t").to_numpy() 
        
        # Compute Class one and Class zero posterior probabilities by Naive Bayes Classifier for each case of Boston75_test
        Y_hat=[]
        for i in range(len(Boston75_test)):
            # Compute the estimate for uj1
            u1=np.mean(X_train_75_1,axis=0)
            # Compute the estimate for sigma^2j1
            si1=np.var(X_train_75_1,axis=0)
            # Compute the estimate for P(C1)
            P1=len(Boston75_train_new_1)/len(Boston75_train_new)
            # Compute the posterior probability of C1 for xi
            si1_multi=1
            H=0
            for x in range(len(si1)):
                si1_multi=si1_multi*si1[x]
                H=H+(X_train_75_1[i,x]-u1[x])**2/(2*si1[x]**2)
            Pos1=P1*1/(((2*ma.pi)**(len(si1)/2))*si1_multi)*ma.e**(-H)
            
            # Compute the estimate for uj0
            u0=np.mean(X_train_75_0,axis=0)
            # Compute the estimate for sigma^2j0
            si0=np.var(X_train_75_0,axis=0)
            # Compute the estimate for P(C0)
            P0=len(Boston75_train_new_0)/len(Boston75_train_new)
            # Compute the posterior probability of C0 for xi
            si0_multi=1
            H=0
            for x in range(len(si1)):
                si0_multi=si0_multi*si0[x]
                H=H+(X_train_75_0[i,x]-u0[x])**2/(2*si0[x]**2)
            Pos0=P0*1/(((2*ma.pi)**(len(si0)/2))*si0_multi)*ma.e**(-H)
            
            # Classify xi into 0 or 1
            if Pos1>Pos0:
                Y_hat.append(1)
            else:
                Y_hat.append(0)
        Y_hat=np.array(Y_hat)
            
        # compute the error rate
        error_num=0
        for i in range(len(X_test_75)):
            if Y_hat[i]!=Y_test_75[i]:
                    error_num=error_num+1
        error_rate_test=error_num/len(X_test_75)
        error_rate_Boston75_Bayes[j,k]=error_rate_test
        
        
 

            

            