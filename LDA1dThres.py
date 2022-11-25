# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:38:13 2022

@author: LemonadeOliver
"""
import pandas as pd
import numpy as np

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

################## 3 (1)
# 10-fold Cross-validation to compute average error rate and Standard Deviation for 1-d FDA

# produce the indexes of testset for 10-fold cross-validation
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
Index=list(split(range(len(Boston50.index)), 10))

# Shuffle the dataset
Boston50=Boston50.sample(frac=1)

# define the error rate array for 10 train sets
error_rate_array_train=np.zeros((10,1),dtype=float)
# define the error rate array for 10 test sets
error_rate_array_test=np.zeros((10,1),dtype=float)

# 10-fold Cross-Validation
for j in range(len(Index)):
    # obtain the ith train set
    Boston50_train=Boston50.iloc[Boston50.index.difference(Index[j]),:]
    # obtain the ith test set
    Boston50_test=Boston50.iloc[Index[j],:]
    # divide the ith train set into two datasets based on the class label
    Boston50x=Boston50_train.drop(columns="t")
    Boston50y=Boston50_train["t"]
    x1=Boston50x[Boston50_train["t"]==1]
    x0=Boston50x[Boston50_train["t"]==0]
    X=Boston50x.to_numpy()
    Y=Boston50y.to_numpy()
    X1=x1.to_numpy()
    X0=x0.to_numpy()
    # obtain the mean of X1 and X0
    X1M=np.zeros((1,13),dtype=float)
    for i in range(len(x1)):
       X1M=X1M+X1[i,:]
    X1M=X1M/(len(x1))

    X0M=np.zeros((1,13),dtype=float)
    for i in range(len(x0)):
       X0M=X0M+X0[i,:]
    X0M=X0M/(len(x0))
    # SB between-class covariance matrix
    SB=np.dot((X1M-X0M).transpose(),X1M-X0M)
    # SW within-class covariance matrix
    S1=np.zeros((13,13),dtype=float)
    for i in range(len(x1)):
       a=X1[i,:]-X1M
       S1=S1+np.dot(a.transpose(),a)
       
    S0=np.zeros((13,13),dtype=float)
    for i in range(len(x0)):
       a=X0[i,:]-X0M
       S0=S0+np.dot(a.transpose(),a)
    SW=S1+S0
    # compute the projection array
    if np.linalg.det(SW)!=0:
        SWi=np.linalg.inv(SW)
    else:
        print("SW has no inverse matrix and FDA fails")
    p=np.dot(SWi,(X1M-X0M).transpose())
    # compute the threshold
    XM=np.zeros((1,13),dtype=float)
    for i in range(len(Boston50x)):
       XM=XM+X[i,:]
    XM=XM/(len(Boston50x))
    w0=np.dot(-p.transpose(),XM.transpose())
    
    # error rate of the jth fold
    # error rate of train set
    tt=np.dot(X,p)+w0
    t_hat=np.zeros((len(Boston50_train),1),dtype=float)
    for i in range(len(Boston50_train)):
        if tt[i,:]>0:
            t_hat[i,:]=1
        else:
            t_hat[i,:]=0
    error_num=0
    for i in range(len(Boston50_train)):
        if t_hat[i,:]!=Y[i]:
            error_num=error_num+1
    error_rate_train=error_num/len(Boston50_train)
    error_rate_array_train[j,:]=error_rate_train
    
    # error rate of test set
    Xt=Boston50_test.drop(columns="t").to_numpy()
    Yt=Boston50_test["t"].to_numpy()
    tt=np.dot(Xt,p)+w0
    t_hat=np.zeros((len(Boston50_test),1),dtype=float)
    for i in range(len(Boston50_test)):
        if tt[i,:]>0:
            t_hat[i,:]=1
        else:
            t_hat[i,:]=0
    error_num=0
    for i in range(len(Boston50_test)):
        if t_hat[i,:]!=Yt[i]:
            error_num=error_num+1
    error_rate_test=error_num/len(Boston50_test)
    error_rate_array_test[j,:]=error_rate_test
    
    
# Report the result
# Average error rates for trainset and testset
# average error rates for trainset
train_error=np.mean(error_rate_array_train)
print("The average error rates for trainset under 10-fold CV is",train_error,"(FDA)\n")

# average error rates for testset
test_error=np.mean(error_rate_array_test)
print("The average error rates for testset under 10-fold CV is",test_error,"(FDA)\n")

# Standard deviation of error rates for trainset and testset
# standard deviation of error rates for trainset
train_SD=np.std(error_rate_array_train)
print("The standard deviation for trainset under 10-fold CV is",train_SD,"(FDA)\n")

# standard deviation of error rates for testset
test_SD=np.std(error_rate_array_test)
print("The standard deviation for testset under 10-fold CV is",test_SD,"(FDA)")




    
