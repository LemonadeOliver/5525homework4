# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 23:23:49 2022

@author: LemonadeOliver
"""

################## 3 (2)
import pandas as pd
import numpy as np
# data cleaning
# detect missing values
data1=pd.read_csv("digits.csv",header=None)
data1.isnull().sum()
# delete columns with excessive number of 0
d=data1.to_numpy()
zero_col=[]
for i in range(data1.shape[1]):
    if np.count_nonzero(d[:,i]==0)>=1500:
        #print("column with excessive number of 0 is",i)
        zero_col.append(i)
data2=data1.drop(columns=zero_col)
zero_col.append(64)
data0=data1.drop(columns=zero_col)
# detect and delete outliers
from scipy import stats
data3=data2[(np.abs(stats.zscore(data0))<5).all(axis=1)]
data3=data3.reset_index(drop=True)
data3.columns=range(data3.columns.size)


# 10-fold Cross-validation to compute average error rate and standard deviation for the method that uses FDA as projection and bi-variate Gaussian generative models as classification

# produce the indexes of testset for 10-fold cross-validation
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
Index=list(split(range(len(data3.index)), 10))

# Shuffle the dataset
data3=data3.sample(frac=1)
data3=data3.reset_index(drop=True)


# define the error rate array for 10 train sets
error_rate_array_train=np.zeros((10,1),dtype=float)
# define the error rate array for 10 test sets
error_rate_array_test=np.zeros((10,1),dtype=float)

# 10-fold Cross-Validation
for j in range(len(Index)):
    # Spilt up the dataset into train set and test set
    trainset=data3.iloc[data3.index.difference(Index[j]),:]
    testset=data3.iloc[Index[j],:]
    X_train=trainset.drop(columns=data3.shape[1]-1).to_numpy()
    Y_train=trainset.iloc[:,data3.shape[1]-1].to_numpy()
    X_test=testset.drop(columns=data3.shape[1]-1).to_numpy()
    Y_test=testset.iloc[:,data3.shape[1]-1].to_numpy()
    # FDA to project the data into 2 dimensions
    # divide the trainset into 10 parts according to their labels
    Xi=dict()
    for i in range(10):
        Xi[i]=[]
    for i in range(len(X_train)):
        if Y_train[i]==0:
                Xi[0].append(X_train[i,:])
        elif Y_train[i]==1:
                Xi[1].append(X_train[i,:])
        elif Y_train[i]==2:
                Xi[2].append(X_train[i,:])
        elif Y_train[i]==3:
                Xi[3].append(X_train[i,:])
        elif Y_train[i]==4:
                Xi[4].append(X_train[i,:])
        elif Y_train[i]==5:
                Xi[5].append(X_train[i,:])
        elif Y_train[i]==6:
                Xi[6].append(X_train[i,:])
        elif Y_train[i]==7:
                Xi[7].append(X_train[i,:])
        elif Y_train[i]==8:
                Xi[8].append(X_train[i,:])
        else:
                Xi[9].append(X_train[i,:])
    for i in range(10):
        Xi[i]=np.array(Xi[i])
        
    # compute means for 10 class and the overall mean
    m=np.zeros((1,data3.shape[1]-1),dtype=float)
    for i in range(len(X_train)):
        m=m+X_train[i,:]
    m=m/len(X_train)
    
    mi=dict()
    for i in range(10):
        mi[i]=np.zeros((1,data3.shape[1]-1),dtype=float)
    for k in range(10):
        for i in range(len(Xi[k])):
           mi[k]=mi[k]+Xi[k][i,:]
        mi[k]=mi[k]/len(Xi[k])
    
    # Compute SW
    Sk=np.zeros((data3.shape[1]-1,data3.shape[1]-1),dtype=float)
    SW=np.zeros((data3.shape[1]-1,data3.shape[1]-1),dtype=float)
    for k in range(10):
        for i in range(len(Xi[k])):
            Sk=Sk+np.dot((Xi[k][i,:]-mi[k]).transpose(),Xi[k][i,:]-mi[k])
        SW=SW+Sk

    # Compute SB
    SB=np.zeros((data3.shape[1]-1,data3.shape[1]-1),dtype=float)
    for k in range(10):
        SB=SB+len(Xi[k])*np.dot((mi[k]-m).transpose(),mi[k]-m)
    
    # Compute the eigenvectors for the two largest eigenvalue of SW^(-1)SB
    if np.linalg.det(SW)!=0:
        SWi=np.linalg.inv(SW)
    else:
        print("SW has no inverse matrix and FDA fails")
    w,v=np.linalg.eig(np.dot(SWi,SB))
    eigen=np.zeros((47,46),dtype=complex)
    for i in range(len(w)):
        eigen[:,i]=np.append(v[:,i],w[i])
    # find the first two largest eigenvalue of SW^(-1)SB
    ind = np.argpartition(w, -2)[-2:]
    # define the projection matrix
    W=eigen[:,ind]
    W=np.delete(W,46,0)

    # project X to 2-dimension and name the new X as Z and divide Z into 10 parts according to the class labels
    Z=np.dot(X_train,W)
    Zi=dict()
    for k in range(10):
        Zi[k]=[]
        Zi[k]=np.dot(Xi[k],W)
        Zi[k]=np.array(Zi[k],dtype=float)
    # use bi-variate Gaussian generative model as class-conditional density for classification
    # MLE for PI, u, Covariance Matrix for each posterior probability
    # PI
    PIi=dict()
    for k in range(10):
        PIi[k]=[]
        PIi[k]=len(Zi[k])/len(X_train)
    # u
    Ui=dict()
    for k in range(10):
        Ui[k]=np.zeros((1,2),dtype=float)
    
    for k in range(10):
        for i in range(len(Zi[k])):
            Ui[k]=Ui[k]+Zi[k][i,:]
        Ui[k]=Ui[k]/len(Zi[k])
    
    # Covariance Matrix
    Si=dict()
    for k in range(10):
        Si[k]=np.zeros((2,2),dtype=float)
    
    S=np.zeros((2,2),dtype=float)
    for k in range(10):
        for i in range(len(Zi[k])):
            Si[k]=Si[k]+np.dot((Zi[k][i,:]-Ui[k]).transpose(),Zi[k][i,:]-Ui[k])
        S=S+(len(Zi[k])/len(X_train))*(1/len(Zi[k]))*Si[k]
    COV=S
    # Compute ak for each x
    wi=dict()
    w0i=dict()
    COVi=np.linalg.inv(COV)
    for k in range(10):
        wi[k]=np.zeros((2,1),dtype=float)
        w0i[k]=[]
        
    for k in range(10):
        wi[k]=np.dot(COVi,Ui[k].transpose())
        w0i[k]=(-1/2)*np.dot(np.dot(Ui[k],COVi),Ui[k].transpose())+np.log(PIi[k])

    a=np.zeros((len(X_train),10),dtype=float)
    for k in range(10):
        a[:,k]=(np.dot(Z,wi[k])+w0i[k]).ravel()
    # Compute the posterior probability of each class for each x
    a_exp=np.exp(a)
    post=np.zeros((len(X_train),10),dtype=float)
    for i in range(len(X_train)):
        summ=np.sum(a_exp[i,:])
        for k in range(10):
            post[i,k]=a_exp[i,k]/summ
    # Compute the predicted response for the train set
    y_hat_train=np.zeros((len(X_train),1),dtype=float)
    for i in range(len(X_train)):
        y_hat_train[i,:]=np.argmax(post[i,:])
        
    # Compute the error rate for the j fold
    # Compute the error rate for train set
    error_num=0
    for i in range(len(X_train)):
        if y_hat_train[i,:]!=Y_train[i]:
            error_num=error_num+1
    error_rate_train=error_num/len(X_train)
    error_rate_array_train[j,:]=error_rate_train
    
    # Compute the error rate for test set
    # project X to 2-dimension and name the new X as Z
    Z=np.dot(X_test,W)
    # Compute the posterior probability of each x under each class
    a=np.zeros((len(X_test),10),dtype=float)
    for k in range(10):
        a[:,k]=(np.dot(Z,wi[k])+w0i[k]).ravel()
    a_exp=np.exp(a)
    post=np.zeros((len(X_test),10),dtype=float)
    for i in range(len(X_test)):
        summ=np.sum(a_exp[i,:])
        for k in range(10):
            post[i,k]=a_exp[i,k]/summ
    # Compute the predicted response for the train set
    y_hat_test=np.zeros((len(X_test),1),dtype=float)
    for i in range(len(X_test)):
        y_hat_test[i,:]=np.argmax(post[i,:])
    # Compute the error rate
    error_num=0
    for i in range(len(X_test)):
        if y_hat_test[i,:]!=Y_test[i]:
            error_num=error_num+1
    error_rate_test=error_num/len(X_test)
    error_rate_array_test[j,:]=error_rate_test
    
# Report the result
# Average error rates for trainset and testset
# average error rates for trainset
train_error=np.mean(error_rate_array_train)
print("The average error rates for trainset under 10-fold CV is",train_error,"(FDA to project into R^2 and Probabilistics Generative Models to classify)\n")

# average error rates for testset
test_error=np.mean(error_rate_array_test)
print("The average error rates for testset under 10-fold CV is",test_error,"(FDA to project into R^2 and Probabilistics Generative Models to classify)\n")

# Standard deviation of error rates for trainset and testset
# standard deviation of error rates for trainset
train_SD=np.std(error_rate_array_train)
print("The standard deviation for trainset under 10-fold CV is",train_SD,"(FDA to project into R^2 and Probabilistics Generative Models to classify)\n")

# standard deviation of error rates for testset
test_SD=np.std(error_rate_array_test)
print("The standard deviation for testset under 10-fold CV is",test_SD,"(FDA to project into R^2 and Probabilistics Generative Models to classify)\n")