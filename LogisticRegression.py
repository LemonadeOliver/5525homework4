# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 09:20:32 2022

@author: LemonadeOliver
"""

####### 4(i)
### 1) Logistic regression for Boston50 and Boston 75
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

# Logistic regression
# For Boston50 and Boston75
# Split up the dataset according to the class Label
# Boston50
Boston50_1=Boston50[Boston50["t"]==1]
Boston50_0=Boston50[Boston50["t"]==0]
# Boston75
Boston75_1=Boston75[Boston75["t"]==1]
Boston75_0=Boston50[Boston75["t"]==0]


# Define error rate matrix
error_rate_Boston50=np.zeros((10,5),dtype=float)
error_rate_Boston75=np.zeros((10,5),dtype=float)

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

        # Logistic regression for Boston50
        b=np.ones((len(X_train_50),1),dtype=float)
        XX_train_50=np.append(X_train_50,b,axis=1)
        # define the gradient of the error function 
        def g(w):
            a=(np.zeros((Boston50_train_new.shape[1],1),dtype=float)).ravel()
            for i in range(len(Boston50_train_new)):
                a=a+((1/(1+np.exp(-np.dot(XX_train_50[i,:],w))))-Y_train_50[i])*(XX_train_50[i,:].transpose())
            return a
        # define sigmoid function
        def sig(X):
            return 1/(1+ma.e**(-X))
        # conduct the gradient descent to find optimal w
        # assign random values to initial w
        w=np.random.randn(len(XX_train_50[0]),1).ravel()
        # With learning rate as 1, conduct 100000 times gradient descent method to update w
        while i<=1000000:
            w=w+(-1)*XX_train_50.T.dot(sig(XX_train_50.dot(w)).ravel()-Y_train_50)
            i=i+1
            
        #err=np.linalg.norm(XX_train_50.T.dot(sig(XX_train_50.dot(w)).ravel()-Y_train_50))
        #print(err,"\n")
        # use w and sigmoid function to predict Y in the test set
        c=np.ones((len(X_test_50),1),dtype=float)
        XX_test_50=np.append(X_test_50,c,axis=1)
        P_test=sig(XX_test_50.dot(w))
        Y_hat_50_test=[]
        for i in range(len(P_test)):
            if P_test[i]>(1/2):
                Y_hat_50_test.append(1)
            else:
                Y_hat_50_test.append(0)
        Y_hat_50_test=np.array(Y_hat_50_test)
        # compute the error rate
        error_num=0
        for i in range(len(X_test_50)):
            if Y_hat_50_test[i]!=Y_test_50[i]:
                error_num=error_num+1
        error_rate_test=error_num/len(X_test_50)
        error_rate_Boston50[j,k]=error_rate_test
        
    
    
    
        # Logistic regression for Boston75
        b=np.ones((len(X_train_75),1),dtype=float)
        XX_train_75=np.append(X_train_75,b,axis=1)
        # define sigmoid function
        def sig(X):
            return 1/(1+ma.e**(-X))
        # conduct the gradient descent to find optimal w
        # assign random values to initial w
        w=np.random.randn(len(XX_train_75[0]),1).ravel()
        # With learning rate as 1, conduct 1000000 times gradient descent method to update w
        while i<=10000:
            w=w+(-1)*XX_train_75.T.dot(sig(XX_train_75.dot(w)).ravel()-Y_train_75)
            i=i+1
            
        #err=np.linalg.norm(XX_train_75.T.dot(sig(XX_train_75.dot(w)).ravel()-Y_train_75))
        #print(err,"\n")
        # use w and sigmoid function to predict Y in the test set
        c=np.ones((len(X_test_75),1),dtype=float)
        XX_test_75=np.append(X_test_75,c,axis=1)
        P_test=sig(XX_test_75.dot(w))
        Y_hat_75_test=[]
        for i in range(len(P_test)):
            if P_test[i]>(1/2):
                Y_hat_75_test.append(1)
            else:
                Y_hat_75_test.append(0)
        Y_hat_75_test=np.array(Y_hat_75_test)
        # compute the error rate
        error_num=0
        for i in range(len(X_test_75)):
            if Y_hat_75_test[i]!=Y_test_75[i]:
                error_num=error_num+1
        error_rate_test=error_num/len(X_test_75)
        error_rate_Boston75[j,k]=error_rate_test
        
# Compute and print the mean error rate of 20% test set for 25%, 50%, 75% and 100% of 80% train set respectively and plot them against train set ratio
ratio=np.array([10,25,50,75,100])
x=ratio
# Boston 50
print("The error rate matrix for Boston50 (the ith row represents the ith time when the dataset is randomly split and the columns represent the ratio of the trainset we take from 10% to 100%)\n",error_rate_Boston50)
mean_error_rate_50=error_rate_Boston50.mean(axis=0)
print("The mean error rate of Boston50 for each ratio of trainset from 25% to 100% \n",mean_error_rate_50)
y=mean_error_rate_50
print("The plot for mean error rate of Boston50 vs ratio of trainset \n")
plt.plot(x,y)
plt.show()

# Boston 75
print("The error rate matrix for Boston75 (the ith row represents the ith time when the dataset is randomly split and the columns represent the ratio of the trainset we take from 10% to 100%)\n",error_rate_Boston75)
mean_error_rate_75=error_rate_Boston75.mean(axis=0)
print("The mean error rate of Boston75 for each ratio of trainset from 25% to 100% \n",mean_error_rate_75)
y=mean_error_rate_75
print("The plot for mean error rate of Boston50 vs ratio of trainset \n")
plt.plot(x,y)
plt.show()

# Compute the standard deviation of the error rate of 20% test set for 25%, 50%, 75% and 100% of 80% train set respectively and plot them against train set ratio
# Boston 50
SD_error_rate_50=error_rate_Boston50.std(axis=0)
print("The standard error for the error rate of Boston50 for each ratio of trainset from 25% to 100% \n",SD_error_rate_50)
y=SD_error_rate_50
print("The plot for standard error for the error rate of Boston50 vs ratio of trainset \n")
plt.plot(x,y)
plt.show()

# Boston 75
SD_error_rate_75=error_rate_Boston75.std(axis=0)
print("The standard error for the error rate of Boston75 for each ratio of trainset from 25% to 100% \n",SD_error_rate_75) 
y=SD_error_rate_75
print("The plot for standard error for the error rate of Boston75 vs ratio of trainset \n")
plt.plot(x,y)
plt.show()   









### 2) Logistic regression for Digit
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

# Split the data into 10 class according to their class label
data3i=dict()
for k in range(10):
    data3i[k]=pd.DataFrame()
    data3i[k]=data3[data3.iloc[:,data3.shape[1]-1]==k]

# Define the error rate matrix
error_rate_digit=np.zeros((10,5),dtype=float)
data3i_80=dict()
data3i_20=dict()
for k in range(10):
    data3i_80[k]=pd.DataFrame()
    data3i_20[k]=pd.DataFrame()
    

# randomly choose samples from each class based on 80-20 spliting rate for train set and test set for 10 times
for j in range(10):
    # extract 80% data from each class for train set and 20% data from each class for test set
    for i in range(10):
        # shuffle the dataset in the jth time
        data3i[i]=data3i[i].sample(frac=1)
        data3i[i]=data3i[i].reset_index(drop=True)
        # Choose the top 80% of the dataset as the train set
        data3i_80[i]=data3i[i].iloc[range((np.ceil(len(data3i[i])*0.8)).astype(int)),:]
        # Choose the rest 20% of the dataset as the test set
        data3i_20[i]=data3i[i].iloc[data3i[i].index.difference(data3i_80[i].index),:]
    
    # integrate all the 80% data into trainset and all the 20% data into testset
    # Trainset
    data3_train=pd.DataFrame()
    for i in range(10):
       data3_train=pd.concat([data3_train,data3i_80[i]])

    data3_train=data3_train.sample(frac=1)
    data3_train=data3_train.reset_index(drop=True)
    
    # Testset
    data3_test=pd.DataFrame()
    for i in range(10):
       data3_test=pd.concat([data3_test,data3i_20[i]])

    data3_test=data3_test.sample(frac=1)
    data3_test=data3_test.reset_index(drop=True)   
    
    # Take 25%, 50%, 75% and 100% of the train set as the new train set respectively and remain the same test set
    for k in range(5):
        if k==0:
            k=0.4
        # data3_train
        data3_train_new=data3_train.iloc[range((np.ceil(len(data3_train)*k*0.25)).astype(int))]
        if k==0.4:
            k=0
        # Split the train set and test set into X and Y
        X_train=data3_train_new.drop(columns=data3_train_new.shape[1]-1).to_numpy()
        c=np.ones((len(X_train),1),dtype=float)
        XX_train=np.append(X_train,c,axis=1)
        Y_train=pd.get_dummies(data3_train_new.iloc[:,data3_train_new.shape[1]-1]).to_numpy()
        X_test=data3_test.drop(columns=data3_test.shape[1]-1).to_numpy()
        cc=np.ones((len(X_test),1),dtype=float)
        XX_test=np.append(X_test,cc,axis=1)
        Y_test=pd.get_dummies(data3_test.iloc[:,data3_test.shape[1]-1]).to_numpy()
        YY_test=data3_test.iloc[:,data3_test.shape[1]-1].to_numpy()
        

            
        # Gradient Descent Method to update weight matrix W
        # Set initial value for W
        W=np.random.random((data3_train_new.shape[1],10))
        i=0
        # Use the gradient for each weight to update each weight
        while i<=1000:
                # create the martrix for aij
                A=XX_train.dot(W)
                maxA=np.reshape(np.amax(A,axis=1),(len(data3_train_new),1))
                A=A-maxA
                # create the matrix for Pij
                e_A=np.exp(A)
                e_A_sum=e_A.sum(axis=1)
                P=np.zeros((len(e_A[:,0]),len(e_A[0,:])),dtype=float)               
                for x in range(len(e_A[:,0])):
                    for y in range(len(e_A[0,:])):
                                   P[x,y]=e_A[x,y]/e_A_sum[x]                                 
                # compute the gradient for the each w and use it to update w               
                for x in range(10):                    
                    gradient=(np.zeros((data3_train_new.shape[1],1),dtype=float)).ravel()
                    for y in range(len(data3_train_new)):
                        gradient=gradient+(P[y,x]-Y_train[y,x])*XX_train[y,:].transpose()
                    W[:,x]=W[:,x]+(-1)*gradient
                i=i+1
       # Use the test set and W matrix to predict the response
       # create the martrix for aij
        A=XX_test.dot(W)
        maxA=np.reshape(np.amax(A,axis=1),(len(XX_test),1))
        A=A-maxA
        # create the matrix for Pij
        e_A=np.exp(A)
        e_A_sum=e_A.sum(axis=1)
        P=np.zeros((len(e_A[:,0]),len(e_A[0,:])),dtype=float)               
        for x in range(len(e_A[:,0])):
            for y in range(len(e_A[0,:])):
                   P[x,y]=e_A[x,y]/e_A_sum[x] 
        y_hat_test=np.zeros((len(data3_test),1),dtype=float)
        for i in range(len(data3_test)):
            y_hat_test[i,:]=np.argmax(P[i,:])
        
        # Compute the error rate for the test set
        error_num=0
        for i in range(len(X_test)):
            if y_hat_test[i,:]!=YY_test[i]:
                error_num=error_num+1
        error_rate_test=error_num/len(X_test)
        error_rate_digit[j,k]=error_rate_test




# Compute and print the mean error rate of 20% test set for 25%, 50%, 75% and 100% of 80% train set respectively and plot them against train set ratio
ratio=np.array([10,25,50,75,100])
x=ratio
# Digit
print("The error rate matrix for Digit (the ith row represents the ith time when the dataset is randomly split and the columns represent the ratio of the trainset we take from 10% to 100%)\n",error_rate_digit)
mean_error_rate_digit=error_rate_digit.mean(axis=0)
print("The mean error rate of Boston50 for each ratio of trainset from 25% to 100% \n",mean_error_rate_digit)
y=mean_error_rate_digit
print("The plot for mean error rate vs ratio of trainset \n")
plt.plot(x,y)
plt.show()
            
# Compute the standard deviation of the error rate of 20% test set for 25%, 50%, 75% and 100% of 80% train set respectively and plot them against train set ratio
# Digit
SD_error_rate_digit=error_rate_digit.std(axis=0)
print("The standard error for the error rate of Digit for each ratio of trainset from 25% to 100% \n",SD_error_rate_digit)
y=SD_error_rate_digit
print("The plot for standard error for the error rate vs ratio of trainset \n")
plt.plot(x,y)
plt.show()        
            
                
        
  