# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:58:19 2020

@author: jcremaldi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)

raw_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')
df = raw_data.copy()
df = df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
print(df.describe())

 
def data_prep(df):
    df['due_date'] = pd.to_datetime(df['due_date'])
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    df['dayofweek'] = df['effective_date'].dt.dayofweek
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
    df['Gender'].replace('male',0,inplace=True)
    df['Gender'].replace('female',1,inplace=True)
    
    #one hot encode the education level, why do they drop 'Master or Above'?
    temp = df[['Principal','terms','age','Gender','weekend']].copy()
    temp = pd.concat([temp,pd.get_dummies(df['education'])], axis=1)
    temp = pd.concat([temp,pd.get_dummies(df['terms'])], axis=1)
    temp.drop(['Master or Above','terms'], axis = 1,inplace=True)
    #fix a spelling error
    temp.rename(columns = {'Bechalor':'Bachelor'}, inplace = True)
    print(temp.columns)
    # standardize the other numerical variables
    ss_col_names = ['Principal','age']
    ss_col = temp[ss_col_names].copy()
    scaler = preprocessing.StandardScaler().fit(ss_col.values)
    features = scaler.transform(ss_col.values)
    temp[ss_col_names] = features
    X = temp
    y = df['loan_status'].values
    return X, y

def plot_trend(n, mean_acc, x_title):
    plt.plot(range(1,n),mean_acc,'g')
    plt.ylabel('Accuracy ')
    plt.xlabel(x_title)
    plt.tight_layout()
    plt.show()    

def strat_k_folds(model_skfold,X, y):
    skfold = model_selection.StratifiedKFold(n_splits = 4)
    results_skfold = model_selection.cross_val_score(model_skfold, X, y, cv=skfold)
    return results_skfold.mean()

X, y = data_prep(df)
#------------------------------------------------------------------------------
Ks = 20
mean_acc_knn = np.zeros((Ks-1))

for n_knn in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors = n_knn)    
    mean_acc_knn[n_knn-1] = strat_k_folds(neigh, X, y)

plot_trend(Ks, mean_acc_knn, 'Number of Neighbors (K)')
best_n = mean_acc_knn.argmax()+1
print( "The best accuracy was ", mean_acc_knn.max(), "with k=", best_n, '\n') 
#------------------------------------------------------------------------------
dep = 15
mean_acc_dt = np.zeros((dep-1))

for n_dep in range(1,dep):
    #Train Model and Predict  
    loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = n_dep)
    mean_acc_dt[n_dep-1] = strat_k_folds(loanTree, X, y) 

plot_trend(dep, mean_acc_dt, 'Max Tree Depth')
best_dep = mean_acc_dt.argmax()+1
print( "The best accuracy was ", mean_acc_dt.max(), "with max_depth = ", best_dep, '\n') 
#------------------------------------------------------------------------------

kern = ['linear','poly','rbf','sigmoid']
mean_acc_svm = np.zeros((dep-1))

temp_acc_svm = 0
temp_max_svm = ''
for k in kern:
    svm_clf = svm.SVC(kernel=str(k),gamma='scale')
    acc = strat_k_folds(svm_clf, X, y)
    if acc > temp_acc_svm:
        temp_acc_svm = acc
        temp_max_svm = k
        
print("The best accuracy was ", temp_acc_svm, "with kernel = ", temp_max_svm, '\n') 

#------------------------------------------------------------------------------
solv = ['newton-cg','lbfgs','liblinear']
temp_acc_lr = 0
temp_max_lr = ''
for s in solv:
    LR = LogisticRegression(C=0.25, solver=str(s))
    acc = strat_k_folds(LR, X, y)
    if acc > temp_acc_lr:
        temp_acc_lr = acc
        temp_max_lr = s
print("The best accuracy was ", temp_acc_lr, "with solver = ", temp_max_lr, '\n') 

c_test = [.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]
mean_acc_lr = np.zeros((len(c_test)-1))
for c in range(len(c_test)):
    LR = LogisticRegression(C = c_test[c], solver = str(temp_max_lr))
    mean_acc_lr[c-1] = strat_k_folds(LR, X, y)
    
best_c = c_test[mean_acc_lr.argmax()+1]
print( "The best accuracy was ", mean_acc_lr.max(), "with c = ", best_c, '\n')     

#------------------------------------------------------------------------------
df_train = raw_data.copy()
df_train = df_train.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)

X_train, y_train = data_prep(df_train)

#apply the same transformations to the data as used in building the model
df_test =  pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')
df_test = df_test.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)

X_test, y_test = data_prep(df_test)
#------------------------------------------------------------------------------
#train models using optimal parameters and full data set
neigh_pred = KNeighborsClassifier(n_neighbors = best_n).fit(X_train, y_train)
predNeigh = neigh_pred.predict(X_test)
print('\n')
print('KNeighborsClassifier')
print('accuracy_score: ', metrics.accuracy_score(y_test, predNeigh))
print('f1_score: ', metrics.f1_score(y_test, predNeigh, average='weighted'))
print('jaccard_score: ',metrics.jaccard_score(y_test, predNeigh, average='weighted'))


clf_pred = svm.SVC(kernel=str(temp_max_svm)).fit(X_train, y_train)
predClf = clf_pred.predict(X_test)
print('\n')
print('svm')
print('accuracy_score: ', metrics.accuracy_score(y_test, predClf))
print('f1_score: ',metrics.f1_score(y_test, predClf, average='weighted'))
print('jaccard_score: ',metrics.jaccard_score(y_test, predClf, average='weighted'))


loanTree_pred = DecisionTreeClassifier(criterion="entropy", max_depth = best_dep).fit(X_train, y_train)
predLoanTree = loanTree_pred.predict(X_test)
print('\n')
print('DecisionTreeClassifier')
print('accuracy_score: ', metrics.accuracy_score(y_test, predLoanTree))
print('f1_score: ',metrics.f1_score(y_test, predLoanTree, average='weighted'))
print('jaccard_score: ',metrics.jaccard_score(y_test, predLoanTree, average='weighted'))


LR_pred = LogisticRegression(C=best_c, solver=str(temp_max_lr)).fit(X_train, y_train)
predLR = LR_pred.predict(X_test)
predLR_prob = LR_pred.predict_proba(X_test)
print('\n')
print('LogisticRegression')
print('accuracy_score: ', metrics.accuracy_score(y_test, predLR))
print('f1_score: ',metrics.f1_score(y_test, predLR, average='weighted'))
print('jaccard_score: ',metrics.jaccard_score(y_test, predLR, average='weighted'))
print('log_loss: ',metrics.log_loss(y_test, predLR_prob))







































