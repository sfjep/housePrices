# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:05:53 2020

@author: Simon
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns 

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, metrics, tree, svm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report, confusion_matrix, jaccard_similarity_score, log_loss, f1_score, mean_squared_error, mean_absolute_error 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline


df = pd.read_csv('featureSelection.csv')

X_test = df.loc[df.SalePrice_log.isna()].copy()
train_df = df.loc[~df.SalePrice_log.isna()].copy()

X_test.drop('SalePrice_log', axis = 1, inplace = True)

y_train = train_df.SalePrice_log.values
X_train = train_df.drop('SalePrice_log', axis = 1)

'''
As found in capstone project
https://github.com/sfjep/IBM_Course_CapstoneProject/blob/master/CapstoneProject%20-%20Model%20Building.ipynb
A combination of RF, SVR, and GB is found to be optimal
'''

kfold = KFold(n_splits=10)

# MODEL TUNING FUNCTIONS
def rfTuning(X_train, y_train, X_test):
    # Optimized at max_depth = 25, n_estimators = 280
    parameters = {
        'n_estimators': range(200, 300, 20), 
        'max_depth': range(20, 30, 5)
        }
    
    rf = RandomForestRegressor(criterion = 'mae')
    
    np.random.seed(1)
    gs_rf = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv = kfold)
    gs_rf.fit(X_train, y_train)
    rf_pred = np.expm1(gs_rf.best_estimator_.predict(X_train))
    print(gs_rf.best_score_)
    print(gs_rf.best_estimator_)
    
    return rf_pred


def gbTuning(X_train, y_train, X_test):
    parameters = {'learning_rate': [0.005, 0.01, 0.015, 0.02], 
                  'n_estimators': range(150, 200, 25)
                  }
    
    gb = GradientBoostingRegressor(loss = 'huber')
    
    np.random.seed(1)
    gs_gb = GridSearchCV(gb, parameters, scoring = 'neg_mean_absolute_error', cv = kfold)
    gs_gb.fit(X_train, y_train)
    gb_pred = np.expm1(gs_gb.best_estimator_.predict(X_test))
    print(gs_gb.best_score_)
    print(gs_gb.best_estimator_)

    return gb_pred

def svrTuning(X_train, y_train, X_test):
    # OPTIMZED AT: C = 0.5, kernel = 'linear'
    parameters = {'C': [0.1, 0.5, 1, 1.5, 2], 
                  'kernel': ('linear', 'sigmoid', 'rbf', 'poly'),
                 }
    
    svr = SVR()
    
    np.random.seed(1)
    gs_svr = GridSearchCV(svr, parameters, scoring = 'neg_mean_absolute_error', cv = kfold)
    gs_svr.fit(X_train, y_train)
    svr_pred = np.expm1(gs_svr.best_estimator_.predict(X_test))
    print(gs_svr.best_score_)
    print(gs_svr.best_estimator_)

    return svr_pred

# TUNE
rf_pred = rfTuning(X_train, y_train, X_test)
gb_pred = gbTuning(X_train, y_train, X_test)
svr_pred = svrTuning(X_train, y_train, X_test)

comb_pred = (rf_pred + gb_pred + svr_pred) / 3


# Submisssions
hid = X_test.Id

def generateSubmissionDf(hid, pred, name):
    submission = pd.DataFrame({
            "Id": hid,
            "SalePrice": pred
        })
    
    submission.to_csv(('submission/' + name + '.csv'), index = False)   

generateSubmissionDf(hid, rf_pred, 'rf')
generateSubmissionDf(hid, gb_pred, 'gb')
generateSubmissionDf(hid, svr_pred, 'svr')
generateSubmissionDf(hid, comb_pred, 'comb')

