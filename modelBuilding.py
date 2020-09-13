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
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv('featureSelection.csv')

X_test = df.loc[df.SalePrice_log.isna()].copy()
train_df = df.loc[~df.SalePrice_log.isna()].copy()

X_test.drop('SalePrice_log', axis = 1, inplace = True)
hid = X_test.Id
train_df.drop('Id', axis = 1, inplace = True)
X_test.drop('Id', axis = 1, inplace = True)


y_train = train_df.SalePrice_log.values
X_train = train_df.drop('SalePrice_log', axis = 1)

'''
As found in capstone project
https://github.com/sfjep/IBM_Course_CapstoneProject/blob/master/CapstoneProject%20-%20Model%20Building.ipynb
A combination of RF, SVR, and GB is found to be optimal
SVR does however present poor predictions.
'''

kfold = KFold(n_splits=10)
RandomizedSearchCV(xg_cl,param,verbose=10)
# MODEL TUNING FUNCTIONS
def rfTuning(X_train, y_train, X_test):
    # Optimized at max_depth = 25, n_estimators = 280
    parameters = {
        'n_estimators': range(200, 300, 20), 
        'max_depth': range(20, 40, 5),
        
        }
    
    rf = RandomForestRegressor(criterion = 'mae')
    
    np.random.seed(1)
    gs_rf = RandomizedSearchCV(rf, parameters, n_jobs=1, cv=kfold)
    gs_rf.fit(X_train, y_train)
    rf_pred = np.expm1(gs_rf.best_estimator_.predict(X_test))
    print(gs_rf.best_score_)
    print(gs_rf.best_estimator_)
    
    return rf_pred


def gbTuning(X_train, y_train, X_test):
    parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1], 
                  'n_estimators': range(150, 200, 25)
                  }
    
    gb = GradientBoostingRegressor(loss = 'huber')
    
    np.random.seed(1)
    gs_gb = RandomizedSearchCV(gb, parameters, n_jobs=1, cv=kfold)
    gs_gb.fit(X_train, y_train)
    gb_pred = np.expm1(gs_gb.best_estimator_.predict(X_test))
    print(gs_gb.best_score_)
    print(gs_gb.best_estimator_)

    return gb_pred

def xgbRegressor(X_train, y_train, X_test):
    
    clf = xgb.XGBRegressor(
        eval_metric = 'rmse',
        nthread = 4,
        eta = 0.1,
        num_boost_round = 80,
        subsample = 0.5,
        silent = 1,
        )
    parameters = {
        'num_boost_round': [10, 25, 50],
        'eta': [0.05, 0.1, 0.3],
        'max_depth': [3, 4, 5],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'n_estimators': [100]
        }
        
            
    clf2 = RandomizedSearchCV(clf, parameters, n_jobs=1, cv=kfold)
    clf2.fit(X_train, y_train)
    xgb_pred = np.expm1(clf2.predict(X_test))
    
    return xgb_pred


# TUNE
rf_pred = rfTuning(X_train, y_train, X_test)
gb_pred = gbTuning(X_train, y_train, X_test)
xgb_pred = xgbRegressor(X_train, y_train, X_test)


comb_pred = (2*rf_pred + gb_pred + 4 * xgb_pred) / 7


# Submisssions


def generateSubmissionDf(hid, pred, name):
    submission = pd.DataFrame({
            "Id": hid,
            "SalePrice": pred
        })
    
    submission.to_csv(('submission/' + name + '.csv'), index = False)   

generateSubmissionDf(hid, rf_pred, 'rf')
generateSubmissionDf(hid, gb_pred, 'gb')
generateSubmissionDf(hid, xgb_pred, 'xgb')

generateSubmissionDf(hid, comb_pred, 'comb')

