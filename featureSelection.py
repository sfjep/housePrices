# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:14:41 2020

@author: Simon
"""

import pandas as pd
import numpy as np

# Import Dataframe
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Combine train and test to ease feature selection
df = train.append(test)

####################################### 
##            Sales Price            ##
#######################################

df['SalePrice_log'] = np.log(df['SalePrice'])
df.drop('SalePrice', axis = 1, inplace = True)


####################################### 
##            Garage                 ##
#######################################

rank_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'None': 1, 'Po':0}


df['GarageCond'].fillna('None', inplace = True)
df['GarageType'].fillna('None', inplace = True)
df['GarageQual'].fillna('None', inplace = True)
df['GarageFinish'].fillna('None', inplace = True)

GarFin_dict = {'Fin':2, 'RFn':1, 'Unf':0}
GarType_dict = {'2Types': '2Types', 'Attchd': 'BuiltIn', 'Basment': 'BuiltIn', 'CarPort': 'Separate', 
               'Detchd': 'Separate'}

# Garage Quality and Condition
df['GarageQual'] = df.GarageQual.replace(rank_dict)
df['GarageQual'].fillna('None', inplace = True)
df['GarageCond'] = df.GarageCond.replace(rank_dict)
df['GarageFinish'] = df.GarageFinish.replace(GarFin_dict)

# Since type of garage is either built in the house or separate from the house, we can categorize some of them together:
df['GarageType'] = df['GarageType'].replace(GarType_dict)
df['GarageType'].fillna('None', inplace = True)
df['GarageArea'].fillna('None', inplace = True)
df['GarageCars'].fillna(0, inplace = True)


####################################### 
##            Basement               ##
#######################################

df['BsmtCond'].fillna('None', inplace = True)
df['BsmtQual'].fillna('None', inplace = True)
df['BsmtFinType1'].fillna('None', inplace = True)
df['BsmtFinType2'].fillna('None', inplace = True)
df['BsmtExposure'].fillna('None', inplace = True)
df['TotalBsmtSF'].fillna(0, inplace = True)

BsmtFin_dict = {'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':2, 'LwQ':1, 'Unf':0, 'None':0}
BsmtEx_dict = {'Gd':3, 'Av':2, 'Mn':1, 'No':0, 'None':0}


#  BsmtFinType1 & 2: Rating of basement finished area 1 and 2 (if multiple types)
df['BsmtFinType1'] = df.BsmtFinType1.replace(BsmtFin_dict)
df['BsmtFinType2'] = df.BsmtFinType2.replace(BsmtFin_dict)

# BsmtExposure: Refers to walkout or garden level walls
df['BsmtExposure'] = df.BsmtExposure.replace(BsmtEx_dict)

# BsmtCond: Condition of the basement
df['BsmtCond'] = df.BsmtCond.replace(rank_dict)

# BsmtQual: Evaluates the height of the basement
df['BsmtQual'] = df.BsmtQual.replace(rank_dict)

df['BsmtType1Inter'] = df['BsmtFinType1'] * df['BsmtFinSF1']
df['BsmtType2Inter'] = df['BsmtFinType2'] * df['BsmtFinSF2']


####################################### 
##            Misc                   ##
#######################################

df['MiscFeature'].fillna('None', inplace = True)
df['Alley'].fillna('None', inplace = True)
df['Fence'].fillna('None', inplace = True)
df['FireplaceQu'].fillna('None', inplace = True)
df['Electrical'].fillna('SBrkr', inplace = True)
df['MasVnrType'].fillna('None', inplace = True)
df['MasVnrArea'].fillna(0, inplace = True)
df.loc[df['LotFrontage'].isna(), 'LotFrontage'] = df['LotFrontage'].mean()
df.loc[~df['PoolQC'].isnull(),'PoolQC'] = 1
df['PoolQC'].fillna(0, inplace = True)
df['KitchenQual'] = df.KitchenQual.replace(rank_dict)
df['KitchenQual'].fillna('None', inplace = True)
df["FireplaceQu"] = df.FireplaceQu.replace(rank_dict)
df['age'] = df['YrSold'] - df['YearBuilt']

####################################### 
##            EDA DROP               ##
#######################################

# DROP VARIABLES BASED ON EDA AND FEATURE ENGINEERING FROM IBM_Course_CapstoneProject
drop_features = ['BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 
                 'BsmtHalfBath', 'BsmtType1Inter', 'BsmtType2Inter', 'KitchenAbvGr', 'YrSold', 'YearBuilt',
                 'Exterior2nd', 'Condition2', 'HalfBath', 'MiscFeature', 'GarageCond', 'GarageYrBlt',
                 '1stFlrSF', '2ndFlrSF', 'MSSubClass']

df.drop(drop_features, axis = 1, inplace = True)


####################################### 
##            GET DUMMIES            ##
#######################################

otherFeatures = ['MSZoning', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
'LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 
'RoofStyle', 'RoofMatl', 'Exterior1st', 'ExterQual', 'ExterCond', 'Foundation', 'CentralAir', 
'Electrical', 'LowQualFinSF', 'GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Functional',  'YearRemodAdd',
'PavedDrive', 'WoodDeckSF', 'Fence', 'MoSold', 'YrSold', 'SaleType', 'LotFrontage', 'SaleCondition']


categorical_df = df.loc[:,(df.dtypes == 'object')]

# Generate final df for analysis
final_df = pd.get_dummies(data=df,columns=categorical_df.columns, prefix=categorical_df.columns)

# Drop all 'None' features to get more informative output and avoid multicollinearity 
# (This is relevant for alley, fireplace, pool etc)
final_df = final_df[final_df.columns.drop(list(final_df.filter(regex='_None')))]

# Drop a variable from each categorical feature without a 'None' category
final_df.drop(['MSZoning_RM', 'LotShape_IR1', 'LandContour_Bnk', 'Utilities_AllPub', 'LotConfig_Corner', 'LandSlope_Gtl', 
               'Condition1_RRNe', 'BldgType_TwnhsE', 'HouseStyle_SLvl', 'RoofStyle_Flat', 
               'RoofMatl_ClyTile', 'Exterior1st_AsbShng', 'ExterQual_Ex', 'Foundation_BrkTil',
               'Heating_Wall', 'Electrical_FuseP', 'Functional_Mod', 'GarageType_2Types', 'GarageFinish_0',
               'PavedDrive_P', 'SaleType_COD', 'SaleCondition_Abnorml', 'ExterCond_Fa', 'CentralAir_N', 
               'Street_Grvl'], axis = 1, inplace = True)


####################################### 
##            VIF DROP               ##
#######################################

# DROP VARIABLES WITH LARGE VIF FACTOR OR LACK OF OBSERVATIONS
final_df.drop(['PoolArea', 'Exterior1st_CBlock', 'Exterior1st_VinylSd', 'SaleType_New', 
         'SaleCondition_Partial', 'GarageType_BuiltIn', 'Electrical_FuseA', 'Heating_GasA', 
         'ExterQual_TA', 'ExterCond_TA', 'Condition1_Norm','RoofStyle_Gable', 'RoofMatl_CompShg', 
         'Functional_Typ', 'HouseStyle_2Story', 'Exterior1st_ImStucc', 'Exterior1st_AsphShn', 
         'ExterCond_Po', 'Exterior1st_BrkComm'], axis = 1, inplace = True)



final_df.to_csv('featureSelection.csv', index = False)



