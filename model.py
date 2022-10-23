#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:43:14 2022

@author: chitresh
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/home/chitresh/Desktop/data science/project/eda_data.csv")

#choose relevant columns
df.columns

df_model = df[['avg_salary','Size' ,'Rating','Num_Comp' ,'same_sate' ,'Type of ownership', 'Industry', 'Sector', 'Revenue', 'job_state', 'job_simp', 'hourly', 'Employer_provided', 'seniority', 'python_yn', 'spark_yn', 'aws_yn', 'excel_yn', 'desc_len', 'age']]

#dummy data
df_dum = pd.get_dummies(df_model)

#test train
from sklearn.model_selection import train_test_split

x = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#multiple linear regression
import statsmodels.api as sm

x_sm = x = sm.add_constant(x)
model = sm.OLS(y, x_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV

lm = LinearRegression()
lm.fit(x_train, y_train)

np.mean(cross_val_score(lm, x_train, y_train, scoring='neg_mean_absolute_error',cv = 3))

#lasso regression
lm_las = Lasso()
np.mean(cross_val_score(lm_las, x_train, y_train, scoring='neg_mean_absolute_error',cv = 3))

alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml, x_train, y_train, scoring='neg_mean_absolute_error',cv = 3)))
    
plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns=['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]

#random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, x_train, y_train, scoring='neg_mean_absolute_error',cv = 3))

#tune model gridsearchcv
parameters = {'n_estimators':range(10, 300, 10), 'criterion':('squared_error', 'mae'), 'max_features':('sqrt', 'log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(x_train, y_train)
# rest ensembles



































