#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 45].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[: , 0] = labelencoder_x.fit_transform(x[: , 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x=onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy variable trap
x = x[:, 1:]

#Spliting the dataset to Training and test set
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=0.25, random_state=0)

#Fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#predicting the test results
ypred = regressor.predict(xtest)
SS_Residual = sum((ytest-ypred)**2)
SS_Total = sum((ytest-np.mean(ytest))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(ytest)-1)/(len(ytest)-xtest.shape[1]-1)

#using backward elimination
import statsmodels.formula.api as sm
x = np.append(arr= np.ones((2500,1)).astype(int), values = x, axis = 1)
xopt = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]]
regressorOLS = sm.OLS(endog= y, exog=xopt).fit()
regressorOLS.summary()
xopt = x[:, [0,5,18,35]]
regressorOLS = sm.OLS(endog= y, exog=xopt).fit()
regressorOLS.summary()

#Fitting the model
xtrain, xtest, ytrain, ytest= train_test_split(xopt, y, test_size=0.25, random_state=0)
regressor.fit(xtrain, ytrain)

#predicting the test results
ypred = regressor.predict(xtest)
SS_Residual = sum((ytest-ypred)**2)
SS_Total = sum((ytest-np.mean(ytest))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(ytest)-1)/(len(ytest)-xtest.shape[1]-1)


