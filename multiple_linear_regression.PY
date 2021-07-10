# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data => creating  dumpy  variables 
from sklearn.preprocessing import  OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[3])],remainder= 'passthrough')
X= np.array(ct.fit_transform(X))
labelencoder = LabelEncoder()

# Avoiding dummy variable trap
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting multiple linear regression to training  set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,1,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xopt_train, Xopt_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression to training  set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xopt_train, y_train)

# Predicting the test results
y_opt = regressor_OLS.predict(Xopt_test)












