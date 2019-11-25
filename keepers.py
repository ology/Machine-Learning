import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv('keepers.csv', sep='\t')

data.fillna(value=0, inplace=True)

pd.set_option('display.max_rows', data.shape[0]+1)

#print(data)
#print(data.head())
#print(data.tail())
#print(data.shape)

feature_cols = ['age', 'seasons', 'mins', 'apps', 'minapp', 'starts', 'subs', 'ga', 'ga90', 'sota', 'win', 'draw', 'loss', 'cs', 'csperc', 'saves']
response_col = ['rank']

#sns.pairplot(data, x_vars=feature_cols, y_vars=response_col, size=7, aspect=0.7, kind='reg')
#plt.show()

X = data[feature_cols]
y = data[response_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#print(X_train)
#print(X_test)
#print(y_test)
#print(y_train)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
#print(np.isnan(X_train))

estimator = LinearRegression()
estimator.fit(X_train, y_train)

#print(estimator.intercept_)
#print(estimator.coef_)
z = zip(feature_cols, estimator.coef_[0])
z = set(z)
print(z)

y_pred = estimator.predict(X_test)

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#print(X_test.iloc[[1]])
#print(y_test.iloc[[1]]) # 29
datum = data.iloc[[88]]
#print(datum)
X = datum[feature_cols]
y = datum[response_col]
y_pred = estimator.predict(X)
print(y_pred) # 25.95248441

#print(X_test.iloc[[2]])
#print(y_test.iloc[[2]]) # 18
datum = data.iloc[[163]]
#print(datum)
X = datum[feature_cols]
y = datum[response_col]
y_pred = estimator.predict(X)
print(y_pred) # 13.79459475
