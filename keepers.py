import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

data = pd.read_csv('keepers.csv', sep='\t')

data.fillna(value=0, inplace=True)

pd.set_option('display.max_rows', data.shape[0]+1)

#print(data)
#print(data.head())
#print(data.tail())
#print(data.shape)

#sns.pairplot(data, x_vars=feature_cols, y_vars=response_col, size=7, aspect=0.7, kind='reg')
#plt.show()

feature_cols = ['age', 'seasons', 'mins', 'apps', 'minapp', 'starts', 'subs', 'ga', 'ga90', 'sota', 'win', 'draw', 'loss', 'cs', 'csperc', 'saves']
response_col = ['rank']

X = data[feature_cols]
#print(X.shape)
y = data[response_col]
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#print('X_train:', X_train)
#print('X_train.shape:', X_train.shape)
#print('y_train:', y_train)
#print('y_train.shape:', y_train.shape)
#print('X_test:', X_test)
#print('y_test:', y_test)
#print('X_train.shape, y_train.shape:', X_train.shape, y_train.shape)
#print('X_test.shape, y_test.shape:', X_test.shape, y_test.shape)
#print('isnan(X_train):', np.isnan(X_train))

k_range = range(1, 26)
scores = []
for k in k_range:
    estimator = KNeighborsClassifier(n_neighbors=k)
    estimator.fit(X_train, y_train.values.ravel())
    y_pred = estimator.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()  # Highest=24

estimator = KNeighborsClassifier(n_neighbors=24)
estimator.fit(X_train, y_train.values.ravel())

#print(estimator.intercept_)
#print(estimator.coef_)
#z = zip(feature_cols, estimator.coef_[0])
#z = set(z)
#print(z)

y_pred = estimator.predict(X_test)

print('mean_absolute_error:', metrics.mean_absolute_error(y_test, y_pred))
print('mean_squared_error:', metrics.mean_squared_error(y_test, y_pred))
print('root mean_squared_error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#print(X_test.iloc[[1]])
#print(y_test.iloc[[1]]) # 29
datum = data.iloc[[88]]
#print(datum)
X = datum[feature_cols]
y_pred = estimator.predict(X)
print('Prediction for 29:', y_pred) # 27

#print(X_test.iloc[[2]])
#print(y_test.iloc[[2]]) # 18
datum = data.iloc[[163]]
#print(datum)
X = datum[feature_cols]
y_pred = estimator.predict(X)
print('Prediction for 18:', y_pred) # 16
