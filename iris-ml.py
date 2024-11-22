from matplotlib import pyplot as plt
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
class_names = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']
dataset['class'] = [ class_names[x] for x in iris.target ]
# print(dataset)
# print(dataset.shape)
# print(dataset.describe())
# print(dataset.groupby('class').size())

# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()
# dataset.hist()
# plt.show()
# scatter_matrix(dataset)
# plt.show()

values = dataset.values
X = values[:, 0:4]
# print(X)
y = values[:, 4]
# print(y)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# inspect algorithms
models = []
models.append(('  LR', LogisticRegression(solver='liblinear')))
models.append((' LDA', LinearDiscriminantAnalysis()))
models.append((' KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('  NB', GaussianNB()))
models.append((' SVM', SVC(gamma='auto')))
print('Evaluate algorithms:')
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: mean=%f (std=%f)' % (name, cv_results.mean(), cv_results.std()))
# plt.boxplot(results, labels=names)
# plt.title('Algorithm Comparison')
# plt.show()

# prediction
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
print('\nAccuracy:', accuracy_score(Y_test, predictions))
# print('\nConfusion:\n', confusion_matrix(Y_test, predictions))
# print('\nClassification:\n', classification_report(Y_test, predictions))
