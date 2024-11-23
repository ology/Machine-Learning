from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
# print(iris.keys())
dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
class_names = ['setosa', 'versicolour', 'virginica']
dataset['species'] = [ class_names[x] for x in iris.target ]
# print(dataset)
# print(dataset.shape)
# print(dataset.describe())
# print("sepal length (cm) Sum:", dataset['sepal length (cm)'].sum(), ", Median:", dataset['sepal length (cm)'].median())
# print("sepal length (cm) Min:", dataset['sepal length (cm)'].min(), ", Max:", dataset['sepal length (cm)'].max())
# print(dataset.groupby('species').size())
# print(dataset.iloc[2])
# print(dataset.loc[ dataset['species'] == 'setosa' ])
# print(dataset['species'].value_counts())
# print(dataset.isnull())
# print(dataset.isnull().sum())

# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()
# dataset.hist()
# plt.show()
# scatter_matrix(dataset)
# plt.show()

# setosa = dataset[ dataset['species'] == 'setosa' ]
# versic = dataset[ dataset['species'] == 'versicolour' ]
# virgin = dataset[ dataset['species'] == 'virginica' ]
# xdata = 'petal length (cm)'
# ydata = 'petal width (cm)'
# fig, ax = plt.subplots()
# fig.set_size_inches(13, 7) # adjusting the length and width of plot
# ax.scatter(setosa[xdata], setosa[ydata], label="Setosa", facecolor="blue")
# ax.scatter(versic[xdata], versic[ydata], label="Versicolor", facecolor="green")
# ax.scatter(virgin[xdata], virgin[ydata], label="Virginica", facecolor="red")
# ax.set_xlabel(xdata)
# ax.set_ylabel(ydata)
# ax.grid()
# ax.set_title("Iris petals")
# ax.legend()
# plt.show()

# seaborn
# df = dataset.copy()
# df['species'] = iris['target']
# # print(df)
# # print(df['species'].unique())
# sns.heatmap(df.corr(), cmap="YlGnBu", linecolor='white', linewidths=1, annot=True)
# plt.show()
# print(df.corr(method='pearson'))
# sns.pairplot(df, hue='species')
# plt.show()

values = dataset.values
x = values[:, 0:4]
# print(X)
y = values[:, 4]
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

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
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: mean=%f (std=%f)' % (name, cv_results.mean(), cv_results.std()))
# plt.boxplot(results, labels=names)
# plt.title('Algorithm Comparison')
# plt.show()

# prediction
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print('\nAccuracy:', accuracy_score(y_test, predictions))
# print('\nConfusion:\n', confusion_matrix(Y_test, predictions))
# print('\nClassification:\n', classification_report(Y_test, predictions))
