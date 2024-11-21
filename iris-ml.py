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
