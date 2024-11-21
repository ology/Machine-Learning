import chess.pgn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
import re
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
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sys

def process_pgns(pgns):
    X = []
    Y = []
    i = 0
    limit = 5 # limit j number of moves below
    for pgn in pgns:
        content = open(pgn)
        try:
            game = chess.pgn.read_game(content)
        except:
            continue
        i = i + 1
        print(i, pgn, game.headers['White'], 'vs', game.headers['Black'])
        board = game.board()
        if re.search('kasparov', game.headers["White"], re.IGNORECASE):
            player = chess.WHITE
            fen = chess.STARTING_FEN
        else:
            player = chess.BLACK
        j = 0
        for move in game.mainline_moves():
            if j >= limit:
                break
            if (board.turn == chess.WHITE and player == chess.WHITE) or (board.turn == chess.BLACK and player == chess.BLACK):
                j = j + 1
                key = fen
                board.push(move)
                val = board.fen()
                if player == chess.BLACK:
                    key = key.swapcase()
                    val = val.swapcase()
                X.append(key)
                Y.append(val)
            else:
                board.push(move)
                fen = board.fen()
    return X, Y

if __name__ == "__main__":
    # size = 8
    # squares_n = size * size
    # pieces_n = 12
    # dim = squares_n * pieces_n
    # blacks = ['p','n','b','r','q','k']
    # whites = ['P','N','B','R','Q','K']
    # pieces = blacks + whites

    pgns = sys.argv[1:]
    X, Y = process_pgns(pgns)
    print(len(X), len(Y))

    x_df = pd.DataFrame(X)
    y_df = pd.DataFrame(Y)
    print(x_df.shape, y_df.shape)
    # print(x_df)

    labelencoder = LabelEncoder()
    x_encoded_data = labelencoder.fit_transform(x_df.values.ravel())
    y_encoded_data = labelencoder.fit_transform(y_df.values.ravel())
    print(x_encoded_data.shape, y_encoded_data.shape)

    x_encoded_df = pd.DataFrame(x_encoded_data)
    y_encoded_df = pd.DataFrame(y_encoded_data)
    # x_encoded_df = x_encoded_df.values.ravel()
    # y_encoded_df = y_encoded_df.values.ravel()
    print(x_encoded_df.shape, y_encoded_df.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(x_encoded_df.iloc[:, 0], y_encoded_df.iloc[:, 0], train_size = 0.8)

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
    # model = SVC(gamma='auto')
    # model.fit(X_train, Y_train)
    # predictions = model.predict(X_test)
    # print('\nAccuracy:', accuracy_score(Y_test, predictions))
    # print('\nConfusion:\n', confusion_matrix(Y_test, predictions))
    # print('\nClassification:\n', classification_report(Y_test, predictions))
 
    # plt.scatter(x_test, y_pred, color='b')
    # plt.plot(x_test, y_pred, color='k')
    # plt.show()
