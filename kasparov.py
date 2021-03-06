# Write-up: http://techn.ology.net/investigating-kasparov-chess-moves-with-scikit-learn/

import chess # https://python-chess.readthedocs.io/en/latest/
import chess.pgn
import re
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


def number_of_moves(node):
    i = 0
    while not node.is_end():
        i = i + 1
        next_node = node.variations[0]
        node = next_node
    return i


def get_nth_move(node, n):
    i = 0
    move = ''
    while not node.is_end():
        i = i + 1
        next_node = node.variations[0]
        if i == n:
            move = node.board().san(next_node.move)
            break
        node = next_node
    return move


def board_transform(b):
    fen = b.fen().split(' ')[0].split('/')
    values = {
        'p' : 10,
        'r' : 11,
        'n' : 12,
        'b' : 13,
        'q' : 14,
        'k' : 15,
        'P' : 20,
        'R' : 21,
        'N' : 22,
        'B' : 23,
        'Q' : 24,
        'K' : 25
    }
    transformed = []
    for row in fen:
        for r in list(row):
            if (re.search('^\d+$', r)):
                for i in range(1, int(r) + 1):
                    transformed.append('0')
            else:
                transformed.append(str(values.get(r)))
    return transformed


def get_pgn(path):
    file_list = []
    for fn in os.listdir(path):
        file_list.append(fn)
    return file_list


def create_csv(threshold):
    threshold = threshold * 2
    # Write CSV
    with open('kasparov.csv', 'wb') as csvfile:
        fh = csv.writer(csvfile)
        # Get the files to parse
        path = 'pgn'
        pgn_files = get_pgn(path)
        for pgn_file in pgn_files:
            # Read the game from the given PGN file
            file_name = path + '/' + pgn_file
            #print 'Reading: ' + file_name
            with open(file_name) as pgn:
                game = chess.pgn.read_game(pgn)
            # Decide what player is Kasperov
            if 'Kasparov' in game.headers['White']:
                player = 1
            else:
                player = 0
            # Get the number of game moves
            moves = number_of_moves(game)
            # Restrict to threshold
            if (threshold > 0) and (moves >= threshold):
                moves = threshold
            # Get a new board
            board = chess.Board()
            # For each move...
            for i in range(1, moves + 1):
                move = get_nth_move(game, i)
                # Write the CSV transformed board for the player move
                if (player and (i % 2)) or (not(player) and not(i % 2)):
                    t = board_transform(board)
                    # Prepend the move number
                    t.insert(0, i)
                    # Add the subsequent move
                    t.append(move)
                    fh.writerow(t)
                board.push_san(move) # Make the move


def train():
    data = pd.read_csv('kasparov.csv', header=None)
    X = data.loc[:, 0:list(data)[-2]]
    y = data.loc[:, list(data)[-1]]
    #print type(X), X.shape
    #print type(y), y.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #print X_train.shape, y_train.shape
    #print X_test.shape, y_test.shape
    return X_train, X_test, y_train, y_test


def decision_tree(X_train, X_test, y_train, y_test):
    decisiontree = DecisionTreeClassifier()
    model = decisiontree.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)


create_csv(0)
X_train, X_test, y_train, y_test = train()


def trial_and_error():

    # K Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred)) # 0.12101669195751139 for all moves in game


    # Better k?
    k_range = range(1, 26)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))

    plt.plot(k_range, scores)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.show()


    # Multinomial Naive Bayes
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred)) # 0.07169954476479515 for all moves in game


    # Support Vector Machine
    from sklearn import svm
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred)) # 0.15440060698027314 for all moves in game


    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    accuracy = decision_tree(X_train, X_test, y_train, y_test)
    print(accuracy) # 0.17071320182094082 for all moves in game


    create_csv(3)
    X_train, X_test, y_train, y_test = train()
    accuracy = decision_tree(X_train, X_test, y_train, y_test)
    print(accuracy)

    create_csv(6)
    X_train, X_test, y_train, y_test = train()
    accuracy = decision_tree(X_train, X_test, y_train, y_test)
    print(accuracy)

    create_csv(12)
    X_train, X_test, y_train, y_test = train()
    accuracy = decision_tree(X_train, X_test, y_train, y_test)
    print(accuracy)

    k_range = range(1, 15)
    scores = []
    for k in k_range:
        create_csv(k)
        X_train, X_test, y_train, y_test = train()
        accuracy = decision_tree(X_train, X_test, y_train, y_test)
        print(accuracy)
        scores.append(accuracy)

    plt.plot(k_range, scores)
    plt.xlabel('Moves')
    plt.ylabel('Accuracy')
    plt.show()
