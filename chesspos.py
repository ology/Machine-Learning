import chess.pgn
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sys

if __name__ == "__main__":
    size = 8
    squares_n = size * size
    pieces_n = 12
    dim = squares_n * pieces_n
    blacks = ['p','n','b','r','q','k']
    whites = ['P','N','B','R','Q','K']
    pieces = blacks + whites

    pgns = sys.argv[1:]
    X = []
    Y = []
    i = 0
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
        for move in game.mainline_moves():
            if (board.turn == chess.WHITE and player == chess.WHITE) or (board.turn == chess.BLACK and player == chess.BLACK):
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
    # print(X[0], Y[0])

    x_df = pd.DataFrame(X)
    y_df = pd.DataFrame(Y)
    # print(x_df)

    encoder = OneHotEncoder()

    x_encoded_data = encoder.fit_transform(x_df)
    y_encoded_data = encoder.fit_transform(y_df)
    # print(type(x_encoded_data))
    x_encoded_df = pd.DataFrame(x_encoded_data.toarray())
    y_encoded_df = pd.DataFrame(y_encoded_data.toarray())
    # print(x_encoded_df.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_encoded_df, y_encoded_df, train_size = 0.8)
    model = LinearRegression().fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(train_score, test_score)
    # y_pred = model.predict(x_test)

    # x = fen2hot(lookup_fen, chess.STARTING_FEN).reshape((1, dim))
    # print(x.shape)
    # y = model.predict(x)
    # print(y.reshape((squares_n, pieces_n)))
    # z = hot2fen(lookup_hot, y.reshape((squares_n, pieces_n)))
    # print(z)

    # plt.scatter(x_test, y_pred, color='b')
    # plt.plot(x_test, y_pred, color='k')
    # plt.show()
