import chess.pgn
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys

def make_lookup(pieces_n, blacks, whites):
    lookup_fen = {}
    pieces = blacks + whites
    # 'p': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
    # 'n': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
    # ...
    # 'K': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool),
    for i, piece in enumerate(pieces):
        mask = np.zeros(pieces_n, dtype=int)
        mask[i] = 1
        lookup_fen[piece] = np.array(mask, dtype=bool)
    # print(lookup_fen)
    # TODO What/why
    lookup_hot = {tuple(value): key for key, value in lookup_fen.items()}
    lookup_hot[tuple([False for _ in range(pieces_n)])] = None
    return lookup_fen, lookup_hot

def piece2vec(lookup_fen, piece):
    return lookup_fen.get(piece, None)

def vec2piece(lookup_hot, vec):
    return lookup_hot[tuple(vec)]

def fen2hot(lookup_fen, fen):
    i = 0
    board = np.zeros((squares_n, pieces_n), dtype=bool)
    for p in fen:
        if p == ' ':
            break
        if p == '/':
            continue
        encoding = piece2vec(lookup_fen, p)
        if encoding is not None:
            board[i, :] = encoding
            i = i + 1
        else:
            i = i + int(p)
    return board

def hot2fen(lookup_hot, hot):
    j = 0
    board = ''
    for i in range(squares_n):
        if i % 8 == 0 and i > 0:
            if j > 0:
                board += str(j)
                j = 0
            board += '/'
        key = vec2piece(lookup_hot, hot[i, :])
        if key is not None:
            if not j == 0:
                board += str(j)
                j = 0
            board += key
        else:
            j += 1
    if not j == 0:
        board += str(j)
        j = 0
    return board

if __name__ == "__main__":
    size = 8
    squares_n = size * size
    pieces_n = 12
    dim = squares_n * pieces_n
    blacks = ['p','n','b','r','q','k']
    whites = ['P','N','B','R','Q','K']
    pieces = blacks + whites

    lookup_fen, lookup_hot = make_lookup(pieces_n, blacks, whites)
    lookup_fen_swap, lookup_hot_swap = make_lookup(pieces_n, whites, blacks)
    # x = fen2hot(lookup_fen, chess.STARTING_FEN)
    # y = hot2fen(lookup_hot, x)
    # print(x,y)

    pgns = sys.argv[1:]
    # pairs = {}
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
                # if not key in pairs:
                #     pairs[key] = []
                board.push(move)
                val = board.fen()
                # pairs[key].append(val)
                if player == chess.WHITE:
                    X.append(fen2hot(lookup_fen, key))
                    Y.append(fen2hot(lookup_fen, val))
                else:
                    X.append(fen2hot(lookup_fen_swap, key))
                    Y.append(fen2hot(lookup_fen_swap, val))
            else:
                board.push(move)
                fen = board.fen()
    # print(len(pairs))
    print(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8)
    # print(x_train) #, x_test, y_train, y_test)
    # model = LinearRegression().fit(x_train, y_train)
    # model.score(x_train, y_train)
    # model.score(x_test, y_test)
