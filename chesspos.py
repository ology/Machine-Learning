import chess
import numpy as np

squares_n = 64
pieces_n = 12
pieces = ['p','n','b','r','q','k','P','N','B','R','Q','K']
lookup_fen = {}
# 'p': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
# 'n': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
# ...
# 'K': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool),
for i, piece in enumerate(pieces):
    mask = np.zeros(pieces_n, dtype=int)
    mask[i] = 1
    lookup_fen[piece] = np.array(mask, dtype=bool)
# print(lookup_fen)

# TODO ...
lookup_hot = {tuple(value): key for key, value in lookup_fen.items()}
# print(lookup_hot)
lookup_hot[tuple([False for _ in range(pieces_n)])] = None
# print(lookup_hot)

def piece2vec(piece):
    return lookup_fen.get(piece, None)

def vec2piece(vec):
    return lookup_hot[tuple(vec)]

def fen2hot(fen):
    i = 0
    board = np.zeros((squares_n, pieces_n), dtype=bool)
    for p in fen:
        if p == ' ':
            break
        if p == '/':
            continue
        encoding = piece2vec(p)
        if encoding is not None:
            board[i, :] = encoding
            i = i + 1
        else:
            i = i + int(p)
    return board

def hot2fen(hot):
    j = 0
    board = ''
    for i in range(squares_n):
        if i % 8 == 0 and i > 0:
            if j > 0:
                board += str(j)
                j = 0
            board += '/'
        key = vec2piece(hot[i, :])
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
    x = fen2hot(chess.STARTING_FEN)
    # print(x)
    y = hot2fen(x)
    print(y)
