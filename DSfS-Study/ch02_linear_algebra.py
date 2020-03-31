import math
from typing import List, Tuple, Callable

Vector = List[float]

height_weight_age = [
    70,  # inches,
    170, # pounds,
    40,  # years
]

grades = [
    95,  # exam1
    80,  # exam2
    75,  # exam3
    62,  # exam4
]

def add(u: Vector, v: Vector) -> Vector:
    assert len(u) == len(v), 'vectors must be the same length'
    return [u_i + v_i for u_i, v_i in zip(u, v)]

#assert add([1,2,3], [4,5,6]) == [5,7,9]
#print('add:', add([1,2,3], [4,5,6]))

def subtract(u: Vector, v: Vector) -> Vector:
    assert len(u) == len(v), 'vectors must be the same length'
    return [u_i - v_i for u_i, v_i in zip(u, v)]

#assert subtract([5,7,9], [4,5,6]) == [1,2,3]
#print('subtract:', subtract([5,7,9], [4,5,6]))

def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors, 'vectors are required'
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), 'all vectors must be the same length'
    return [
        sum(vector[i] for vector in vectors)
        for i in range(num_elements)
    ]

#assert vector_sum([[1,2],[3,4],[5,6],[7,8]]) == [16,20]
#print('vector_sum:', vector_sum([[1,2],[3,4],[5,6],[7,8]]))

def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]

#assert scalar_multiply(2, [1,2,3]) == [2,4,6]
#print('scalar_multiply:', scalar_multiply(2, [1,2,3]))

def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

#assert vector_mean([[1,2],[3,4],[5,6]]) == [3,4]
#print('vector_mean:', vector_mean([[1,2],[3,4],[5,6]]))

def dot(u: Vector, v: Vector) -> float:
    assert len(u) == len(v), 'vectors must be the same length'
    return sum(u_i * v_i for u_i, v_i in zip(u, v))

#assert dot([1,2,3], [4,5,6]) == 32
#print('dot:', dot([1,2,3], [4,5,6]))

def sum_of_squares(v: Vector) -> float:
    return dot(v, v)

#assert sum_of_squares([1,2,3]) == 14
#print('sum_of_squares:', sum_of_squares([1,2,3]))

def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))

#assert magnitude([3,4]) == 5
#print('magnitude:', magnitude([3,4]))

def distance(u: Vector, v: Vector) -> float:
    return magnitude(subtract(u, v))

#assert distance([2,3,4,2], [1,-2,1,3]) == 6
#print('distance:', distance([2,3,4,2], [1,-2,1,3]))

######################################################################

Matrix = List[List[float]]

def shape(M: Matrix) -> Tuple[int, int]:
    num_rows = len(M)
    num_cols = len(M[0]) if M else 0
    return num_rows, num_cols

#assert shape([[1,2,3], [4,5,6]]) == (2, 3)
#print('shape:', shape([[1,2,3], [4,5,6]]))

def get_row(M: Matrix, i: int) -> Vector:
    return M[i]

def get_col(M: Matrix, j: int) -> Vector:
    return [
        M_i[j]
        for M_i in M
    ]

#assert get_row([[1,2,3], [4,5,6]], 1) == [4,5,6]
#print('get_row:', get_row([[1,2,3], [4,5,6]], 1))

#assert get_col([[1,2,3], [4,5,6]], 1) == [2,5]
#print('get_col:', get_col([[1,2,3], [4,5,6]], 1))

def make_matrix(
    num_rows: int,
    num_cols: int,
    entry_fn: Callable[[int, int], float]
) -> Matrix:
    """
    Return a num_rows x num_cols matrix whose (i,j)-th entry is entry_fn(i,j)
    """
    return [
        [
            entry_fn(i, j)             # given i, create a list
            for j in range(num_cols)   # [entry_fn(i, 0), ... ]
        ]
        for i in range(num_rows)       # create one list for each i
    ]

def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

#print('identity_matrix:', identity_matrix(5))
