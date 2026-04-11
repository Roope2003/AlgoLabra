from .matrix import Matrix, MatrixMultiplication, Transpose, ValidMatrix
from .vector import DotProduct, Norm, Vector


def compute_mean(A: Matrix) -> Vector:

    rows = len(A)
    cols = len(A[0])

    mean: Vector = []
    for c in range(cols):
        summa=0.0
        for r in range(rows):
            summa+=A[r][c]
        mean.append(summa/rows)
    return mean

def center_data(Data: Matrix, Mean: Vector) -> Matrix:

    rows = len(Data)
    cols = len(Data[0])

    centered: Matrix = []

    for i in range(rows):
        centered.append([])
        for j in range(cols):
            centered_val = Data[i][j] - Mean[j]
            centered[i].append(centered_val)

    return centered


def covariance(A:Matrix)->Matrix:
    AT=Transpose(A)
    C=MatrixMultiplication(AT,A)
    M=len(A)
    for i in range(len(C)):
        for j in range(len(C[0])):
            C[i][j] = C[i][j]/ M

    return C