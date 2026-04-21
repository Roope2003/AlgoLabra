from matrix import Matrix, MatrixMultiplication, Transpose, MatrixVectorMultiplication
from vector import DotProduct, Norm, Vector, Normalize, VectorSubstract
from QR import eigendecompose
from math import sqrt
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
    C=MatrixMultiplication(A,AT)
    M=len(A)
    for i in range(len(C)):
        for j in range(len(C[0])):
            C[i][j] = C[i][j]/ M

    return C



def train_eigenfaces(A:Matrix, k:int, iterations:int, tolerance:float):
    mean=compute_mean(A)
    center=center_data(A,mean)
    L=covariance(center)
    eigenvalues,eigenvectors=eigendecompose(L,iterations,tolerance)

    valid_eigenvalues=[]
    valid_eigenvectors=[]
    center_transpose=Transpose(center)
    for i in range(len(eigenvectors)):
        v_i=MatrixVectorMultiplication(center_transpose,eigenvectors[i])

        if Norm(v_i) > 0:
            v_i_normalized=Normalize(v_i)
            valid_eigenvalues.append(eigenvalues[i])
            valid_eigenvectors.append(v_i_normalized)
    k_eff=min(k,len(valid_eigenvectors))
    top_eigenvalues=valid_eigenvalues[:k_eff]
    top_eigenvectors=valid_eigenvectors[:k_eff]
    train_weights: Matrix=[]

    for image in center:
        image_weight=[]

        for eigenface in top_eigenvectors:
            image_weight.append(DotProduct(eigenface,image))
        train_weights.append(image_weight)
    return mean, top_eigenvalues, top_eigenvectors, train_weights


def predict_face(image:Vector, mean:Vector,top_eigenfaces:Matrix):
    centered_image=VectorSubstract(image,mean)
    image_weight=[]
    for eigenface in top_eigenfaces:
        image_weight.append(DotProduct(eigenface,centered_image))
    return image_weight

def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors are not the same length")
    square_sum=0
    n=len(vector1)

    for i in range(n):
        diff=vector1[i]-vector2[i]
        square_sum+=diff*diff
    return sqrt(square_sum)


def nearest_neighbor(image_weight:Vector, train_weights:Matrix):
    nearest_dist=float("inf")
    best_index=-1

    for i in range(len(train_weights)):
        current_dist=euclidean_distance(train_weights[i],image_weight)
        if current_dist< nearest_dist:
            nearest_dist=current_dist
            best_index=i
    return (best_index,nearest_dist)