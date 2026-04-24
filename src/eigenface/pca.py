from math import sqrt
from eigenface.qr import eigendecompose
from eigenface.matrix import Matrix, matrix_multiplication, transpose, matrix_vector_multiplication
from eigenface.vector import dot_product, norm, Vector, normalize, vector_substract

# pylint: disable=invalid-name

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
    AT=transpose(A)
    C=matrix_multiplication(A,AT)
    M=len(A)
    for i, row in enumerate(C):
        for j, val in enumerate(row):
            C[i][j] = val / M

    return C



def train_eigenfaces(A:Matrix, k:int, iterations:int, tolerance:float):
    # pylint: disable=too-many-locals
    mean=compute_mean(A)
    center=center_data(A,mean)
    L=covariance(center)
    eigenvalues,eigenvectors=eigendecompose(L,iterations,tolerance)

    valid_eigenvalues=[]
    valid_eigenvectors=[]
    center_transpose=transpose(center)

    for i, vector in enumerate(eigenvectors):
        v_i = matrix_vector_multiplication(center_transpose, vector)

        if norm(v_i) > 0:
            v_i_normalized = normalize(v_i)
            valid_eigenvalues.append(eigenvalues[i])
            valid_eigenvectors.append(v_i_normalized)

    k_eff=min(k,len(valid_eigenvectors))
    top_eigenvalues=valid_eigenvalues[:k_eff]
    top_eigenvectors=valid_eigenvectors[:k_eff]
    train_weights: Matrix=[]

    for image in center:
        image_weight=[]

        for eigenface in top_eigenvectors:
            image_weight.append(dot_product(eigenface,image))
        train_weights.append(image_weight)
    return mean, top_eigenvalues, top_eigenvectors, train_weights


def predict_face(image:Vector, mean:Vector,top_eigenfaces:Matrix):
    centered_image=vector_substract(image,mean)
    image_weight=[]
    for eigenface in top_eigenfaces:
        image_weight.append(dot_product(eigenface,centered_image))
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
    for i, weight in enumerate(train_weights):
        current_dist = euclidean_distance(weight, image_weight)
        if current_dist < nearest_dist:
            nearest_dist = current_dist
            best_index = i

    return best_index, nearest_dist
