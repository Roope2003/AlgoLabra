"""PCA ja tunnistusputken funktiot Eigenface algoritmille.

Moduuli sisältää datan keskittämisen, kovarianssimatriisin laskemisen,
eigenface koulutuksen, kasvokuvan projisoinnin ja lähimmän naapurin haun.
"""

from math import sqrt
from eigenface.qr import eigendecompose
from eigenface.matrix import Matrix, matrix_multiplication, transpose, matrix_vector_multiplication
from eigenface.vector import dot_product, norm, Vector, normalize, vector_substract
# pylint: disable=invalid-name


def compute_mean(A: Matrix) -> Vector:
    """Laskee keskiarvon matriisin sarakkeille

    Args:
        A (Matrix): Matriisi

    Returns:
        Vector: Vektori jonka alkioina on matriisin sarakkeiden keskiarvot
    """
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
    """Poistaa matriisin keskiarvon jokaisesta sarakkeesta

    Args:
        Data (Matrix): Matriisi
        Mean (Vector): Matriisin keskiarvo

    Returns:
        Matrix: Matriisi josta on poistettu keskiarvo
    """

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
    """Laskee covarianssi matriisin matriisille

    Args:
        A (Matrix): Matriisi

    Returns:
        Matrix: kovarianssi matriisi
    """

    AT=transpose(A)
    C=matrix_multiplication(A,AT)
    M=len(A)
    for i, row in enumerate(C):
        for j, val in enumerate(row):
            C[i][j] = val / M

    return C



def train_eigenfaces(A:Matrix, k:int, iterations:int, tolerance:float):
    """Kouluttaa eigenface-mallin annetusta opetusdatasta.

    Funktio laskee keskiarvon, keskittää datan, muodostaa kovarianssimatriisin
    ja suorittaa ominaisarvohajotelman. Lopuksi valitaan korkeintaan k
    eigenfacea ja lasketaan opetuskuvien painot.

    Args:
        A (Matrix): Opetusdata jossa jokainen rivi on kuva vektorina.
        k (int): Haluttu ylälraja eigenfacejen määrälle.
        iterations (int): QR dekomposition iteraatiot.
        tolerance (float): dekomposition toleranssi.

    Returns:
        tuple[Vector, Vector, Matrix, Matrix]:
            - mean: Datan keskiarvovektori.
            - top_eigenvalues: Valittujen eigenfacejen ominaisarvot.
            - top_eigenvectors: Valitut eigenfacet.
            - train_weights: Opetuskuvien painot eigenface avaruudessa.
    """

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


def predict_face(image:Vector, mean:Vector,top_eigenfaces:Matrix) -> Vector:
    """Projisoi kuvan eigenface avaruuteen

    Args:
        image (Vector): Kuva vektorina
        mean (Vector): datan keskiarvo
        top_eigenfaces (Matrix): valitut eigenfacet

    Returns:
        Vector: kuvan painot eigenface avaruudessa
    """

    centered_image=vector_substract(image,mean)
    image_weight=[]
    for eigenface in top_eigenfaces:
        image_weight.append(dot_product(eigenface,centered_image))
    return image_weight

def euclidean_distance(vector1:Vector, vector2:Vector)-> float:
    """Laskee kahden vektorin välisen euklidisen etäisyyden

    Args:
        vector1 (Vector): Ensimmäinen vektori
        vector2 (Vector: Toinen vektori

    Raises:
        ValueError: Jos vektorien pituudet eivät ole samat

    Returns:
        float: vektorien välinen euklidinen etäisyys
    """

    if len(vector1) != len(vector2):
        raise ValueError("Vectors are not the same length")
    square_sum=0
    n=len(vector1)

    for i in range(n):
        diff=vector1[i]-vector2[i]
        square_sum+=diff*diff
    return sqrt(square_sum)


def nearest_neighbor(image_weight:Vector, train_weights:Matrix)->tuple[int,float]:
    """Etsii lähimmän näytteen datasta annetulle painovektorille

    Args:
        image_weight (Vector): ennustettavan kuvan painovektori
        train_weights (Matrix): datan painovektorit

    Returns:
        tuple[int,float]: int: lähimmän näytteen indeksi, float: lähimmän naapurin etäisyys
    """

    nearest_dist=float("inf")
    best_index=-1
    for i, weight in enumerate(train_weights):
        current_dist = euclidean_distance(weight, image_weight)
        if current_dist < nearest_dist:
            nearest_dist = current_dist
            best_index = i

    return best_index, nearest_dist
