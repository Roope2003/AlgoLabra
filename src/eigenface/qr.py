from math import sqrt
from eigenface.matrix import Matrix, matrix_multiplication, transpose, identitymatrix
from eigenface.vector import dot_product, norm, vector_substract, vector_division_scalar

# pylint: disable=invalid-name


def qr_decompose(A: Matrix):
    rows = len(A)
    cols = len(A[0])

    # Q on matriisi johon tulee orthonormaalit sarake vektoriy.
    Q = [[0 for c in range(cols)] for r in range(rows)]
    # R on matriisi josta tulee yläkolmio matriisi.
    R = [[0 for c in range(cols)] for c in range(cols)]

    for j in range(cols):
        v = [A[i][j] for i in range(rows)]  # otetaan matriisista j:s sarake
        for i in range(j):
            q_i = [Q[r][i] for r in range(rows)]  # otetaan Q:n i:s sarake

            # Pistetulo kertoo kuinka paljon v osoittaa q_i suuntaan.
            R[i][j] = dot_product(q_i, v)

            # Poistetaan v:stä q_i suuntaan osoittava osa.
            v = vector_substract(v, [R[i][j] * x for x in q_i])

        R[j][j] = norm(v)  # lasketaan normi eli vektorin pituus
        # Tehdaan uudesta v:sta yksikkovektori eli normalisoidaan.
        q_j = vector_division_scalar(v, R[j][j])

        for r in range(rows):
            Q[r][j] = q_j[r]  # asetetaan q_j Q:n j:nnelle sarakkeelle

    return Q, R

def eigendecompose(A: Matrix, iterations: int, toleranssi: float):
    # pylint: disable=too-many-locals,too-many-nested-blocks
    rows = len(A)
    cols = len(A[0])

    A_k = [row[:] for row in A]
    Q_i = identitymatrix(len(A))

    for i in range(iterations):
        Q, R = qr_decompose(A_k)
        A_k = matrix_multiplication(R, Q)
        Q_i = matrix_multiplication(Q_i, Q)

        diag_sum = 0
        for r in range(rows):
            for c in range(cols):
                if r != c:
                    diag_sum += A_k[r][c] * A_k[r][c]

        diag_norm = sqrt(diag_sum)

        if diag_norm < toleranssi:
            break

    eigenvalues = [A_k[i][i] for i in range(rows)]
    eigenvectors = transpose(Q_i)  # transpoosi jotta helpompi indeksoida

    order = sorted(range(len(eigenvalues)),key=lambda x:eigenvalues[x],reverse=True) # indeksi lista
    # molemmat lajitellaan suuruusjärjestykseen
    eigenvalues = [eigenvalues[i] for i in order]
    eigenvectors = [eigenvectors[i] for i in order]

    return eigenvalues, eigenvectors
