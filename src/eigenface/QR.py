from matrix import Matrix, MatrixMultiplication, Transpose, identitymatrix
from vector import DotProduct, Norm, VectorSubstract,VectorDivisionScalar
from math import sqrt


def qr_decompose(A: Matrix):
    rows = len(A)
    cols = len(A[0])

    Q = [[0 for c in range(cols)] for r in range(rows)] #Q on matriisi johon tulee orthonormaalit sarake vektoriy
    R = [[0 for c in range(cols)] for c in range(cols)] # R on matriisi josta tulee yläkolmio matriisi

    for j in range(cols):
        v = [A[i][j] for i in range(rows)]  #otetaan matriisista j:s sarake
        for i in range(j):
            q_i=[Q[r][i] for r in range(rows)] # otetaan Q:n i:s sarake

            R[i][j]=DotProduct(q_i,v) #pistetulo keroo kuinka paljon v osoittaa q_i suuntaan

            v = VectorSubstract(v, [R[i][j] * x for x in q_i]) #poistettaan v:stä q_i suuntaan osoittava osa

        R[j][j] = Norm(v) #lasketaan normi eli vektorin pituus
        q_j = VectorDivisionScalar(v,R[j][j]) #tehdään uudesta v:stä yksikkövektori eli normalisoidaan

        for r in range(rows):
            Q[r][j] = q_j[r] #asetetaan q_j Q:n j:nnelle sarakkeelle

    return Q, R


def eigendecompose(A:Matrix, iterations:int,toleranssi:float):
    rows = len(A)
    cols = len(A[0])

    A_k= [row[:] for row in A]
    Q_i= identitymatrix(len(A))

    for i in range(iterations):
        Q, R = qr_decompose(A_k)
        A_k = MatrixMultiplication(R, Q)
        Q_i=MatrixMultiplication(Q_i,Q)

        diag_sum=0
        for r in range(rows):
            for c in range(cols):
                if r!=c:
                    diag_sum += A_k[r][c] * A_k[r][c]

        diag_norm=sqrt(diag_sum)

        if diag_norm < toleranssi:
            break

    eigenvalues= [A_k[i][i] for i in range(rows)]
    eigenvectors= Transpose(Q_i) # transpoosi jotta helpompi indeksoida

    order=sorted(range(len(eigenvalues)), key=lambda x: eigenvalues[x], reverse=True) #indeksi lista
    eigenvalues = [eigenvalues[i] for i in order] #molemmat lajitellaan suuruusjärjestykseen
    eigenvectors= [eigenvectors[i] for i in order]

    return eigenvalues, eigenvectors