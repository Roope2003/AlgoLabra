"""Matriisilaskennan perusoperaatiot Eigenface algoritmille.

Moduuli sisältää matriisien validoinnin, transponoinnin, matriisikertolaskun,
matriisi-vektori-kertolaskun ja identiteettimatriisin luonnin.
"""

from eigenface.vector import Vector

Matrix = list[list[float]]
# pylint: disable=invalid-name


def valid_matrix(A: Matrix) -> bool:
    """Tarkistaa onko annettu matriisi kelvollinen

    Args:
        A (Matrix): Matriisi jota tarkistetaan

    Returns:
        bool: True, jos matriisi on kelvollinen. Muuten False
    """

    if len(A) == 0:
        return False
    if len(A[0]) == 0:
        return False

    cols = len(A[0])
    for row in A:
        if len(row) != cols:
            return False

    return True


def transpose(A: Matrix) -> Matrix:
    """Laskee matriisin transpoosin

    Args:
        A (Matrix): Transponitava matriisi

    Raises:
        ValueError: Jos annettu matriisi ei ole kelvollinen

    Returns:
        Matrix: Uusi matriisi jonka rivit ja sarakkeet ovat vaihtuneet
    """

    if not valid_matrix(A):
        raise ValueError("Invalid matrix")

    cols = len(A[0])
    rows = len(A)

    transpoosi: Matrix = []
    for c in range(cols):
        uusi_rivi: list[float] = []
        for r in range(rows):
            uusi_rivi.append(A[r][c])
        transpoosi.append(uusi_rivi)

    return transpoosi


def matrix_multiplication(A: Matrix, B: Matrix) -> Matrix:
    """Laskee kahden matriisin välisen tulon

    Args:
        A (Matrix): ensimmäinen matriisi
        B (Matrix): toinen matriisi

    Raises:
        ValueError: Jos jompikumpi matriisi ei ole kelvollinen
        ValueError: JOs matriisiin dimensiot eivät sovi kertolaskuun

    Returns:
        Matrix: Matriisitulon A * B
    """

    if not valid_matrix(A) or not valid_matrix(B):
        raise ValueError("Invalid matrix")

    rivit_a = len(A)
    sarakkeet_a = len(A[0])
    rivit_b = len(B)
    sarakkeet_b = len(B[0])

    if sarakkeet_a != rivit_b:
        raise ValueError("Incompatible matrix dimensions")

    C: Matrix = []
    for i in range(rivit_a):
        C.append([])
        for j in range(sarakkeet_b):
            C[i].append(0)
    for i in range(rivit_a):
        for j in range(sarakkeet_b):
            for k in range(sarakkeet_a):
                C[i][j] = C[i][j]+A[i][k]*B[k][j]

    return C


def matrix_vector_multiplication(A: Matrix, v: Vector) -> Vector:
    """Laskee matriisin ja vektorin välisen kertolaskun A*v

    Args:
        A (Matrix): Matriisi
        v (Vector): vektori

    Raises:
        ValueError: Jos matriisi ei ole kelvollinen
        ValueError: Jos vektorin pituus ei ole sama kuin matriisin sarakkeiden määrä

    Returns:
        Vector: tulosvektori
    """

    if not valid_matrix(A):
        raise ValueError("Invalid matrix")

    rivit = len(A)
    sarakkeet = len(A[0])

    if len(v) != sarakkeet:
        raise ValueError("The vector is of wrong size")

    y: Vector = []
    for i in range(rivit):
        y.append(0)
        for j in range(sarakkeet):
            y[i] = y[i]+A[i][j] * v[j]

    return y


def identitymatrix(size: int) -> Matrix:
    """Luo annetun kokoisen identiteettimatriisiin

    Args:
        size (int): kertoo matriisin rivien ja sarakkeiden määrän

    Returns:
        Matrix: size x size kokoinen identiteetti matriisi
    """

    I = [[0 for c in range(size)] for r in range(size)]
    for i in range(size):
        I[i][i] = 1
    return I
