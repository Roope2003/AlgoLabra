# pylint: skip-file
import pytest
from eigenface.qr import qr_decompose, eigendecompose
from eigenface.matrix import matrix_multiplication, transpose, matrix_vector_multiplication, identitymatrix
def test_qr_decompose_basic():
    A = [[1.0 ,2.0], [3.0, 4.0]]
    Q, R = qr_decompose(A)
    reconstruct = matrix_multiplication (Q,R)
    assert reconstruct[0] == pytest.approx(A[0])
    assert reconstruct[1] == pytest.approx(A[1])

def test_qr_is_Q_orthonormal():
    A = [[1.0 ,2.0], [3.0, 4.0]]
    Q, R = qr_decompose(A)
    Q_T = transpose(Q)
    Q_T_Q = matrix_multiplication(Q_T,Q)
    expected = identitymatrix(len(A[0]))
    assert Q_T_Q[0] == pytest.approx(expected[0])
    assert Q_T_Q[1] == pytest.approx(expected[1])

def test_qr_R_is_upper_triangular():
    A = [[1.0 ,2.0], [3.0, 4.0]]
    Q, R = qr_decompose(A)

    for i in range(len(R)):
        for j in range(len(R[0])):
            if i>j:
                assert R[i][j] == pytest.approx(0,0)


def test_eigendecompose():
    iterations = 20
    tolerance = 1e-5
    D = [[3.0, 0.0], [0.0, 4.0]]
    eigenvalues, eigenvectors=eigendecompose(D, iterations, tolerance)
    assert eigenvalues == pytest.approx([4.0, 3.0])
    assert len(eigenvectors) == 2
