# pylint: skip-file
import pytest
from eigenface.matrix import valid_matrix, transpose, matrix_multiplication, matrix_vector_multiplication, identitymatrix

def test_valid_matrix():
    assert valid_matrix([[1.0, 2.0], [3.0, 4.0]]) is True

def test_valid_matrix_empty():
    assert valid_matrix([]) is False
    assert valid_matrix([[]]) is False

def test_valid_matrix_invalid():
    assert valid_matrix([[1.0], [2.0, 2.0]]) is False

def test_transpose_basic():
    result = transpose([[1.0, 2.0], [3.0, 4.0]])
    assert result[0] == pytest.approx([1.0, 3.0])
    assert result[1] == pytest.approx([2.0, 4.0])

def test_transpose_invalid_raises():
    with pytest.raises(ValueError):
        transpose([])

def test_matrix_multiplication_basic():
    X = [[1.0, 2.0], [3.0, 4.0]]
    Y = [[4.0, 2.0], [1.0, 3.0]]
    result = matrix_multiplication(X,Y)
    assert result[0] == pytest.approx([6.0, 8.0])
    assert result[1] == pytest.approx([16.0, 18.0])

def test_matrix_multiplication_incombatible_raises():
    with pytest.raises(ValueError):
        matrix_multiplication([[1.0, 2.0]], [[1.0, 2.0]])

def test_matrix_mutliplication_invalid_matrix_raises():
    with pytest.raises(ValueError):
        matrix_multiplication([],[[2.0]])

def test_matrix_vector_multiplication_basic():
    X = [[1.0, 2.0], [3.0, 4.0]]
    v= [2.0, 1.0]
    result = matrix_vector_multiplication(X,v)
    assert result == pytest.approx([4.0, 10.0])

def test_matrix_vector_multiplication_wrong_size_raises():
    with pytest.raises(ValueError):
        X = [[1.0, 2.0, 6.0], [3.0, 4.0, 6.0], [8.0, 9.0, 6.0]]
        v= [2.0, 1.0]
        matrix_vector_multiplication(X,v)

def test_matrix_vector_multiplication_invalid_matrix_raises():
    with pytest.raises(ValueError):
        matrix_vector_multiplication([[1.0, 2.0], [1.0]], [1.0, 2.0])

def test_identity_matrix():
    result = identitymatrix(3)
    assert result[0] == [1, 0, 0]
    assert result[1] == [0, 1, 0]
    assert result[2] == [0, 0, 1]