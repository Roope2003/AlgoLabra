# pylint: skip-file
import pytest
from math import sqrt
from eigenface.pca import compute_mean, center_data, covariance,train_eigenfaces, predict_face, nearest_neighbor, euclidean_distance


def test_compute_mean_basic():
    A= [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    result= compute_mean(A)
    assert result == pytest.approx([3.0, 4.0])

def test_center_data_basic():
    A= [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    mean = compute_mean(A)
    A_centered = center_data(A,mean)
    assert A_centered[0] == pytest.approx([-2.0,-2.0])
    assert A_centered[1] == pytest.approx([0.0, 0.0])
    assert A_centered[2] == pytest.approx([2.0, 2.0])

def test_covariance_basic():
    A= [[1.0, 2.0], [3.0, 4.0]]
    expected=[[2.5, 5.5], [5.5, 12.5]]
    result = covariance(A)
    assert result[0] == pytest.approx(expected[0])
    assert result[1] == pytest.approx(expected[1])

def test_train_eigenfaces_smoke():
    data=[[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]
    mean, eigenvalues, eigenfaces, train_weights =train_eigenfaces(data, k=2, iterations=30, tolerance=1e-5)

    assert len(mean) == 2
    assert len(train_weights) == len(data)
    assert len(eigenfaces) == len(eigenvalues)
    assert len(eigenfaces) <=2
    assert len(eigenfaces[0]) == len(data[0])
    assert len(train_weights[0]) == len(eigenfaces)

def test_train_eigenfaces_empty_data_raises():
    with pytest.raises(Exception):
        train_eigenfaces([], k=2, iterations=10, tolerance=1e-5)


def test_predict_face_with_nonzero_mean():
    image = [5.0, 7.0]
    mean = [2.0, 3.0]
    top_eigenfaces = [[1.0, 0.0], [0.0, 1.0]]

    weights = predict_face(image, mean, top_eigenfaces)
    assert weights == pytest.approx([3.0, 4.0])


def test_predict_face_basic():
    image = [3.0, 4.0]
    mean = [0.0, 0.0]
    top_eigenfaces = [[1.0, 0.0], [0.0, 1.0]]

    weights = predict_face(image, mean, top_eigenfaces)
    assert weights == pytest.approx([3.0, 4.0])

def test_euclidean_distance_basic():
    a=[1.0, 2.0, 3.0]
    b=[-1.0, 4.0, -2.0]
    result= euclidean_distance(a,b)
    assert result == pytest.approx(sqrt(33))

def test_euclidean_distance_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        euclidean_distance([1.0], [1.0, 2.0])


def test_nearest_neighbor_returns_best_index_and_distance():
    image_weight = [3.0, 4.0]
    train_weights = [[10.0, 10.0], [2.0, 5.0], [0.0, 0.0]]

    idx, dist = nearest_neighbor(image_weight, train_weights)

    assert idx == 1
    assert dist == pytest.approx(euclidean_distance(image_weight, train_weights[1]))


def test_nearest_neighbor_empty_train_weights():
    idx, dist = nearest_neighbor([1.0, 2.0], [])
    assert idx == -1
    assert dist == float("inf")

