import pytest
from eigenface.vector import dot_product, norm, normalize, vector_substract, vector_division_scalar

#pylint: skip-file

def test_dot_product_basic():
    result = dot_product([1.0, 2.0], [3.0, 4.0])
    assert result == pytest.approx(11.0)

def test_dot_product_mismatched_sizes_raises():
    with pytest.raises(ValueError):
        dot_product([1.0], [1.0, 2.0])

def test_norm_basic():
    result= norm([3.0,4.0])
    assert result == pytest.approx(5.0)

def test_normalize_basic():
    result = normalize([3.0, 4.0])
    assert result == pytest.approx([0.6, 0.8])
    assert norm(result) == pytest.approx(1.0)

def test_normalize_zero_vector():
    with pytest.raises(ValueError):
        normalize([0.0,0.0])

def test_vector_substract_basic():
    result = vector_substract([5.0, 2.0], [3.0, 1.0])
    assert result == pytest.approx([2.0, 1.0])

def test_vector_substract_size_mismatch():
    with pytest.raises(ValueError):
        vector_substract([1.0], [1.0, 2.0])

def test_vector_division_scalar_basic():
    result = vector_division_scalar([6.0, 3.0], 3)
    assert result == pytest.approx([2.0, 1.0])