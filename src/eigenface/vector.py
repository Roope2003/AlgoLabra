Vector = list[float]


def dot_product(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors are of different sizes")
    summa = 0.0
    for index, value in enumerate(a):
        summa = summa + value * b[index]

    return summa


def norm(a: Vector) -> float:
    return dot_product(a, a) ** 0.5


def normalize(v: Vector) -> Vector:
    n = norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    normalized: Vector = [i / n for i in v]

    return normalized


def vector_substract(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors are of different sizes")
    result = []
    for index, value in enumerate(a):
        result.append(value - b[index])

    return result


def vector_division_scalar(v: Vector, s: int):
    if s == 0:
        raise ValueError("Cannot divide vector by zero")
    result = []
    for _, value in enumerate(v):
        result.append(value / s)
    return result
