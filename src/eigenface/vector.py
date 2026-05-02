"""Vektorilaskennan apufunktiot Eigenface-algoritmille.

Moduuli sisältää perusoperaatiot eli pistetulon, normin,
normalisoinnin, vektorivähennyksen ja skalaarilla jaon.
"""





Vector = list[float]


def dot_product(a: Vector, b: Vector) -> float:
    """Laskee kahden vektorin pistetulon.

    Args:
        a (Vector): Ensimmäinen vektori
        b (Vector): Toinen vektori

    Returns:
        Vektoreiden pistetulo

    Raises:
        ValueError: Jos vektoreiden pituudet eivät täsmää
    """

    if len(a) != len(b):
        raise ValueError("Vectors are of different sizes")
    summa = 0.0
    for index, value in enumerate(a):
        summa = summa + value * b[index]

    return summa


def norm(a: Vector) -> float:
    """Laskee vektorin normin eli pituuden

    Args:
        a: Vektori

    Returns:
        float: Vektorin pituus/normi
    """
    return dot_product(a, a) ** 0.5


def normalize(v: Vector) -> Vector:
    """Palauttaa vektorin jonka pituus on 1

    Args:
        v (Vector): vektori

    Raises:
        ValueError: Jos vektorin pituus on 0

    Returns:
        Vector: Normalisoitu vektori
    """


    n = norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    normalized: Vector = [i / n for i in v]

    return normalized


def vector_substract(a: Vector, b: Vector) -> Vector:
    """Vähentää kaksi vektoria toisistaan alkioittain

    Args:
        a (Vector): Vektori josta vähennetään
        b (Vector): Vähennettävä vektori

    Raises:
        ValueError: Jos vektorien pituudet eroavat

    Returns:
        Vector: Vektori a, jonka jokaisesta alkiosta on vähennetty vektorin b vastaava alkio
    """
    if len(a) != len(b):
        raise ValueError("Vectors are of different sizes")
    result = []
    for index, value in enumerate(a):
        result.append(value - b[index])

    return result


def vector_division_scalar(v: Vector, s: int) -> Vector:
    """Jakaa vektorin jokaisen alkion skalaarilla

    Args:
        v (Vector): vektori jonka alkiot jaetaan
        s (int): Skalaari jolla jaetaan

    Raises:
        ValueError: Jos jakava skalaari on 0

    Returns:
        Vector: vektori v jonka jokainen alkio on jaettu skalaarilla s
    """

    if s == 0:
        raise ValueError("Cannot divide vector by zero")
    result = []
    for _, value in enumerate(v):
        result.append(value / s)
    return result
