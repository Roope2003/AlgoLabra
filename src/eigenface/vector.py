Vector = list[float]


def DotProduct(a: Vector, b: Vector) -> float:
	if len(a) != len(b):
		raise ValueError("Vectors are of different sizes")

	summa = 0.0
	for i in range(len(a)):
		summa = summa + a[i] * b[i]

	return summa

def Norm(a: Vector) ->float:
	return DotProduct(a, a) ** 0.5

def Normalize(v: Vector) -> Vector:
    n = Norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    normalized: Vector = []
    for i in range(len(v)):
        normalized.append(v[i] / n)
    return normalized


def VectorSubstract(a:Vector, b:Vector)-> Vector:
	if len(a) != len(b):
		raise ValueError("Vectors are of different sizes")
	result=[]
	for i in range(len(a)):
		result.append(a[i]-b[i])

	return result