from .vector import Vector

Matrix = list[list[float]]


def ValidMatrix(A: Matrix) -> bool:
	if len(A) == 0:
		return False
	if len(A[0]) == 0:
		return False

	cols = len(A[0])
	for row in A:
		if len(row) != cols:
			return False

	return True


def Transpose(A: Matrix) -> Matrix:
	if not ValidMatrix(A):
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

def MatrixMultiplication(A: Matrix, B: Matrix) -> Matrix:
	if not ValidMatrix(A) or not ValidMatrix(B):
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
				C[i][j]=C[i][j]+A[i][k]*B[k][j]

	return C


def MatrixVectorMultiplication(A: Matrix, v: Vector) -> Vector:
	if not ValidMatrix(A):
		raise ValueError("Invalid matrix")

	rivit = len(A)
	sarakkeet = len(A[0])

	if len(v) != sarakkeet:
		raise ValueError("The vector is of wrong size")

	y: Vector = []
	for i in range(rivit):
		y.append(0)
		for j in range(sarakkeet):
			y[i]=y[i]+A[i][j] * v[j]

	return y