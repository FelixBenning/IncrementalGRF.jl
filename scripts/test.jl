# entry file for setting debug points, and clicking run and debug in vscode
using IncrementalGRF

A = rand(5,5)

L = BlockPackedLowerTri(A, 3)

b = [
	0.  1.
	2.  3.
	4.  5.
]	
L\b
