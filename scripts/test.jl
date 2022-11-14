# entry file for setting debug points, and clicking run and debug in vscode
using IncrementalGRF
using LinearAlgebra: tril!

A = rand(20,20)
b = rand(20,4)
tril!(A)
L = BlockPackedLowerTri(A, 3)
x = L\b
L * x ≈ b # consistency
x ≈ A\b
