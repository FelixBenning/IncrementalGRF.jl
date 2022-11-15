# entry file for setting debug points, and clicking run and debug in vscode
using IncrementalGRF
using LinearAlgebra: tril!
using CUDA
using BenchmarkTools: @btime

A = rand(20,20) 
#A = transpose(A) * A
b = rand(20,4)
tril!(A)
L = @btime BlockPackedLowerTri($A, 3, Vector{Float64}([0.0])) # substitute Vector for CuArray to run on GPU
x = L\b 
@assert (@btime L * x ≈ b) "Not exact solution to equation system L" # consistency
@assert x ≈ A\b "Not exact solution to equation system A"

