using Test
using Plots: plot
using LinearAlgebra: triu!, tril!

using incrementalGRF 

@testset "Testing \\(::PackedLowerTriangular{T}, ::AbstractVector{T}) where T<:Union{Float32,Float64}" begin
	A = PackedLowerTriangular([1., 2, 3])
	b = [1., 2.]
	@test A\b ≈ [1.,0]
	@test A\[2.,2.] ≈ [2.0, -0.66666666666666]

	for type in [Float32, Float64]
		for n in [50, 100, 1000]
			M = randn(type,(n,n));
			M = M' * M; # generate positive definite matrix (symmetric)
			A = PackedLowerTriangular(M[triu!(trues(size(M)))]);
			# extract lower triangular matrix as vector in row major format using symmetry of M
			tril!(M) # do the same with M (non-packed format)

			b = rand(type, n); # generate RHS

			x = A\b
			@test M * x ≈ b # consistency
			@test M\b ≈ x # same as non-packed
		end
	end
end

@testset "Testing GaussianRandomField" begin
	grf = GaussianRandomField{Float64}(Kernels.squaredExponential)
	x = -10:0.1:10
	y = vcat(grf.(x)...)
	plot(x,y)
end