using Test
using Plots: plot
using LinearAlgebra: LinearAlgebra, triu!, tril!

using IncrementalGRF 

include("benchmark_suite.jl")

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

# @testset "Testing 1-dim GaussianRandomField" begin
# 	function oneDimGaussian()
# 		grf = GaussianRandomField(Kernels.SquaredExponential{Float64, 1}(1))
# 		grf.(-10:0.2:10)
# 	end

# 	grf = GaussianRandomField(Kernels.SquaredExponential{Float64, 1}(1))
# 	x = -10:0.1:10
# 	y = vcat(grf.(x)...)
# 	plot(x,y)
# end


@testset "Performance Benchmarks" begin
	try
		old_results = B.load("local_benchmark.json")
		results = runTunedSuite("params.json", "new_local_benchmark.json", verbose=true, seconds=100)
		judgements = B.judge(B.minimum(results), B.minimum(old_results))
		@test isempty(B.regressions(judgements))
	catch e
		@test error("No benchmark yet.") skip=true
	end
end