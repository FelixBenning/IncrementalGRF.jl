using Test
using Plots: plot
using LinearAlgebra: LinearAlgebra, triu!, tril!

using IncrementalGRF 

include("benchmark_suite.jl")

@testset "\\(::PackedLowerTriangular{T}, ::AbstractVector{T}) where T<:Union{Float32,Float64}" begin
	A = PackedLowerTriangular([1., 2, 3])
	b = [1., 2.]
	@test A\b ≈ [1.,0]
	@test A\[2.,2.] ≈ [2.0, -0.66666666666666]

	@testset "Random Test with type=$type" for type in [Float32, Float64]
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

@testset "Kernels" begin
	@testset "SquaredExponential" begin
		@test_throws ArgumentError Kernels.SquaredExponential{Float64, 1}(scale=-1.)
		@test_throws ArgumentError Kernels.SquaredExponential{Float64, 1}(scale=0.)

		@testset "TaylorCovariance (type=$type, dim=$dim)" for type in [Float32, Float64], dim in [1,3,50]
			l = randn(type)^2
			tk = Kernels.TaylorCovariance{1}(Kernels.SquaredExponential{type, dim}(scale=l))
			x = randn(type, dim)
			y = randn(type, dim)
			d = x-y
			@test tk(x,y) ≈ tk(d)
			@test tk(d) ≈ invoke(
				Kernels._taylor1, 
				Tuple{IsotropicKernel{type, dim}, AbstractVector{type}},
				tk.k, d
			)
			@test tk(d) ≈ invoke(
				Kernels._taylor1, 
				Tuple{StationaryKernel{type, dim}, AbstractVector{type}},
				tk.k, d
			)
			@test tk(d) ≈ invoke(
				Kernels._taylor1, 
				Tuple{CovarianceKernel{type, dim}, AbstractVector{type}, AbstractVector{type}},
				tk.k, x, y
			)
		end
		@testset "Scaling with dim =$dim" for dim in [1,3, 50]
			x = randn(Float64, dim)
			scale = randn(Float64)^2
			k = Kernels.SquaredExponential{Float64, dim}(scale=1.)
			scaled_k = Kernels.SquaredExponential{Float64, dim}(scale=scale)
			@test k(x) ≈ scaled_k(scale * x)

			tay = Kernels.TaylorCovariance{1}(k)
			scaled_tay = Kernels.TaylorCovariance{1}(scaled_k)
			C = tay(x)
			C_scaled = scaled_tay(scale * x)
			@test C[1,1] ≈ C_scaled[1,1]
			@test C[2:end, 1] ≈ scale * C_scaled[2:end, 1]
			@test C[1, 2:end] ≈ scale * C_scaled[1, 2:end]
			@test C[2:end, 2:end] ≈ scale^2 * C_scaled[2:end, 2:end]
		end
	end

	@testset "Matern" begin
		@testset "Scaling with ν=$nu, dim=$dim" for nu in [1., 1.1, 1.5, 2.,2.5], dim in [1,3, 50]
			x = randn(Float64, dim)
			scale = randn(Float64)^2
			k = Kernels.Matern{Float64, dim}(nu=nu, scale=1.)
			scaled_k = Kernels.Matern{Float64, dim}(nu=nu, scale=scale)
			@test k(x) ≈ scaled_k(scale * x)

			tay = Kernels.TaylorCovariance{1}(k)
			scaled_tay = Kernels.TaylorCovariance{1}(scaled_k)
			C = tay(x)
			C_scaled = scaled_tay(scale * x)
			@test C[1,1] ≈ C_scaled[1,1]
			@test C[2:end, 1] ≈ scale * C_scaled[2:end, 1]
			@test C[1, 2:end] ≈ scale * C_scaled[1, 2:end]
			@test C[2:end, 2:end] ≈ scale^2 * C_scaled[2:end, 2:end]
		end
	end
end
