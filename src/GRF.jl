using LinearAlgebra: LinearAlgebra, issuccess
using ElasticArrays: ElasticMatrix
using Random: AbstractRNG, default_rng

struct GaussianRandomFunction{T<:Number, N}
	rng::AbstractRNG
	cov::CovarianceKernel{T, N}
	jitter::T
	outdim::Int
	randomness::ElasticMatrix{T}
	evaluated_points::ElasticMatrix{T}
	chol_cov::PackedLowerTriangular{T}
end

function GaussianRandomFunction(
	rng::AbstractRNG,
	cov::CovarianceKernel{T,N};
	jitter=10*eps(T)
) where {T<:Number,N}
	outdim = size(cov(zeros(T, N), zeros(T, N)), 1)
	GaussianRandomFunction{T,N}(
		rng, cov, jitter, outdim,
		#=randomness=# ElasticMatrix{T}(Matrix{T}(undef, (outdim, 0))), 
		#=evaluated_points=# ElasticMatrix{T}(Matrix{T}(undef, (N, 0))),
		#=chol_cov=# PackedLowerTriangular{T}(T[])
	)
end

function GaussianRandomFunction(cov::CovarianceKernel{T,N}; jitter=10*eps(T)) where {T,N}
	GaussianRandomFunction(default_rng(), cov, jitter=jitter)
end

function covariance(grf::GaussianRandomFunction{T, N}, x) where {T,N}
	mixed_cov = Matrix{T}(
		undef, size(grf.evaluated_points,2)*grf.outdim, grf.outdim
	)
	for (idx, pt) in zip(
		Iterators.countfrom(0, grf.outdim), 
		eachcol(grf.evaluated_points)
	)
		mixed_cov[idx+1:idx+grf.outdim, :] .= grf.cov(pt, x)
	end
	return mixed_cov
end

"""
evaluate random function at point x
"""
function (grf::GaussianRandomFunction{T, N})(x::AbstractVector{T}) where {T<:Number, N}
	coeff::Matrix{T} = solve!(grf.chol_cov, covariance(grf, x))

	cond_expectation, cond_var = conditionals(grf, x)
	
	cond_var[LinearAlgebra.diagind(cond_var)] .+= grf.jitter # numerical stability
	cond_var_cholesky = LinearAlgebra.cholesky(cond_var, check=false)

	if(!issuccess(cond_var_cholesky))
		@warn "existing points determine new point (up to numeric errors) perfectly. Maybe your evaluations are too close to each other. This can build up to sharp kinks when moving far enough away until there is real stochasticity again.
		Increase the distance of your evaluation points or increase jitter"
		return cond_expectation
	end
	# reshape AbstractVector of length n to nx1 matrix to append
	append!(grf.evaluated_points, reshape(x, :, 1))
	extend!(
		grf.chol_cov,
		coeff,
		cond_var_cholesky.U
	)
	new_randomness =  randn(grf.rng, T, grf.outdim)
	append!(grf.randomness, reshape(new_randomness, :, 1))
	return cond_expectation + cond_var_cholesky.L * new_randomness
end

@inline function conditionals(grf::GaussianRandomFunction{T,N}, x::AbstractVector) where {T,N}
	"""
	Calculates the conditional expectation and variance of the grf at point x
	given the values of grf at previously evaluated points. Does not count as an
	evaluation of grf yet. I.e.

	return (
		condExp= E[grf(x) | grf.( grf.evaluated_points)],
		condVar= Var[grf(x) | grf.( grf.evaluated_points)]
	) """
	coeff::Matrix{T} = solve!(grf.chol_cov, covariance(grf, x))
	return (
		condExp= reshape(reshape(grf.randomness, 1, :) * coeff, :),
		condVar= grf.cov(x,x) .- transpose(coeff)*coeff
	)
end

function conditionals(grf::GaussianRandomFunction{T,N}) where {T,N}
	return x::AbstractVector{T} -> conditionals(grf, x)
end

function conditionalExpectation(grf::GaussianRandomFunction{T,N}) where {T,N}
	return x::AbstractVector{T} -> conditionals(grf,x)[:condExp]
end

function (grf::GaussianRandomFunction{T, 1})(x::T) where T
	return grf([x])
end
