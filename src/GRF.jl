using LinearAlgebra: LinearAlgebra, issuccess
using ElasticArrays: ElasticMatrix
using Random: AbstractRNG, default_rng

struct GaussianRandomField{T<:Number, N, Outdim}
	rng::AbstractRNG
	cov::CovarianceKernel{T, N}
	jitter::T
	randomness::ElasticMatrix{T}
	evaluated_points::ElasticMatrix{T}
	chol_cov::PackedLowerTriangular{T}
end

function GaussianRandomField{T, N}(
	rng::AbstractRNG,
	cov::CovarianceKernel{T,N};
	jitter=10*eps(T)
) where {T<:Number,N} 
	outdim = size(cov(zeros(T, N), zeros(T, N)), 1)
	GaussianRandomField{T, N, outdim}(
		rng, cov, jitter,
		#=randomness=# ElasticMatrix{T}(undef, T, (outdim, 0)), 
		#=evaluated_points=# ElasticMatrix{T}(undef, T, (outdim, 0)),
		#=chol_cov=# PackedLowerTriangular{T}([])
	)
end
function GaussianRandomField{T, N}(cov::Function{T,N}; jitter=10*eps(T)) where {T,N}
	GaussianRandomField{T,N}(default_rng(), cov, jitter=jitter)
end

function covariance(grf::GaussianRandomField{T, N}, x) where {T,N}
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
evaluate random field at point x
"""
function (grf::GaussianRandomField{T, N})(x::AbstractVector{T}) where {T<:Number, N}
	try
		coeff::Matrix{T} = grf.chol_cov \ covariance(grf, x)

		cond_expectation = reshape(reshape(grf.randomness, 1, :) * coeff, :)
		
		cond_var = grf.cov(x,x) .- transpose(coeff)*coeff
		cond_var[LinearAlgebra.diagind(cond_var)] .+= grf.jitter

		cond_var_cholesky = LinearAlgebra.cholesky(cond_var, check=false)
		if(!issuccess(cond_var_cholesky))
			@warn "existing points determine new point (up to numeric errors) perfectly. Maybe your evaluations are too close to each other. This can build up to sharp kinks when moving far enough away until there is real stochasticity again.
			Increase the distance of your evaluation points or increase jitter"
			return cond_expectation
		end
		append!(grf.evaluated_points, reshape(x, :, 1))
		extend!(
			grf.chol_cov,
			coeff,
			cond_var_cholesky.U
		)
		new_randomness =  randn(grf.rng, T, grf.outdim)
		append!(grf.randomness, reshape(new_randomness, :, 1))
		return cond_expectation + cond_var_cholesky.L * new_randomness
	catch e
		e isa UndefRefError || rethrow(e)
		# unevaluated random field
		grf.evaluated_points = reshape(x, :, 1)
		var = grf.cov(x,x)
		grf.outdim = size(var, 1)
		
		grf.randomness = ElasticMatrix(randn(grf.rng, T, (grf.outdim,1)))
		var_chol = LinearAlgebra.cholesky(var)
		grf.chol_cov = PackedLowerTriangular{T}([])
		extend!(
			grf.chol_cov, 
			Matrix{T}(undef, (0, grf.outdim)), 
			var_chol.U
		)
		return grf.chol_cov * reshape(grf.randomness, :)
	end

end

function (grf::GaussianRandomField{T, 1})(x::T) where T
	return grf([x])
end
