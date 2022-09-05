using LinearAlgebra: LinearAlgebra, issuccess
using ElasticArrays: ElasticMatrix
using Random: AbstractRNG, default_rng

mutable struct GaussianRandomField{T<:Number}
	rng::AbstractRNG
	cov::Function
	jitter::T
	outdim::Int
	randomness::ElasticMatrix{T}
	evaluated_points::ElasticMatrix{T}
	chol_cov::PackedLowerTriangular{T}


	GaussianRandomField{T}(rng::AbstractRNG, cov::Function; jitter=10*eps(T)) where T<:Number = new(
		rng, cov, jitter
	)
end

GaussianRandomField{T}(cov::Function; jitter=10*eps(T)) where T = GaussianRandomField{T}(default_rng(), cov, jitter=jitter)

function covariance(grf::GaussianRandomField{T}, x) where T
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
function (grf::GaussianRandomField{T})(x::AbstractVector{T}) where {T<:Number}
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

function (grf::GaussianRandomField{T})(x::T) where T
	# ease of use for one dimensional random fields
	return grf([x])
end
