module Kernels

using LinearAlgebra: LinearAlgebra
using Zygote: Zygote
import ..StationaryKernel, ..IsotropicKernel, ..CovarianceKernel

@inline function (k::StationaryKernel{T,N})(
	x::Union{AbstractVector{T}, Vector{Zygote.ForwardDiff.Dual{Nothing, T, N}}},
	y::Union{AbstractVector{T}, Vector{Zygote.ForwardDiff.Dual{Nothing, T, N}}}
) where {T,N}
	@boundscheck length(x) == N || throw(ArgumentError(
		"The first input is of size $(length(x)), but should be of size $N"))
	@boundscheck length(y) == N || throw(ArgumentError(
		"The second input is of size $(length(y)), but should be of size $N"))
	return @inbounds k(x-y) 
end

@inline function (k::IsotropicKernel{T,N})(
	d::Union{AbstractVector{T},Vector{Zygote.ForwardDiff.Dual{Nothing, T, N}}}
) where {T,N}
	@boundscheck length(d) == N || throw(ArgumentError(
		"The input is of size $(length(d)), but should be of size $N"))
	return sqNormEval(k, LinearAlgebra.dot(d,d))
end

struct TaylorCovariance{Order, T, N} <: CovarianceKernel{T,N}
	k::CovarianceKernel{T,N}
end

function (cov::TaylorCovariance{1,T,N})(x::AbstractVector{T}, y::AbstractVector{T}) where {T,N}
	return taylor1Covariance(cov.k, x, y)
end

@inline function taylor1Covariance(k::StationaryKernel{T,N}, x::AbstractVector{T}, y::AbstractVector{T}) where {T,N}
	@boundscheck length(x) == N || throw(ArgumentError(
		"The first input is of size $(length(x)), but should be of size $N"))
	@boundscheck length(y) == N || throw(ArgumentError(
		"The second input is of size $(length(y)), but should be of size $N"))
	return @inbounds taylor1Covariance(k, x-y)
end

"""
	Auto-diff Fallback of the taylor1Covariance for StationaryKernel
"""
@inline function taylor1Covariance(k::StationaryKernel{T,N}, d::AbstractVector{T}) where {T,N}
	@boundscheck length(d) == N || throw(ArgumentError(
		"The input is of size $(length(d)), but should be of size $N"))

	result = Matrix{T}(undef, N+1, N+1)
	val, grad = @inbounds Zygote.withgradient(k, d)
	grad = first(grad) # only one input
	result[1,1] = val 
	result[2:end, 1] = - grad
	result[1, 2:end] = grad
	result[2:end, 2:end] = @inbounds -Zygote.hessian(k, d)
	return result
end


"""
	Auto-diff Fallback of the taylor1Covariance for IsotropicKernel
"""
@inline function taylor1Covariance(k::IsotropicKernel{T,N}, d::AbstractVector{T}) where {T,N}
	@boundscheck length(d) == N || throw(ArgumentError(
		"The input is of size $(length(d)), but should be of size $N"))

	result = Matrix{T}(undef, N+1, N+1)
	val, grad_1dim = Zygote.withgradient(x->sqNormEval(k, x), LinearAlgebra.dot(d,d))

	result[1,1] = val 

	grad_1dim = first(grad_1dim) # only one input
	grad = 2*grad_1dim*d
	result[2:end, 1] = - grad
	result[1, 2:end] = grad

	hess_1dim = Zygote.hessian(x->sqNormEval(k,x), LinearAlgebra.dot(d,d))
	hess = 4*hess_1dim * d*d' + 2*grad_1dim* I
	result[2:end, 2:end] = - hess 
	return result
end

struct SquaredExponential{T<:Number, N} <: IsotropicKernel{T,N}
	lengthScale::T
end

@inline function sqNormEval(
	k::SquaredExponential{T,N},
	d::Union{T, Zygote.ForwardDiff.Dual{Nothing, T, N}}
) where {T,N}
	return exp(-d/(2*k.lengthScale^2))
end


function squaredExponential(x,y)
	return exp(-LinearAlgebra.norm(x-y)^2/2)
end


function gaussKernel(x)
	exp(-x^2/2)
end

function sqExpKernelWithGrad(x::AbstractVector{T},y::AbstractVector{T}) where T<: Number
	d = x-y
	dim = length(d)
	result = Matrix{T}(undef, dim+1, dim+1)
	result[1,1] = 1
	result[2:end, 1] = - d
	result[1, 2:end] = d
	result[2:end, 2:end] = (LinearAlgebra.I - d * transpose(d))
	return result * gaussKernel(LinearAlgebra.norm(d))
end

end # module