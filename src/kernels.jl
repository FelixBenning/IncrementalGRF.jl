module Kernels

using LinearAlgebra: LinearAlgebra
using Zygote: Zygote
import ..StationaryKernel, ..IsotropicKernel, ..CovarianceKernel

@inline function (k::StationaryKernel{T,N})(
    x::Union{AbstractVector{T},Vector{Zygote.ForwardDiff.Dual{Nothing,T,N}}},
    y::Union{AbstractVector{T},Vector{Zygote.ForwardDiff.Dual{Nothing,T,N}}}
) where {T,N}
    @boundscheck length(x) == N || throw(ArgumentError(
        "The first input is of size $(length(x)), but should be of size $N"))
    @boundscheck length(y) == N || throw(ArgumentError(
        "The second input is of size $(length(y)), but should be of size $N"))
    return @inbounds k(x - y)
end

@inline function (k::IsotropicKernel{T,N})(
    d::Union{AbstractVector{T},Vector{Zygote.ForwardDiff.Dual{Nothing,T,N}}}
) where {T,N}
    @boundscheck length(d) == N || throw(ArgumentError(
        "The input is of size $(length(d)), but should be of size $N"))
    return sqNormEval(k, LinearAlgebra.dot(d, d))
end

struct TaylorCovariance{Order,T,N,K<:CovarianceKernel{T,N}} <: CovarianceKernel{T,N}
    k::K
end

function TaylorCovariance{Order}(k::K) where {Order,T,N,K<:CovarianceKernel{T,N}}
    return TaylorCovariance{Order,T,N,K}(k)
end

@inline function (cov::TaylorCovariance{1,T,N,K})(x::AbstractVector{T}, y::AbstractVector{T}) where {T,N,K<:StationaryKernel{T,N}}
    return cov(x - y)
end

"""
	Auto-diff Fallback for the TaylorCovariance{1,T,N,StationaryKernel{T,N}}
"""
@inline function (cov::TaylorCovariance{1,T,N,K})(d::AbstractVector{T}) where {T,N,K<:StationaryKernel{T,N}}
    @boundscheck length(d) == N || throw(ArgumentError(
        "The input is of size $(length(d)), but should be of size $N"))

    result = Matrix{T}(undef, N + 1, N + 1)
    val, grad = @inbounds Zygote.withgradient(cov.k, d)
    grad = first(grad) # only one input
    result[1, 1] = val
    result[2:end, 1] = grad
    result[1, 2:end] = -grad
    result[2:end, 2:end] = @inbounds -Zygote.hessian(cov.k, d)
    return result
end


"""
	Auto-diff Fallback for TaylorCovariance{1,T,N,IsotropicKernel{T,N}}
"""
@inline function (cov::TaylorCovariance{1,T,N,K})(d::AbstractVector{T}) where {T,N,K<:IsotropicKernel{T,N}}
    @boundscheck length(d) == N || throw(ArgumentError(
        "The input is of size $(length(d)), but should be of size $N"))

    result = Matrix{T}(undef, N + 1, N + 1)
    val, grad_1dim = Zygote.withgradient(x -> sqNormEval(cov.k, x), LinearAlgebra.dot(d, d))

    result[1, 1] = val

    grad_1dim = first(grad_1dim) # only one input
    grad = 2 * grad_1dim * d
    result[2:end, 1] = grad
    result[1, 2:end] = -grad

    hess_1dim = Zygote.hessian(x -> sqNormEval(cov.k, x), LinearAlgebra.dot(d, d))
    hess = 4 * hess_1dim * d * d' + 2 * grad_1dim * LinearAlgebra.I
    result[2:end, 2:end] = -hess
    return result
end

struct SquaredExponential{T<:Number,N} <: IsotropicKernel{T,N}
    lengthScale::T
	SquaredExponential{T,N}(lengthScale) where {T<:Number,N} = begin 
		lengthScale > 0 ? new(lengthScale) : throw(
			ArgumentError("lengthScale is not positive")
		)
	end
end

@inline function sqNormEval(
    k::SquaredExponential{T,N},
    d::Union{T,Zygote.ForwardDiff.Dual}
) where {T,N}
    return exp(-d / (2 * k.lengthScale^2))
end


# @inline function (cov::TaylorCovariance{1,T,N,K})(d::AbstractVector{T}) where {T,N,K<:SquaredExponential{T,N}}
#     dim = length(d)
#     result = Matrix{T}(undef, dim + 1, dim + 1)
# 	factor = sqNormEval(cov.k, LinearAlgebra.dot(d,d))
#     result[1, 1] = factor
# 	dl = d/(cov.k.lengthScale^2) * factor
#     result[2:end, 1] = - dl
#     result[1, 2:end] = dl
#     result[2:end, 2:end] = factor * (
# 		LinearAlgebra.I/cov.k.lengthScale^2 - dl * transpose(dl)
# 	)
#     return result 
# end

end # module