module Kernels

using LinearAlgebra: LinearAlgebra
using Zygote: Zygote, ForwardDiff
using SpecialFunctions: besselk, gamma
import ..StationaryKernel, ..IsotropicKernel, ..CovarianceKernel

@inline function (k::StationaryKernel{T,N})(
    x::Union{AbstractVector{T},Vector{ForwardDiff.Dual{Nothing,T,n}}},
    y::Union{AbstractVector{T},Vector{ForwardDiff.Dual{Nothing,T,n}}}
) where {T,N,n}
    @boundscheck length(x) == N || throw(ArgumentError(
        "The first input is of size $(length(x)), but should be of size $N"))
    @boundscheck length(y) == N || throw(ArgumentError(
        "The second input is of size $(length(y)), but should be of size $N"))
    return @inbounds k(x - y)
end

@inline function (k::IsotropicKernel{T,N})(
    d::Union{AbstractVector{T},Vector{ForwardDiff.Dual{Nothing,T,n}}}
) where {T,N,n}
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

@inline function (cov::TaylorCovariance{1,T,N,K})(x::AbstractVector{T}, y::AbstractVector{T}) where {T,N,K<:CovarianceKernel{T,N}}
    @boundscheck length(x) == N || throw(ArgumentError(
        "The first input is of size $(length(x)), but should be of size $N"))
    @boundscheck length(y) == N || throw(ArgumentError(
        "The second input is of size $(length(y)), but should be of size $N"))
    return _taylor1(cov.k, x, y)
end

@inline function (cov::TaylorCovariance{1,T,N,K})(x::AbstractVector{T}, y::AbstractVector{T}) where {T,N,K<:StationaryKernel{T,N}}
    @boundscheck length(x) == N || throw(ArgumentError(
        "The first input is of size $(length(x)), but should be of size $N"))
    @boundscheck length(y) == N || throw(ArgumentError(
        "The second input is of size $(length(y)), but should be of size $N"))

    return _taylor1(cov.k, x - y)
end
@inline function (cov::TaylorCovariance{1,T,N,K})(d::AbstractVector{T}) where {T,N,K<:StationaryKernel{T,N}}
    @boundscheck length(d) == N || throw(ArgumentError(
        "The input is of size $(length(d)), but should be of size $N"))

    return _taylor1(cov.k, d)
end

"""
	Auto-diff Fallback for the TaylorCovariance{1,T,N,CovarianceKernel{T,N}}
"""
@inline function _taylor1(kernel::CovarianceKernel{T,N}, x::AbstractVector{T}, y::AbstractVector{T}) where {T,N}
    result = Matrix{T}(undef, N + 1, N + 1)
    val, grad = @inbounds Zygote.withgradient(kernel, x, y)
    result[1, 1] = val
    result[2:end, 1] = grad[1]
    result[1, 2:end] = grad[2]
    
    # not efficient at all:
    full_hess = @inbounds Zygote.hessian(v->kernel(v[1:N],v[N+1:end]), vcat(x,y))
    result[2:end, 2:end] = full_hess[N+1:end, 1:N] # throw away 3/4 of the matrix
    return result
end

"""
	Auto-diff Fallback for the TaylorCovariance{1,T,N,StationaryKernel{T,N}}
"""
@inline function _taylor1(kernel::StationaryKernel{T,N}, d::AbstractVector{T}) where {T,N}
    result = Matrix{T}(undef, N + 1, N + 1)
    val, grad = @inbounds Zygote.withgradient(kernel, d)
    grad = first(grad) # only one input
    result[1, 1] = val
    result[2:end, 1] = grad
    result[1, 2:end] = -grad
    result[2:end, 2:end] = @inbounds -Zygote.hessian(kernel, d)
    return result
end


"""
	Auto-diff Fallback for TaylorCovariance{1,T,N,IsotropicKernel{T,N}}
"""
@inline function _taylor1(kernel::IsotropicKernel{T,N}, d::AbstractVector{T}) where {T,N}
    result = Matrix{T}(undef, N + 1, N + 1)
    val, grad_1dim = Zygote.withgradient(x -> sqNormEval(kernel, x), LinearAlgebra.dot(d, d))

    result[1, 1] = val

    grad_1dim = first(grad_1dim) # only one input
    grad = 2 * grad_1dim * d
    result[2:end, 1] = grad
    result[1, 2:end] = -grad

    hess_1dim = Zygote.hessian(x -> sqNormEval(kernel, x), LinearAlgebra.dot(d, d))
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

struct Matern{T<:Number, Dim} <: IsotropicKernel{T,Dim}
    nu::Real
    lengthScale::T
    Matern{T, Dim}(nu::Real, lengthScale::T) where {T<:Number, Dim} = begin
        lengthScale > 0 || throw(ArgumentError("lengthScale is not positive"))
        nu > 0 || throw(ArgumentError("nu is not positive"))
        new(nu, lengthScale)
    end
end

@inline function sqNormEval(
    k::Matern{T, N},
    d::Union{T,ForwardDiff.Dual}
) where {T,N}
    if d > 0
        x = sqrt(2*k.nu*d)/k.lengthScale
        return 2^(1-k.nu)/gamma(k.nu) * x^k.nu * besselk(k.nu,x)
    else
        return 1 # variance
    end
    throw(ArgumentError("the squared Norm should never be negative!"))
end

@inline function sqNormEval(
    k::SquaredExponential{T,N},
    d::Union{T,ForwardDiff.Dual}
) where {T,N}
    return exp(-d / (2 * k.lengthScale^2))
end

@inline function _taylor1(kernel::SquaredExponential{T,N}, d::AbstractVector{T}) where {T,N}
    dim = length(d)
    result = Matrix{T}(undef, dim + 1, dim + 1)
    factor = sqNormEval(kernel, LinearAlgebra.dot(d, d))
    result[1, 1] = factor
    dl = d / (kernel.lengthScale^2)
    grad = factor * dl
    result[2:end, 1] = -grad
    result[1, 2:end] = grad
    result[2:end, 2:end] = (
        factor * LinearAlgebra.I / kernel.lengthScale^2 - grad * transpose(dl)
    )
    return result
end


# """ Broken: Need special treatment of case d=0 """
@inline function _taylor1(kernel::Matern{T,N}, d::AbstractVector{T}) where {T,N}
    if 0 == d
        
    end
    result = Matrix{T}(undef, n + 1, n + 1)
    eta = LinearAlgebra.norm(d)
    x = sqrt(2*kernel.nu)*eta/kernel.lengthScale
    factor = 2^(1-kernel.nu)/gamma(kernel.nu) * x^kernel.nu
    grad_1dim = - factor * besselk(kernel.nu-1, x) / eta
    result[1, 1] = factor * besselk(kernel.nu, x)
    grad = grad_1dim * d
    result[2:end, 1] = grad
    result[1, 2:end] = -grad
    hess_1dim = factor/eta^2 * (
        2*kernel.nu/kernel.lengthScale^2 * besselk(kernel.nu-2, x)
        + sqrt(2*kernel.nu)/kernel.lengthScale/eta
    )
    hess = hess_1dim * d * d' + grad_1dim * LinearAlgebra.I
    result[2:end, 2:end] = -hess
    return result
end


end # module