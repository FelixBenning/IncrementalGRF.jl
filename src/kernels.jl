module Kernels

using LinearAlgebra: LinearAlgebra
using Zygote: Zygote, ForwardDiff
using SpecialFunctions: besselk, gamma
import ..StationaryKernel, ..IsotropicKernel, ..CovarianceKernel

@inline function (k::StationaryKernel{T,Dim})(
    x::Union{AbstractVector{T},Vector{ForwardDiff.Dual{Nothing,T,n}}},
    y::Union{AbstractVector{T},Vector{ForwardDiff.Dual{Nothing,T,n}}}
) where {T,Dim,n}
    @boundscheck length(x) == Dim || throw(ArgumentError(
        "The first input is of size $(length(x)), but should be of size $Dim"))
    @boundscheck length(y) == Dim || throw(ArgumentError(
        "The second input is of size $(length(y)), but should be of size $Dim"))
    return @inbounds k(x - y)
end

@inline function (k::IsotropicKernel{T,Dim})(
    d::Union{AbstractVector{T},Vector{ForwardDiff.Dual{Nothing,T,n}}}
) where {T,Dim,n}
    @boundscheck length(d) == Dim || throw(ArgumentError(
        "The input is of size $(length(d)), but should be of size $Dim"))
    return k(LinearAlgebra.norm(d))
end

struct TaylorCovariance{Order,T,Dim,K<:CovarianceKernel{T,Dim}} <: CovarianceKernel{T,Dim}
    k::K
end

function TaylorCovariance{Order}(k::K) where {Order,T,Dim,K<:CovarianceKernel{T,Dim}}
    return TaylorCovariance{Order,T,Dim,K}(k)
end

@inline function (cov::TaylorCovariance{1,T,Dim,K})(
    x::AbstractVector{T}, y::AbstractVector{T}
) where {T,Dim,K<:CovarianceKernel{T,Dim}}
    @boundscheck length(x) == Dim || throw(ArgumentError(
        "The first input is of size $(length(x)), but should be of size $Dim"))
    @boundscheck length(y) == Dim || throw(ArgumentError(
        "The second input is of size $(length(y)), but should be of size $Dim"))
    return _taylor1(cov.k, x, y)
end

@inline function (cov::TaylorCovariance{1,T,Dim,K})(
    x::AbstractVector{T}, y::AbstractVector{T}
) where {T,Dim,K<:StationaryKernel{T,Dim}}
    @boundscheck length(x) == Dim || throw(ArgumentError(
        "The first input is of size $(length(x)), but should be of size $Dim"))
    @boundscheck length(y) == Dim || throw(ArgumentError(
        "The second input is of size $(length(y)), but should be of size $Dim"))

    return _taylor1(cov.k, x - y)
end
@inline function (cov::TaylorCovariance{1,T,Dim,K})(d::AbstractVector{T}) where {T,Dim,K<:StationaryKernel{T,Dim}}
    @boundscheck length(d) == Dim || throw(ArgumentError(
        "The input is of size $(length(d)), but should be of size $Dim"))

    return _taylor1(cov.k, d)
end

"""
	Auto-diff Fallback for the TaylorCovariance{1,T,Dim,CovarianceKernel{T,Dim}}
"""
@inline function _taylor1(kernel::CovarianceKernel{T,Dim}, x::AbstractVector{T}, y::AbstractVector{T}) where {T,Dim}
    result = Matrix{T}(undef, Dim + 1, Dim + 1)
    val, grad = @inbounds Zygote.withgradient(kernel, x, y)
    result[1, 1] = val
    result[2:end, 1] = grad[1]
    result[1, 2:end] = grad[2]
    
    # not efficient at all:
    full_hess = @inbounds Zygote.hessian(v->kernel(v[1:Dim],v[Dim+1:end]), vcat(x,y))
    result[2:end, 2:end] = full_hess[Dim+1:end, 1:Dim] # throw away 3/4 of the matrix
    return result
end

"""
	Auto-diff Fallback for the TaylorCovariance{1,T,Dim,StationaryKernel{T,Dim}}
"""
@inline function _taylor1(kernel::StationaryKernel{T,Dim}, d::AbstractVector{T}) where {T,Dim}
    result = Matrix{T}(undef, Dim + 1, Dim + 1)
    val, grad = @inbounds Zygote.withgradient(kernel, d)
    grad = first(grad) # only one input
    result[1, 1] = val
    result[2:end, 1] = grad
    result[1, 2:end] = -grad
    result[2:end, 2:end] = @inbounds -Zygote.hessian(kernel, d)
    return result
end


"""
	Auto-diff Fallback for TaylorCovariance{1,T,Dim,IsotropicKernel{T,Dim}}
"""
@inline function _taylor1(kernel::IsotropicKernel{T,Dim}, d::AbstractVector{T}) where {T,Dim}
    result = Matrix{T}(undef, Dim + 1, Dim + 1)
    val, grad_1dim = Zygote.withgradient(x -> kernel(x), LinearAlgebra.norm(d))

    result[1, 1] = val

    grad_1dim = first(grad_1dim) # only one input
    grad = 2 * grad_1dim * d
    result[2:end, 1] = grad
    result[1, 2:end] = -grad

    hess_1dim = Zygote.hessian(x -> kernel(x), LinearAlgebra.norm(d))
    hess = 4 * hess_1dim * d * d' + 2 * grad_1dim * LinearAlgebra.I
    result[2:end, 2:end] = -hess
    return result
end

struct SquaredExponential{T<:Number,Dim} <: IsotropicKernel{T,Dim}
    lengthScale::T
    SquaredExponential{T,Dim}(lengthScale) where {T<:Number,Dim} = begin
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

@inline function matern(nu, scale, variance, distance)
    if distance > 0 
        arg = sqrt(2 * nu) * distance / scale
        return variance * 2^(1-nu)/gamma(nu) * arg^nu * besselk(nu, arg)
    else
        return variance 
    end
    throw(ArgumentError("the distance should never be negative!"))
end

@inline function (k::Matern{T,Dim})(h::Union{T, ForwardDiff.Dual}) where {T, Dim}
    return matern(k.nu, k.lengthScale, 1, h)
end


@inline function (k::SquaredExponential{T,Dim})(h::Union{T, ForwardDiff.Dual}) where {T,Dim}
    return exp(-0.5 * (h / k.lengthScale)^2)
end

@inline function _taylor1(kernel::SquaredExponential{T,Dim}, d::AbstractVector{T}) where {T,Dim}
    dim = length(d)
    result = Matrix{T}(undef, dim + 1, dim + 1)
    factor = kernel(LinearAlgebra.norm(d))
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

@inline function _taylor1(kernel::Matern{T,Dim}, d::AbstractVector{T}) where {T,Dim}
    result = Matrix{T}(undef, n + 1, n + 1)
    h = LinearAlgebra.norm(d)

    result[1,1] = matern(kernel.nu, kernel.lengthScale, 1, h)
    rat = kernel.nu/(kernel.nu-1)
    fd = -rat / kernel.lengthScale^2 * matern(kernel.nu-1, kernel.lengthScale, 1, sqrt(rat) * h) 
    grad = fd * d
    result[2:end, 1] = grad
    result[1, 2:end] = -grad

    rat2 = kernel.nu/(kernel.nu-2)
    dfd = rat * rat2 / kernel.lengthScale^2 * matern(kernel.nu-2, kernel.lengthScale, 1, sqrt(rat2) * h)
    hess = (-fd) * LinearAlgebra.I + (-dfd) * d * d'
    result[2:end, 2:end] = hess

    return result
end


end # module