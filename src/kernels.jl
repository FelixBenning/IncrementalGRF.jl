module Kernels

using LinearAlgebra: LinearAlgebra
using Zygote: Zygote, ForwardDiff
using SpecialFunctions: besselk, gamma
import ..StationaryKernel, ..IsotropicKernel, ..CovarianceKernel

"""
    function (k::StationaryKernel{T,Dim})(x::AbstractVector{T}, y::AbstractVector{T})

    Calculate `k(x,y)` where `k` is a stationary covariance kernel.
"""
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
    scale::T
    variance::T
    @doc raw"""

        SquaredExponential{T,Dim} <: IsotropicKernel{T,Dim}
        
        SquaredExponential{T,Dim}(
            ;scale::T=one(T),
            variance::T=one(T),
            scale_var_by_dim::Bool=true
        )
    
    Construct the Squared Exponential Kernel
    ```math
        C(h) = \sigma^2\exp\Bigl(-\frac{h^2}{2s^2}\Bigr)
    ```
    where
    - s = scale
    - ``\sigma^2`` = (variance / Dim)   if `scale_var_by_dim` is true
    - ``\sigma^2`` = variance         if `scale_var_by_dim` is false
    """
    SquaredExponential{T,Dim}(
        ;scale::T=one(T), variance::T=one(T), scale_var_by_dim::Bool=true
    ) where {T<:Number, Dim} = begin
        scale > 0 || throw(ArgumentError("scale is not positive"))
        variance > 0 || throw(ArgumentError("variance is not positive"))
        return new(scale, scale_var_by_dim ? variance/Dim : variance)
    end
end

struct Matern{T<:Number, Dim} <: IsotropicKernel{T,Dim}
    nu::Real
    scale::T
    variance::T
    @doc raw"""

        Matern{T<:Number, Dim} <: IsotropicKernel{T,Dim}

        Matern{T, Dim}(
            ;nu::Real,
            scale::T=one(T),
            variance::T=one(T),
            scale_var_by_dim::Bool=true
        )

    Construct the Matérn covariance Kernel
    ```math
        C_\nu(h) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \Bigl(\frac{\sqrt{2\nu}}{s} h \Bigr)^\nu
        K_\nu\Bigl(\frac{\sqrt{2\nu}}{s}h\Bigr)
    ```
    where
    - ``\nu`` = nu
    - s = scale
    - ``\sigma^2`` = (variance / Dim)   if `scale_var_by_dim` is true
    - ``\sigma^2`` = variance         if `scale_var_by_dim` is false
    """
    Matern{T, Dim}(
        ; nu::Real, scale::T=one(T), variance::T=one(T), scale_var_by_dim::Bool=true
    ) where {T<:Number, Dim} = begin
        scale > 0 || throw(ArgumentError("scale is not positive"))
        nu > 0 || throw(ArgumentError("nu is not positive"))
        variance > 0 || throw(ArgumentError("variance is not positive"))
        return new(nu, scale, scale_var_by_dim ? variance/sqrt(Dim) : variance)
    end
end


"""
    Calculates ``arg^ν K_ν(arg)``, scaled such that it is 1 for arg=0.
    Here K_ν is the modified bessel function.
"""
@inline function xbesselk(nu, arg)
    if arg > 0
        return 2^(1-nu)/gamma(nu) * arg^nu * besselk(nu, arg)
    else
        return 1
    end
    throw(ArgumentError("only positive arg expected"))
end

@inline function (k::Matern{T,Dim})(h::Union{T, ForwardDiff.Dual}) where {T, Dim}
    arg = sqrt(2*k.nu) * h / k.scale
    return k.variance * xbesselk(k.nu, arg)
end


@inline function (k::SquaredExponential{T,Dim})(h::Union{T, ForwardDiff.Dual}) where {T,Dim}
    return k.variance * exp(-0.5 * (h / k.scale)^2)
end

@inline function _taylor1(kernel::SquaredExponential{T,Dim}, d::AbstractVector{T}) where {T,Dim}
    dim = length(d)
    result = Matrix{T}(undef, dim + 1, dim + 1)
    factor = kernel(LinearAlgebra.norm(d))
    result[1, 1] = factor
    dl = d / (kernel.scale^2)
    grad = factor * dl
    result[2:end, 1] = -grad
    result[1, 2:end] = grad
    result[2:end, 2:end] = (
        factor * LinearAlgebra.I / kernel.scale^2 - grad * transpose(dl)
    )
    return result
end

@inline function _taylor1(kernel::Matern{T,Dim}, d::AbstractVector{T}) where {T,Dim}
    result = Matrix{T}(undef, Dim + 1, Dim + 1)
    h = LinearAlgebra.norm(d) / kernel.scale

    arg = sqrt(2*kernel.nu) * h

    result[1,1] = kernel.variance * xbesselk(kernel.nu, arg) 
    rat = kernel.nu/(kernel.nu-1)
    fd = - kernel.variance * rat / kernel.scale^2 * xbesselk(kernel.nu -1, arg)
    grad = fd * d
    result[2:end, 1] = grad
    result[1, 2:end] = -grad

    rat2 = kernel.nu/(kernel.nu-2)
    dfd = kernel.variance * rat * rat2 / kernel.scale^2 * xbesselk(kernel.nu -2, arg)
    hess = (-fd) * LinearAlgebra.I + (-dfd) * d * d'
    result[2:end, 2:end] = hess

    return result
end


end # module