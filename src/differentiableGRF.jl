
struct DifferentiableGRF{Order,T<:Number,N}
    grf::GaussianRandomFunction{T,N}
end

function DifferentiableGRF{Order}(cov::CovarianceKernel{T,N}; jitter=10 * eps(T)) where {Order, T<:Number,N}
    return DifferentiableGRF{Order,T,N}(
        GaussianRandomFunction(
            Kernels.TaylorCovariance{Order}(cov),
            jitter=jitter
        )
    )
end

function DifferentiableGRF(cov::CovarianceKernel{T,N}; jitter=10 * eps(T)) where {T<:Number,N}
    return DifferentiableGRF{1}(cov, jitter=jitter)
end

function (dgrf::DifferentiableGRF{1,T,N})(x::AbstractVector{T}) where {T,N}
    res = dgrf.grf(x)
    return (val=res[1], gradient=res[2:end])
end

function conditionals(dgrf::DifferentiableGRF{1,T,N}) where {T,N}
    cond = conditionals(dgrf.grf)
    return x::AbstractVector{T} -> begin
        exp, var = cond(x)
        return (val=exp[1], gradient=exp[2:end], cov=var)
    end
end
function conditionalExpectation(dgrf::DifferentiableGRF{1,T,N}) where {T,N}
    cond = conditionalExpectation(dgrf.grf)
    return x::AbstractVector{T} -> begin
        res = cond(x)
        return (val=res[1], gradient=res[2:end])
    end
end
