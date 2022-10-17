
struct DifferentiableGRF{Order,T<:Number,N}
    grf::GaussianRandomField{T,N}
end

function DifferentiableGRF{Order}(cov::CovarianceKernel{T,N}; jitter=10 * eps(T)) where {Order, T<:Number,N}
    return DifferentiableGRF{Order,T,N}(
        GaussianRandomField(
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

function conditionalExpectation(dgrf::DifferentiableGRF{1,T,N}) where {T,N}
    cond = conditionalExpectation(dgrf.grf)
    return x::AbstractVector{T} -> begin
        res = cond(x)
        return (val=res[1], gradient=res[2:end])
    end
end
