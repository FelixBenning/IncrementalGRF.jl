
struct DifferentiableGRF{T<:Number, N}
	grf::GaussianRandomField{T, N}
end

function DifferentiableGRF(cov::CovarianceKernel{T,N}; jitter = 10*eps(T)) where {T<: Number, N}
	return DifferentiableGRF{T,N}(
		GaussianRandomField(
			Kernels.TaylorCovariance{1, T, N}(cov),
			jitter=jitter
		)
	)
end

function (dgrf::DifferentiableGRF{T, N})(x::AbstractVector{T}) where {T,N}
	res = dgrf.grf(x)
	return (val=res[1], gradient=res[2:end])
end