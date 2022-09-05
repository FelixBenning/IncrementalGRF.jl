
struct DifferentiableGRF{T<:Number}
	grf::GaussianRandomField{T}
	
	DifferentiableGRF{T}(cov;jitter = 10*eps(T)) where T<: Number = new(
		# TODO: modify cov/auto-diff here in the future
		GaussianRandomField{T}(cov, jitter=jitter)
	)
end

function (dgrf::DifferentiableGRF{T})(x::Union{T, Vector{T}}) where T
	res = dgrf.grf(x)
	return (val=res[1], gradient=res[2:end])
end