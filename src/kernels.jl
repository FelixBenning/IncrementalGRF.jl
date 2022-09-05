module Kernels

using LinearAlgebra: LinearAlgebra

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