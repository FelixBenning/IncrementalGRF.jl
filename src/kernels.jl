module Kernels

using LinearAlgebra: LinearAlgebra

function squaredExponential(x,y)
	return exp(-LinearAlgebra.norm(x-y)^2/2)
end

end # module