module IncrementalGRF

export
	PackedLowerTriangular,
	BlockPackedLowerTri,
	GaussianRandomFunction,
	DifferentiableGRF,
	Kernels,
	CovarianceKernel,
	StationaryKernel,
	IsotropicKernel,
	conditionalExpectation,
	conditionals

abstract type CovarianceKernel{T<:Number,N} end
abstract type StationaryKernel{T<:Number,N} <: CovarianceKernel{T,N} end
abstract type IsotropicKernel{T<:Number,N} <: StationaryKernel{T,N} end

include("blas.jl")
include("packedLowerTriangular.jl")
include("blockPackedLowerTriangular.jl")
include("GRF.jl")
include("differentiableGRF.jl")
include("kernels.jl")

end # module
