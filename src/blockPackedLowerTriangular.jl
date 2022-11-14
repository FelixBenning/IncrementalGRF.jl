using LinearAlgebra: LinearAlgebra
using CUDA: CuArray, @allowscalar

VectorStorage = Union{Vector{T},CuArray{T}} where T;

"""
	Block Packed Lower Triangular Matrix in Row Major

	Blocksize kxk
"""
struct BlockPackedLowerTri{T, k} <: AbstractArray{T,2}
	data::VectorStorage{T}
	used_rows::Int
end

@inline g(k::Int) = k*(k+1) ÷ 2

@inline function Base.size(L::BlockPackedLowerTri{T,k}) where {T,k}
	return (L.used_rows, L.used_rows)
end

@inline @allowscalar function Base.getindex(L::BlockPackedLowerTri{T,k}, i::Int, j::Int) where {T,k}
	@boundscheck checkbounds(L, i, j)
	if i < j
		return 0
	end
	b_size = k*k
	block_loc = (row=(i-1)÷b_size, col=(j-1)÷b_size) # subtract 1 from i,j to think 0 based
	block_nr = g(block_loc.row) + block_loc.col
	
	# add 1 because julia is annoyingly 1 based
	@inbounds return L.data[block_nr*b_size + ((i-1)%b_size)*k + (j-1)%b_size + 1] 
end

@inline function Base.setindex(L::BlockPackedLowerTri{T,k}, x::T, i::Int, j::Int) where {T,k}
	@boundscheck checkbounds(L, i, j)
	if i < j
		x == 0 || throw(ArgumentError(
            "cannot set index in the upper triangular part " *
            "($i, $j) of a LowerTriangular matrix to a nonzero value ($x)")
		)
	else
		b_size = k*k
		block_loc = (row=(i-1)÷b_size, col=(j-1)÷b_size) # subtract 1 from i,j to think 0 based
		block_nr = g(block_loc.row) + block_loc.col
		
		# add 1 because julia annoyingly 1 based
		@inbounds return L.data[block_nr*b_size + ((i-1)%b_size)*k + (j-1)%b_size + 1] 
	end
end
		
"""
	\\(L,v)

finds and returns x such that L * x = v
"""
@inline function \(A::BlockPackedLowerTri{T,k}, v::AbstractVector{T}) where {T,k}
	v_len = length(v)
	@boundscheck v_len == L.used_rows || throw("L is of size $(L.used_rows) while v is of size $(v_len)")
	result = Vector{T}(undef, v_len)
	p = 0
	# TODO
end