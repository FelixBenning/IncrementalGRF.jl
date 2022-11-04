using LinearAlgebra: LinearAlgebra

"""
	Block Packed Lower Triangular Matrix in Row Major

	Blocksize kxk
"""
struct BlockPackedLowerTri{T, k} <: AbstractMatrix{T}
	data::Vector{T}
	used_rows::Int
end

@inline g(k::Int) = k*(k+1) ÷ 2

@inline function Base.size(L::BlockPackedLowerTri{T,k}) where {T,k}
	return (L.used_rows, L.used_rows)
end

@inline function Base.getindex(L::BlockPackedLowerTri{T,k}, i::Int, j::Int) where {T,k}
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
	\\(L,B)

finds and returns x such that L * x = v
"""
@inline function \(L::BlockPackedLowerTri{T,k}, B::AbstractMatrix{T}) where {T,k}
	v_len = length(v)
	@boundscheck v_len == L.used_rows || throw("L is of size $(L.used_rows) while v is of size $(v_len)")
	result = Vector{T}(undef, v_len)
	p = 0
	# TODO
end

"""
	Solve X * L^T = B in-place

	L is (kn x kn) BlockPackedLowerTri{T,k} for some n
	B is kx(kn) matrix appended to L.data, changed in place into X.

	Finally L.used_rows is incremented by k.
"""
@inline function _extending_aligned_solve!(L::BlockPackedLowerTri{T,k}) where {T,k}
	b_size = k*k
	n = L.used_rows / k
	for row in 0:(n-1)
		t = (g(n) + row) * b_size
		C = reshape(L.data[t+1:t+b_size], k, k) # +1 because 1-based indexing
		for idx in 0:(row-1)
			loc = (g(row) + idx) * b_size
			L = reshape(L.data[loc+1:loc+b_size], k, k) # +1 because 1-based indexing
			loc = (g(n) + idx) * b_size
			Γ = reshape(L.data[loc+1:loc+b_size], k, k) # +1 because 1-based indexing
			C -= Γ * transpose(L)
		end
		loc = (g(row) + row) * b_size
		L = reshape(L.data[loc+1: loc + b_size], k, k) # +1 because still f**ing 1-based indexing
		C /= transpose(L)
		# L.data[t+1:t+b_size] = reshape(C, :) # needed? should be inplace - fix above!
	end
	L.used_rows += k
end