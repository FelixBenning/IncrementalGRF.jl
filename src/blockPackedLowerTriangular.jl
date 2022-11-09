using LinearAlgebra: LinearAlgebra, UpperTriangular

@inline g(k::Int) = k * (k + 1) ÷ 2

"""
	Block Packed Lower Triangular Matrix in Row Major

	Blocksize kxk
"""
struct BlockPackedLowerTri{T,k} <: AbstractMatrix{T}
    data::Vector{T}
    used_rows::Int
    BlockPackedLowerTri(L::AbstractMatrix{T}, blocksize::Int) where {T} = begin
        rows, cols = size(L)
        if rows == cols
            n = rows ÷ blocksize  # filled rows of blocks
            r = rows % blocksize

            b_size = blocksize * blocksize # elements in a block
            data = Vector{T}(undef, g(n + Int(r > 0)) * b_size) # number of blocks * b_size 
            b_idx = 0
            for b_row in 1:n
                data[b_idx+1:(b_idx+=b_row * b_size)] = vec(transpose(L[
                    ((b_row-1)*blocksize+1):b_row*blocksize,
                    1:b_row*blocksize
                ]))
            end
            if r > 0 # residual
                for block in 0:(n-1)
                    data[b_idx+1:b_idx + r * blocksize] = vec(transpose(L[
                        (n*blocksize+1):end,
                        block*blocksize+1:(block+1)*blocksize
                    ]))
                    b_idx += b_size
                end
                # corner block
                for idx in 0:(r-1)
                    data[b_idx+idx+1:b_idx + idx + r] = vec(transpose(L[
                        (n*blocksize+idx+1):(n*blocksize+idx+1),
                        (n*blocksize+1):end
                    ]))
                end
            end
            return new{T,blocksize}(data, rows)
        else
            ArgumentError(
                "The matrix L is not quadratic. As the number of rows is $rows," *
                "but the number of columns is $cols"
            )
        end
    end
end


@inline function Base.size(L::BlockPackedLowerTri{T,k}) where {T,k}
    return (L.used_rows, L.used_rows)
end

@inline function Base.getindex(L::BlockPackedLowerTri{T,k}, i::Int, j::Int) where {T,k}
    @boundscheck checkbounds(L, i, j)
    if i < j
        return zero(T)
    end
    b_size = k * k
    block_loc = (row=(i - 1) ÷ k, col=(j - 1) ÷ k) # subtract 1 from i,j to think 0 based
    block_nr = g(block_loc.row) + block_loc.col

    # add 1 because julia is annoyingly 1 based
    @inbounds return L.data[block_nr*b_size+((i-1)%k)*k+(j-1)%k+1]
end

@inline function Base.setindex(L::BlockPackedLowerTri{T,k}, x::T, i::Int, j::Int) where {T,k}
    @boundscheck checkbounds(L, i, j)
    if i < j
        x == zero(T) || throw(ArgumentError(
            "cannot set index in the upper triangular part " *
            "($i, $j) of a LowerTriangular matrix to a nonzero value ($x)")
        )
    else
        b_size = k * k
        block_loc = (row=(i - 1) ÷ b_size, col=(j - 1) ÷ b_size) # subtract 1 from i,j to think 0 based
        block_nr = g(block_loc.row) + block_loc.col

        # add 1 because julia annoyingly 1 based
        @inbounds return L.data[block_nr*b_size+((i-1)%b_size)*k+(j-1)%b_size+1]
    end
end

"""
	\\(L,B)

finds and returns x such that L * x = B
"""
@inline function Base.:\(L::BlockPackedLowerTri{T,k}, B::AbstractMatrix{T}) where {T,k}
    B_len, B_width = size(B)
    @boundscheck B_len == L.used_rows || throw("L is of size $(L.used_rows) while v is of size $(v_len)")
    n = L.used_rows ÷ k
    result = Matrix{T}(undef, B_len, B_width)
    b_size = k * k
    b_idx = 0
    for row in 0:(n-1)
        C = B[row*k+1:(row+1)*k, :] # k-sized slice of rows from B, +1 because 1-based indexing
        for idx in 0:(row-1)
            L_block = reshape(L.data[b_idx+1:(b_idx+=b_size)])
            Γ = result[idx*k+1:(idx+1)*k, :] # k-sized slice of result
            C -= Γ * transpose(L_block)
        end
        # Julia is column major, BlockPackedLowerTri is row major, so UpperTriangular is correct
        # (transpose is equivalent to switching col maj to row maj)
        L_block = transpose(UpperTriangular(reshape(L.data[b_idx+1:(b_idx+=b_size)], k, k)))
        result[row*k+1:(row+1)*k, :] = L_block \ C
    end

    n_ = L.used_rows % k
    C = B[(n*k+1):end, :]
    for idx in 0:(n-1)
        L_block = reshape(L.data[b_idx+1:(b_idx+=b_size)], k, k)[:, 1:n_]
        Γ = result[idx*k+1:(idx+1)*k, :]
        C -= transpose(L_block) * Γ
    end
    L_block = transpose(UpperTriangular(reshape(L.data[b_idx+1:(b_idx+=b_size)], k, k)[1:n_, 1:n_]))
    result[n*k+1:end, :] = L_block \ C

    return result
end

"""
	Solve X * L^T = B in-place

	L is (kn x kn) BlockPackedLowerTri{T,k} for some n
	B is kx(kn) matrix appended to L.data, changed in place into X.

	Finally L.used_rows is incremented by k.
"""
@inline function _extending_aligned_solve!(L::BlockPackedLowerTri{T,k}, new_rows::Int) where {T,k}
    b_size = k * k
    n = L.used_rows ÷ k
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
        L = reshape(L.data[loc+1:loc+b_size], k, k) # +1 because still f**ing 1-based indexing
        C /= transpose(L)
        # L.data[t+1:t+b_size] = reshape(C, :) # needed? should be inplace - fix above!
    end
    L.used_rows += k
end