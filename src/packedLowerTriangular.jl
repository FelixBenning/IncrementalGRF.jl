using LinearAlgebra: LinearAlgebra

using ..Blas: tpsv!


"""
	Packed LowerTriangular Matrix in Row Major format

	E.g.
	A = [
		1 
		2 3
		4 5 6
	]
	is saved as A.data=[1,2,3,4,5,6]
"""
struct PackedLowerTriangular{T} <: AbstractArray{T,2}
    data::Vector{T}
end

@inline function Base.size(L::PackedLowerTriangular{T}) where {T}
    n = (isqrt(1 + 8 * length(L.data)) - 1) ÷ 2 # length(L.data) = n(n+1)/2
    return (n, n)
end

@inline function Base.getindex(L::PackedLowerTriangular{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(L, i, j)
    if (i < j)
        return 0
    end
    @inbounds return L.data[i*(i-1)÷2+j]
end

@inline function Base.setindex(L::PackedLowerTriangular{T}, x::T, i::Int, j::Int) where {T}
    @boundscheck checkbounds(L, i, j)
    if i < j
        x == 0 || throw(ArgumentError(
            "cannot set index in the upper triangular part " *
            "($i, $j) of a LowerTriangular matrix to a nonzero value ($x)")
        )
    else
        @inbounds L.data[i*(i-1)÷2+j] = x
    end
    return L
end

"""
    \\(A,v)

finds and returns x such that A * x = v

Manually implemented Fallback
"""
@inline function Base.:\(A::PackedLowerTriangular{T}, v::AbstractVector{T}) where T
    n = length(v)
    result = Vector{T}(undef, n)
    p = 0
    for idx in 0:(n-1)
        result[idx+1] =
            (v[idx+1] - dot(result[1:idx], A.data[(p+1):p+idx])) / A.data[p+idx+1]
        p += idx + 1
    end
    return result
end

"""
    \\(A, v)

finds and returns x such that A * x = v

BLAS acceleration for Float32 and Float64
"""
@inline function Base.:\(A::PackedLowerTriangular{T}, v::AbstractVector{T}) where {T<: Union{Float32, Float64}}
	@boundscheck size(A,1) == size(v,1) || throw("Dimensions of A and v do not match")
	x = deepcopy(v)
	@inbounds solve!(A, x) 
	return x
end

"""
    solve!(A, v)

finds and returns x such that A * x = v

!!!! Overrides v with x !!!!

Manually implemented Fallback
"""
@inline function solve!(A::PackedLowerTriangular{T}, v::AbstractVector{T}) where T
    n = length(v)
    p = 0
    for idx in 0:(n-1)
        v[idx+1] -= dot(result[1:idx], A.data[(p+1):p+idx])
        v[idx+1] /= A.data[p+idx+1]
        p += idx + 1
    end
    return v
end

"""
    solve!(A, v)

finds and returns x such that A * x = v

!!!! OVERRIDES v with the solution x !!!!

BLAS acceleration for Float32 and Float64
"""
@inline function solve!(A::PackedLowerTriangular{T}, v::AbstractVector{T}) where {T<: Union{Float32, Float64}}
	@boundscheck size(A,1) == size(v,1) || throw("Dimensions of A and v do not match")
	# we are storing PackedLowerTriangular row major,
	# so we need to tell LAPACK it is 'U'pper triangular, and solve the 'T'ransposed
	# equation. Lastly entries on our diagonal might be different from 1, so
	# we do 'N'ot have a unit diagonal
	@inbounds tpsv!('U', 'T', 'N', A.data, v)
	return v
end



"""
    solve!(A, B)

solves A * X = B, where A is a PackedLowerTriangular matrix, and B is a matrix and returns X

!!!! OVERRIDES B with the solution X !!!!
"""
@inline function solve!(A::PackedLowerTriangular{T}, B::Matrix{T}) where T
    for col in eachcol(B)
        solve!(A, col)
    end
    return B
end

"""
    \\(A, B)

solves A * X = B, where A is a PackedLowerTriangular matrix, and B is a matrix and returns X
"""
function Base.:\(A::PackedLowerTriangular{T}, B::Matrix{T}) where {T}
    # possible LAPACK: https://netlib.org/lapack/explore-html/d1/df5/group__real_o_t_h_e_rcomputational_ga0f647463c7ff1c2f9bdd74cecfa263c5.html#ga0f647463c7ff1c2f9bdd74cecfa263c5
    # (apparently loops over vectors as well)
    return mapslices(x -> A \ x, B, dims=[1])
end


"""
modify L to become

[
	L  0
	l^T l_0^T
]
where l is a matrix with the same number of rows as L and l_0 is an upper triangular matrix with the same number of columns as l.

Notice how l, and l_0 are transposed because L is stored in row major format, while julia stores everything column major.
"""
function extend!(
    L::PackedLowerTriangular{T},
    l::Matrix{T},
    l_0::LinearAlgebra.UpperTriangular{T,Matrix{T}}
) where {T<:Number}
    L_size, k = size(l)
    @boundscheck L_size == size(L, 1) || throw(
        "The number of rows in L and l do not match")
    @boundscheck k == size(l_0, 1) || throw(
        "The number of columns in l and l_0 do not match")

    left = length(L.data)
    new_size = L_size + k
    resize!(L.data, (new_size) * (new_size + 1) ÷ 2) # preallocation
    for (i, l_col) in enumerate(eachcol(l))
        L.data[(left+1):(left+L_size)] = l_col # add L_size elements
        L.data[(left+L_size+1):(left+L_size+i)] = l_0[1:i, i] # add i
        left += L_size + i # this many elements where added
    end
    return L
end
