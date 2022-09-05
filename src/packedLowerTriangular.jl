using LinearAlgebra.BLAS: @blasfunc 
using LinearAlgebra: BlasFloat, BlasInt, tril!

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
finds and returns x such that A * x = b
"""
function \(A::PackedLowerTriangular{T}, v::AbstractVector{T}) where T
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
BLAS acceleration for Float32 and Float64
"""
@inline function \(A::PackedLowerTriangular{T}, v::AbstractArray{T}) where {T<: Union{Float32, Float64}}
	@boundscheck size(A,1) == size(v,1) || throw("Dimensions of A and v do not match")
	x = deepcopy(v)
	# we are storing PackedLowerTriangular row major,
	# so we need to tell LAPACK it is "U"pper triangular, and solve the "T"ransposed
	# equation. Lastly entries on our diagonal might be different from 1, so
	# we do "N"ot have a unit diagonal
	@inbounds tpsv!("U", "T", "N", A.data, x)
	return x
end


for (tpsv, elty) in ((:stpsv_, :Float32), (:dtpsv_, :Float64))
    # This function solves the system of equations
	# A * x = b inplace.
    # For documentation of function arguments, see 
    # This function does not yield information on success or failure
    @eval begin
		"""
			Solve
				A * x = b  OR A' * x = b
			where A is a packed storage triangular matrix

			uplo: 
				= "U" A is upper triangular
				= "L" A is lower triangular
			
			trans:
				= "N" solve A * x = b
				= "T" solve A^T * x = b
			
			diag:
				= "U" assume that A has unit diagonal
				= "N" do not assume that
			
			AP: A vector of length n*(n+1)/2 where n=length(b) representing the packed matrix A

			x: when passed to this function should contain the values in b,
			they will then be overwritten by the result

			Calls LAPACK stpv/dtpsv
			- https://netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2_gae6fb0355e398779dc593ced105ce373d.html#gae6fb0355e398779dc593ced105ce373d
			- https://netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_ga0fff73e765a7655a67da779c898863f1.html#ga0fff73e765a7655a67da779c898863f1
		"""
        @inline function tpsv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, AP::AbstractVector{$elty}, x::AbstractVector{$elty})
            N = size(x, 1)
            incx = 1
            @boundscheck size(AP, 1) == Int32(N * (N + 1) / 2) || throw(
                ArgumentError("AP needs to be a vector, packing the upper or lower part of a square matrix of the same number of rows as x."
                ))
            @boundscheck ((uplo in ['U', 'L']) && (trans in ['N', 'T'])) && (diag in ['U', 'N']) || throw(
                ArgumentError("Wrong character input in scalar arguments")
            )

            ccall(
                (@blasfunc($tpsv), libblastrampoline), # :stpsv_ or :dtpsv_ in "libblastrampoline" library
                Cvoid, # return type void
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}), # argument types
                uplo, trans, diag, Int32(N), AP, x, Int32(incx) # arguments
            )

        end
    end
end


"""
solves A * X = B, where A is a PackedLowerTriangular matrix, and B is a matrix and returns X
"""
function \(A::PackedLowerTriangular{T}, B::Matrix{T}) where {T}
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