using LinearAlgebra.BLAS: @blasfunc 
using LinearAlgebra: LinearAlgebra, BlasFloat, BlasInt 

const libblastrampoline = "libblastrampoline"

for (tpsv, elty) in ((:stpsv_, :Float32), (:dtpsv_, :Float64))
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
