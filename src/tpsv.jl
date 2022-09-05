using LinearAlgebra.BLAS: @blasfunc 

using LinearAlgebra: BlasFloat, BlasInt, tril!
using Test

const libblastrampoline = "libblastrampoline"

for (tpsv, elty) in ((:stpsv_, :Float32), (:dtpsv_, :Float64))
    # This function solves the system of equations A * x = b inplace.
    # For documentation of function arguments, see https://netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2_gae6fb0355e398779dc593ced105ce373d.html#gae6fb0355e398779dc593ced105ce373d
    # This function does not yield information on success or failure
    @eval begin
        function tpsv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, AP::AbstractVector{$elty}, x::AbstractVector{$elty})
            N = size(x, 1);
            incx = 1;            
            @assert (size(AP,1) == Int32(N * (N+1)/2) ) "AP needs to be a vector, packing the upper or lower part of a square matrix of the same number of rows as x."
            @assert ((uplo ∈ ['U','L']) & (trans ∈ ['N','T'])) & (diag ∈ ['U','N']) "Wrong character input in scalar arguments"
            
            ccall((@blasfunc($tpsv), libblastrampoline), Cvoid,(Ref{UInt8},Ref{UInt8},Ref{UInt8},Ref{BlasInt},Ptr{$elty}, Ptr{$elty},Ref{BlasInt}), uplo, trans, diag, Int32(N), AP, x, Int32(incx))

        end
    end
end

@testset showtiming = false verbose = false "Test correct linkage of tpsv for $type, size $k\n" for type in (Float32, Float64), k in [5,50,500]
        M = randn(type,(k,k));
        M = M * M; # generate positive definite matrix
        AP = M[tril!(trues(size(M)))]; # extract lower triangular matrix as vecotr
        b = rand(type,k); # generate RHS
        x = deepcopy(b);
        tpsv!('U', 'N', 'N', AP, x);
        @test sum(abs.(M * x == b)) < 0.01  
end
