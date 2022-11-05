using CUDA
using LinearAlgebra

ENV["JULIA_CUDA_USE_BINARYBUILDER"]=false;

if abspath(PROGRAM_FILE) == @__FILE__ # equivalent to if name == __main__
    a = CUDA.rand(2,2);
    a = a * a';
    C = cholesky(a);
    println("\n",C)
end