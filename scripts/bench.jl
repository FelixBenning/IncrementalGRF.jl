Pkg.activate(".")
using IncrementalGRF, Pkg
Pkg.activate("test")

include("../test/benchmark_suite.jl")

if isfile("test/manual_local_benchmark.json")
    old_results = B.load("test/manual_local_benchmark.json")[1]
end

results = runTunedSuite("test/params.json")

B.judge(B.minimum(results), B.minimum(old_results))

