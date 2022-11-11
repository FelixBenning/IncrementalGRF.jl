Pkg.activate(".")
using IncrementalGRF, Pkg
Pkg.activate("test")

include("../test/benchmark_suite.jl")

results = runTunedSuite("test/params.json")
old_results = B.load("test/manual_local_benchmark.json")[1]

B.judge(B.minimum(results), B.minimum(old_results))


