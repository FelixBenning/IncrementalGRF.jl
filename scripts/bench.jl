using IncrementalGRF, Pkg
Pkg.activate("test")

include("../test/benchmark_suite.jl")

old_results = B.load("test/manual_local_benchmark.json")[1]
results = runTunedSuite("test/params.json")

B.judge(B.minimum(results), B.minimum(old_results))

