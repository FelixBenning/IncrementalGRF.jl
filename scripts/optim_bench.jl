
include("optimiser.jl")
include("optim_benchmarking.jl")

client = Mongoc.Client("localhost", 27017)

database = client["optimizer-benchmarking"]
collection = database["recorded-optim-runs"]


doc = Mongoc.find_one(collection)
# Mongoc.delete_many(collection, Mongoc.BSON())


mongoOptimRF(SquaredExponentialGrad(), dim=1000, scale=0.05, steps=30)