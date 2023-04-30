
include("optimiser.jl")
include("optim_benchmarking.jl")

function connect(;
	user = "felixbenning",
	password=ENV["mongoDBpassword"]
)
	# https://github.com/felipenoris/Mongoc.jl/issues/69#issuecomment-946953526
	# Need julia version 1.7: suffix = "tlsCAFile=$(pkgdir(RandomFunctions, "scripts/cert.pem"))"
	suffix = "tlsCAFile=$(joinpath(dirname(pathof(RandomFunctions)), "..", "scripts/cert.pem"))"
	cluster = "rf-simulations.lqksh0j.mongodb.net"
	uri = "mongodb+srv://$user:$password@$cluster/?$(suffix)"
	return Mongoc.Client(uri)
end

client = connect()

database = client["optimizer-benchmarking"]
collection = database["recorded-optim-runs"]

Mongoc.ping(client)

doc = Mongoc.find_one(collection)
# Mongoc.delete_many(collection, Mongoc.BSON())


mongoOptimRF(SquaredExponentialGrad(), dim=100, scale=0.05, steps=30)