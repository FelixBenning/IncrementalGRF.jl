
using Mongoc: Mongoc

using Dates: Dates
using ProgressLogging: ProgressLogging as PL
using RandomFunctions

function mongoOptimRF(opt; dim, scale, steps)
	document = Mongoc.BSON()
	document["scale"] = scale
	document["dim"] = dim
	document["steps"] = steps
	document["optimiser"] = repr(opt)
	document["git-hash"] = readchomp(`git rev-parse HEAD`)
	document["date"] = string(Dates.now())

	document["covariance-function"] = "SquaredExponential" 
	rf = DifferentiableGRF(
		Kernels.SquaredExponential{Float64,dim}(scale=scale), 
		jitter=0.000001
	)

	local pos = zeros(dim)
	vals = Vector{Float64}(undef, steps)
	grads = Vector{Vector{Float64}}(undef, steps)

	PL.progress(name="Optimization on Random Function") do id
		for step in 1:steps
			@info "$(step)/$(steps) steps" progress=(step/steps)^3 _id=id
			pos, vals[step], grads[step] = opt(rf, pos)
		end
	end
	document["values"] = vals
	document["gradients"] = grads
	push!(collection, document)
end

function dbfilter(
	collection=collection;
	optimiser=nothing, 
	dim=nothing,
	min_steps=nothing,
	scale=nothing
)
	filter = Mongoc.BSON()
	if !isnothing(optimiser)
		filter["optimiser"] = repr(optimiser)
	end
	if !isnothing(dim)
		filter["dim"] = dim
	end
	if !isnothing(scale)
		filter["scale"] = scale
	end
	if !isnothing(min_steps)
		filter["steps"] = Dict(raw"$gte" => min_steps)
	end
	Mongoc.find(collection, filter)
end


