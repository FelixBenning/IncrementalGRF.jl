using Mongoc: Mongoc
using Dates: Dates
using Flux: Flux
using ProgressLogging: ProgressLogging as PL
using IncrementalGRF

client = Mongoc.Client("localhost", 27017)

database = client["optimizer-benchmarking"]
collection = database["recorded-optim-runs"]

abstract type Optimizer end

mutable struct SquaredExponentialMomentum <: Optimizer
	step::Int
	scale
	velocity
	
	SquaredExponentialMomentum() = new(1)
end

function (opt::SquaredExponentialMomentum)(rf::DifferentiableGRF, pos)
	y = copy(pos)
	try
		y += (opt.step-1)/(opt.step+2) * opt.velocity
	catch e
		e isa UndefRefError || rethrow(e)
		opt.scale = rf.grf.cov.k.scale
	end
	val, grad = rf(y)
	opt.step += 1
	new_pos = y - opt.scale * optimal_rate(val, grad) * 0.5 * grad
	opt.velocity = new_pos - pos
	return new_pos, val, grad

end
mutable struct SquaredExponentialGrad <: Optimizer
	scale
	SquaredExponentialGrad() = new()
end

function (opt::SquaredExponentialGrad)(rf::DifferentiableGRF, pos)
	try
		opt.scale
	catch e
		e isa UndefRefError || rethrow(e)
		opt.scale = rf.grf.cov.k.scale
	end
	val, grad = rf(pos)
	new_pos = pos - opt.scale * optimal_rate(val, grad) * grad
	return new_pos,val, grad
end

function (opt::Flux.Optimise.AbstractOptimiser)(rf::DifferentiableGRF, pos)
	pos_copy = copy(pos)
	val, grad = rf(pos_copy)
	Flux.update!(opt, pos_copy, grad)
	return pos_copy, val, grad
end

function optimRF(opt, dim, scale, steps)
	document = Mongoc.BSON()
	document["scale"] = scale
	document["dim"] = dim
	document["steps"] = steps
	document["optimizer"] = repr(opt)
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

function dbfilter(collection=collection;optimizer=nothing, dim=nothing, min_steps=nothing)
	filter = Mongoc.BSON()
	if !isnothing(optimizer)
		filter["optimizer"] = repr(optimizer)
	end
	if !isnothing(dim)
		filter["dim"] = dim
	end
	if !isnothing(min_steps)
		filter["steps"] = Dict(raw"$gte" => min_steps)
	end
	Mongoc.find(collection, filter)
end


doc = Mongoc.find_one(collection)
# Mongoc.delete_many(collection, Mongoc.BSON())