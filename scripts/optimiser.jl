using Flux: Flux
using LinearAlgebra: LinearAlgebra
using RandomFunctions

abstract type Optimizer end

function (opt::Flux.Optimise.AbstractOptimiser)(rf::DifferentiableGRF, pos)
	pos_copy = copy(pos)
	val, grad = rf(pos_copy)
	Flux.update!(opt, pos_copy, grad)
	return pos_copy, val, grad
end

function optimal_rate(loss, grad)
	g_norm = LinearAlgebra.norm(grad)
	a= loss/(2*g_norm)
	return (a + sqrt(a^2+1))/g_norm
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

mutable struct DiminishingLRGD <: Optimizer
	constant
	step
	DiminishingLRGD(constant) = new(constant, 1)
end

function (opt::DiminishingLRGD)(rf::DifferentiableGRF, pos)
	val, grad= rf(pos)
	new_pos  = pos - opt.constant/opt.step * grad
	opt.step +=1
	return new_pos, val, grad
end