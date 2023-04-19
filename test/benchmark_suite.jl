using LinearAlgebra: LinearAlgebra
using BenchmarkTools: BenchmarkTools as B, BenchmarkGroup

function oneDimGaussian(n)
	grf = GaussianRandomFunction(Kernels.SquaredExponential{Float64,1}(1.))
	grf.(range(-50, stop=50, length=n))
end

function gradientDescent(dim, steps)
	high_dim_rf = DifferentiableGRF(
		Kernels.SquaredExponential{Float64, dim}(1.), jitter=0.00001)

	local position = zeros(dim)
	vals = Vector{Float64}(undef, steps)
	grads = Matrix{Float64}(undef, dim, steps)
	for step in 1:steps
		vals[step], grads[:,step] = high_dim_rf(position)
		g_norm = LinearAlgebra.norm(grads[:,step])
		a = vals[step]/(2*g_norm)
		lr = a + sqrt(a^2 + 1)
		position -= lr * grads[:,step]/g_norm
	end
	return vals, grads, high_dim_rf
end

function defineSuite()
	suite = BenchmarkGroup()
	for dim in [1, 10, 100, 300]
		suite[["GradientDescent", "$(dim)-dim"]] = B.@benchmarkable gradientDescent($dim, 30)
	end

	for n in [10, 100, 500]
		suite[["oneDimGaussian","$n points"]] = B.@benchmarkable oneDimGaussian($n)
	end

	for dim in [1,10, 100]
		local k = Kernels.SquaredExponential{Float64, dim}(randn()^2)
		local x = randn(dim)
		local y = randn(dim)

		suite[["Kernel", "SquaredExponential", "$(dim)-dim"]] = B.@benchmarkable $k($x,$y)


		tk = Kernels.TaylorCovariance{1}(k)
		suite[
			["Kernel", "SquaredExponential", "TaylorCovariance", "$(dim)-dim"]
		] = B.@benchmarkable $tk($x,$y)
	end
			
	return suite
end

function tuneSuite(suite, param_json::AbstractString="")
	if isfile(param_json)
		B.loadparams!(suite, B.load(param_json)[1], :evals, :samples)
	else 
		B.tune!(suite)
	end
	return suite
end

function runTunedSuite(param_json::AbstractString="", result_path::AbstractString=""; verbose::Bool=true, seconds::Int=100)
	suite = tuneSuite(defineSuite(), param_json)
	result = B.run(suite, verbose=verbose, seconds=seconds)
	if isfile(param_json) || (splitext(param_json)[2] == ".json")
		B.save(param_json, B.params(suite))
	end
	if isfile(result_path) || (splitext(result_path)[2] == ".json")
		B.save(result_path, result)
	end
	return result
end
