### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 60e95558-aeaf-4759-9460-8da1dbc28c54
begin
	import Pkg
	Pkg.activate(".")
	Pkg.develop(path="..")
	Pkg.add("LinearAlgebra")
	Pkg.add("Plots")
	Pkg.add("ProgressLogging")
	Pkg.add("PlutoUI")
end

# ╔═╡ 4d5ceb64-18e2-40b6-b6ab-9a7befbe27b2
begin
	using LinearAlgebra: LinearAlgebra, dot, issuccess
	using Test: @test
	using Random: Random
	using Plots: plot, plot!
	using ProgressLogging: @progress
	using Logging: Logging, SimpleLogger, with_logger
	using PlutoUI: Slider
end

# ╔═╡ 42170044-fed1-4e1c-8254-93e33b21a0b7
using IncrementalGRF

# ╔═╡ b7b14883-2aae-40ab-bde1-6c0a186a9da8
k = Kernels.TaylorCovariance{1}(Kernels.SquaredExponential{Float64,3}(1))

# ╔═╡ 9618cca3-4d80-48c4-b3a1-15ef1235dec0
begin
	struct Test{O, T, V<:AbstractVector{T}}
		v::V
	end
	function Test{Ord}(v::V) where {Ord, T, V<:AbstractVector{T}}
		return Test{Ord, T, V}(v)
	end
end

# ╔═╡ f6664955-89b3-492d-8296-39cf70d0184c
Test{1}([1,2])

# ╔═╡ 32b7614a-f3aa-4cd3-82ba-0a5f6015be57
k([0.,0, 0],[2.,0, 0])

# ╔═╡ a5ab4c31-4a85-484b-984e-0b72311368f3
md"# Test 1-dim Gaussian Random Field"

# ╔═╡ 51be2a30-538d-4d10-bb69-53c0aac3d92f
rf = GaussianRandomField(Kernels.SquaredExponential{Float64, 1}(1.))

# ╔═╡ 310164cc-ad23-4db0-bcfe-ccf487d721ea
x = -10:0.1:10

# ╔═╡ a99bbd91-a5f1-4b21-bc63-90014d7b3914
plot(x, vcat(rf.(x)...), show=true)

# ╔═╡ 702178e1-d0b6-4b0e-bf47-3a31acb34b77
md"# Test 2-dim Gaussian Random Field"

# ╔═╡ d85c6f84-91a1-4b90-a19a-c981ed331d5c
pairs(x) = ( (a,b) for (k,a) in enumerate(x) for b in Iterators.drop(x, k) )

# ╔═╡ 5e63220a-5bec-443b-b0a1-ebb20763ca1f
begin
	drf = DifferentiableGRF(
		Kernels.SquaredExponential{Float64,2}(1.), 
		jitter=0.00001
	)

	discr = -5:0.5:5
	grid = [drf([x,y]) for x in discr for y in discr]

	

	discr_fine = -5:0.1:5 # 0.1 takes 160seconds
	cond = conditionalExpectation(drf)
	grid_fine = [cond([x,y]) for x in discr_fine for y in discr_fine]
	
	plt = plot(
		discr_fine, discr_fine, (x->x[:val]).(grid_fine),
		seriestype=:contour
	)
end

# ╔═╡ 9dbbc977-7641-4a68-98bc-31d5e5847233
plot!(plt, [0], [0], quiver=(drf([0.,0])[:gradient]), seriestype=:quiver)

# ╔═╡ 601ef169-392c-4c6b-857d-eb20139d4e81
md"# Gradient Descent"

# ╔═╡ 424b60c3-ff83-420f-90f6-503e1b03bb34
dim= 100

# ╔═╡ 5fc2a003-0f07-4c0b-91a2-9cf99a7af62b
steps = 25

# ╔═╡ d329a235-fe41-4a03-a4b3-8a57c5898626
function optimal_rate(step, loss, grad)
	g_norm = LinearAlgebra.norm(grad)
	a= loss/(2*g_norm)
	return (a + sqrt(a^2+1))/g_norm
end

# ╔═╡ 8bddd6fc-b434-41f3-b958-5cf33ee024fd
function gradientDescent(dim, steps, lr=optimal_rate)
	high_dim_rf = DifferentiableGRF(
		Kernels.SquaredExponential{Float64, dim}(1), jitter=0.00001)

	local position = zeros(dim)
	vals = Vector{Float64}(undef, steps)
	grads = Matrix{Float64}(undef, dim, steps)
	for step in 1:steps
		vals[step], grads[:,step] = high_dim_rf(position)
		position -= lr(step, vals[step], grads[:,step]) * grads[:,step]
	end
	return vals, grads, high_dim_rf
end

# ╔═╡ 102fe6f5-5177-4a7d-ae30-516ff851358c
repeats=10

# ╔═╡ 0402ec92-b8be-4e5f-8643-2d8382fc130e
begin
	gradPlot = plot()
	@progress for it in 1:repeats # good GD
		vals, _, _ =  gradientDescent(dim, steps)
		plot!(gradPlot, vals, label=((it==1) ? "optimal GD" : ""), color=1)
	end
	@progress for it in 1:repeats # 1/n GD
		vals, _, _ =  gradientDescent(dim, steps, (step,_,_)->1/step)
		plot!(gradPlot, vals, label=((it==1) ? "1/n GD" : ""), color=2)
	end
	@progress for it in 1:repeats # 2/n GD
		vals, _, _ =  gradientDescent(dim, steps, (step,_,_)->2/step)
		plot!(gradPlot, vals, label=((it==1) ? "2/n GD" : ""), color=3)
	end
	@progress for it in 1:repeats # 0.5/n GD
		vals, _, _ =  gradientDescent(dim, steps, (step,_,_)->0.1/step)
		plot!(gradPlot, vals, label=((it==1) ? "0.1/n GD" : ""), color=4)
	end
	gradPlot
end

# ╔═╡ 11a92e07-aa82-4f04-adda-d7227858061e
begin
	orthPlot = plot()
	vals, grads, _ = gradientDescent(dim, steps)
	local grid = reshape(
		[
			dot(g1, g2)/(LinearAlgebra.norm(g1)*LinearAlgebra.norm(g2))
			for g1 in eachcol(grads) for g2 in eachcol(grads)
		], 
		size(grads, 2), :
	)
	plot!(orthPlot, grid, seriestype=:heatmap)
	plot(vals, label=nothing)
end

# ╔═╡ 68e7f3bf-e06e-4440-af93-b7e6fe54379d
orthPlot

# ╔═╡ 775e3420-6a1c-420d-bcba-7383dd35e617
md"# Appendix"

# ╔═╡ Cell order:
# ╠═4d5ceb64-18e2-40b6-b6ab-9a7befbe27b2
# ╠═42170044-fed1-4e1c-8254-93e33b21a0b7
# ╠═b7b14883-2aae-40ab-bde1-6c0a186a9da8
# ╠═9618cca3-4d80-48c4-b3a1-15ef1235dec0
# ╠═f6664955-89b3-492d-8296-39cf70d0184c
# ╠═32b7614a-f3aa-4cd3-82ba-0a5f6015be57
# ╟─a5ab4c31-4a85-484b-984e-0b72311368f3
# ╠═51be2a30-538d-4d10-bb69-53c0aac3d92f
# ╠═310164cc-ad23-4db0-bcfe-ccf487d721ea
# ╠═a99bbd91-a5f1-4b21-bc63-90014d7b3914
# ╟─702178e1-d0b6-4b0e-bf47-3a31acb34b77
# ╠═d85c6f84-91a1-4b90-a19a-c981ed331d5c
# ╠═5e63220a-5bec-443b-b0a1-ebb20763ca1f
# ╠═9dbbc977-7641-4a68-98bc-31d5e5847233
# ╟─601ef169-392c-4c6b-857d-eb20139d4e81
# ╠═424b60c3-ff83-420f-90f6-503e1b03bb34
# ╠═5fc2a003-0f07-4c0b-91a2-9cf99a7af62b
# ╠═d329a235-fe41-4a03-a4b3-8a57c5898626
# ╠═8bddd6fc-b434-41f3-b958-5cf33ee024fd
# ╠═102fe6f5-5177-4a7d-ae30-516ff851358c
# ╠═0402ec92-b8be-4e5f-8643-2d8382fc130e
# ╠═11a92e07-aa82-4f04-adda-d7227858061e
# ╠═68e7f3bf-e06e-4440-af93-b7e6fe54379d
# ╟─775e3420-6a1c-420d-bcba-7383dd35e617
# ╟─60e95558-aeaf-4759-9460-8da1dbc28c54
