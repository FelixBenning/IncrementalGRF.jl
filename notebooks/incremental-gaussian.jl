### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 60e95558-aeaf-4759-9460-8da1dbc28c54
begin
	import Pkg
	Pkg.activate(".")
	Pkg.develop(path="..")
	Pkg.add("LinearAlgebra")
	Pkg.add("Plots")
	Pkg.add("Mongoc")
	Pkg.add("ProgressLogging")
	Pkg.add("PlutoUI")
	Pkg.add("LaTeXStrings")
	Pkg.add("Flux")
	Pkg.add("Distributions")
	Pkg.add("RandomMatrices")
	Pkg.add("HypertextLiteral")
	Pkg.add("PGFPlots")
	Pkg.add("Measures")
end

# ╔═╡ 4d5ceb64-18e2-40b6-b6ab-9a7befbe27b2
begin
	using LinearAlgebra: LinearAlgebra, dot, issuccess
	using Test: @test
	using Random: Random
	using Plots: plot, plot!, savefig, Plots
	using ProgressLogging: @progress
	using Logging: Logging, SimpleLogger, with_logger
	using PlutoUI: Slider, PlutoUI
	using HypertextLiteral: @htl, HypertextLiteral
	using LaTeXStrings
	using Measures
end

# ╔═╡ 42170044-fed1-4e1c-8254-93e33b21a0b7
using IncrementalGRF

# ╔═╡ 9af5239a-7090-4f40-9ac4-96c9760a5d35
begin
	using Mongoc: Mongoc
	using Dates: Dates
	using ProgressLogging: ProgressLogging as PL
end

# ╔═╡ 08a40e67-33ac-424c-806c-e775e90b4bd7
using Flux: Flux

# ╔═╡ 506140dd-5b00-4475-b367-f101260aa637
using RandomMatrices

# ╔═╡ a5ab4c31-4a85-484b-984e-0b72311368f3
md"# Test 1-dim Gaussian Random Function"

# ╔═╡ cde271c5-f62a-44a5-aaca-b7ebcbec1788
function differentiabilitySlider()
	values = round.(tan.(range(start=atan(1.01),stop=π/2,length=100)), sigdigits=3)
	values[57:end] = round.(values[57:end], sigdigits=2)
	values=values[1:end-1]
    values[end] = Inf64
	# plot(1:length(values), values)
	return Slider(values, show_value=true)
end

# ╔═╡ 41f523c5-c3c9-4dc9-9185-4bf3c1285345
function maternParamPicker(;sdv=0.01:0.01:10, scale=0.1:0.1:5, dim=1:300)
	return PlutoUI.combine() do Child
		@htl("""
		<h3>Matern Random Function</h3>
		<dl class="controls">
			<div class="item">
				<dt>Dimension</dt>
				<dd>$(Child(:dim, Slider(dim, default=1, show_value=true)))</dd>
			</div>

			<div class="item">
		    	<dt>Differentiability (ν)</dt>
				<dd>$(Child(:nu, differentiabilitySlider()))</dd>
			</div>
		
			<div class="item">
		    	<dt>Scale</dt>
				<dd>
				$(Child(:scale, Slider(scale, default=1, show_value=true)))
				</dd>
			</div>

			<div class="item">
				<dt>Standard Deviation</dt>
				<dd>
				$(Child(:sdv, Slider(sdv, default=1, show_value=true)))
				</dd>
			</div>
		</dl>
		<style>
              .controls {
		 			display: grid;
					grid-template-columns: repeat(auto-fit, minmax(300px, 1fr))
			  }
			  .item {
			  }
		</style>
		""")
	end
end

# ╔═╡ 4498205d-e0e6-452d-900b-ef6ca0de30cc
function kernel_from_param(k_param::NamedTuple{(:dim, :nu, :scale, :sdv)})
	return k_param.nu < Inf ? 
		Kernels.Matern{Float64, k_param.dim}(
			nu=k_param.nu, 
			scale=k_param.scale, 
			variance=k_param.sdv^2
		) : Kernels.SquaredExponential{Float64, k_param.dim}(
			scale=k_param.scale,
			variance=k_param.sdv^2
		)
end

# ╔═╡ 310164cc-ad23-4db0-bcfe-ccf487d721ea
x = -10:0.02:10

# ╔═╡ 8439b954-81b0-4e34-9e76-0cc4d290dd5b
@bind k_param maternParamPicker(dim=1:2)

# ╔═╡ 7abb752e-1ea4-476f-92d1-58ea2b02511b
kernel = kernel_from_param(k_param)

# ╔═╡ 51be2a30-538d-4d10-bb69-53c0aac3d92f
rf = DifferentiableGRF(kernel)

# ╔═╡ 63f0a57a-5b91-4518-bf3b-f5d21fcf3f0e
plot(x, kernel.([[elt] for elt in x]), label="covariance kernel", fontfamily="Computer Modern")

# ╔═╡ 80c50c25-0af3-408e-9df1-2c5a99ecb059
 plot(x, vcat(GaussianRandomFunction(kernel,jitter=1e-10).(x)...), 
	 label=L"Z(x)", legend=:topright, fontfamily="Computer Modern")

# ╔═╡ 3e33bc57-b014-4618-ace5-1d14e9f313b1
function plotExpectationAgainstPlot(rf::DifferentiableGRF{1,Float64, 1})
	val, grad = rf([0.])
	cE = conditionalExpectation(rf)
	
	cE_evals = map(r->r.val, cE.([[elt] for elt in x]))
	Z_evals = map(r->r.val, rf.([[elt] for elt in x]))
	lims = (
		min(minimum(Z_evals), minimum(cE_evals)), 
		max(maximum(Z_evals), maximum(cE_evals))
	) .* 1.2
	
	pl = plot(legend=:topright, fontfamily="Computer Modern", ylim=lims)

	plot!(pl, x, Z_evals, label=L"\mathcal{L}(x)")
	plot!(pl, x, cE_evals, 
		label=L"\mathbb{E}[\mathcal{L}(x)\mid \mathcal{L}(0),\nabla \mathcal{L}(0)]", linestyle=:dash)
	plot!(pl, x, map(t-> val + t* first(grad), x), 
		label=L"T[\mathcal{L}(x)\mid \mathcal{L}(0), \nabla\mathcal{L}(0)]",
		linestyle=:dot)

	plot!(pl, [0.], [val], seriestype=:scatter, color=1, label="", ms=1.5)
	plot!(pl, size=(300,300))
	return pl
end

# ╔═╡ a8f77ffa-1c24-4bd9-ba43-86dd8bee4fe8
begin
	cond_plot = plotExpectationAgainstPlot(DifferentiableGRF(kernel, jitter=1e-10))
	savefig("plots/scale=$(kernel.scale),std=$(sqrt(kernel.variance)),nu=$(k_param.nu).svg")
	cond_plot
end

# ╔═╡ 702178e1-d0b6-4b0e-bf47-3a31acb34b77
md"# Test 2-dim Gaussian Random Function"

# ╔═╡ d85c6f84-91a1-4b90-a19a-c981ed331d5c
pairs(x) = ( (a,b) for (k,a) in enumerate(x) for b in Iterators.drop(x, k) )

# ╔═╡ 5e63220a-5bec-443b-b0a1-ebb20763ca1f
begin
	drf = DifferentiableGRF(
		Kernels.SquaredExponential{Float64,2}(), 
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
md"# Optimization on Random Function"

# ╔═╡ 76da78ed-c48e-45d6-bdd4-f7c2668fdab7
function connect(;
	user = "felixbenning",
	password=ENV["mongoDBpassword"]
)
	# https://github.com/felipenoris/Mongoc.jl/issues/69#issuecomment-946953526
	# Need julia version 1.7: suffix = "tlsCAFile=$(pkgdir(IncrementalGRF, "scripts/cert.pem"))"
	suffix = "tlsCAFile=$(joinpath(dirname(pathof(IncrementalGRF)), "..", "scripts/cert.pem"))"
	cluster = "rf-simulations.lqksh0j.mongodb.net"
	uri = "mongodb+srv://$user:$password@$cluster/?$(suffix)"
	return Mongoc.Client(uri)
end

# ╔═╡ c28573ad-d698-4258-9b51-cd87d4a9a84b
md"MongoDB password: $(@bind pwd PlutoUI.PasswordField())"

# ╔═╡ 8850bf86-9da2-4f20-8ec0-8fa338eb8b16
begin
	client = connect(password=pwd)
	database = client["optimizer-benchmarking"]
	collection = database["recorded-optim-runs"]
	Mongoc.ping(client)
end

# ╔═╡ 86c6e77e-9301-46ac-bf22-7d5dc78db3d9
# https://github.com/fonsp/Pluto.jl/issues/115#issuecomment-661722426
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 2f621535-e1f5-44fc-b64e-de2512e439b4
function (opt::Flux.Optimise.AbstractOptimiser)(rf::DifferentiableGRF, pos)
	pos_copy = copy(pos)
	val, grad = rf(pos_copy)
	Flux.update!(opt, pos_copy, grad)
	return pos_copy, val, grad
end

# ╔═╡ f4bb022d-3857-4378-bd9c-08c39f12132f
abstract type Optimizer end

# ╔═╡ d329a235-fe41-4a03-a4b3-8a57c5898626
function optimal_rate(loss, grad)
	g_norm = LinearAlgebra.norm(grad)
	a= loss/(2*g_norm)
	return (a + sqrt(a^2+1))/g_norm
end

# ╔═╡ 6769a366-071e-4b3a-99b7-3db5939fe537
begin
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

	function Base.repr(s::SquaredExponentialMomentum)
		try
			return "SquaredExponentialMomentum($(s.step), $(s.scale), $(s.velocity))"
		catch e
			if (e isa UndefRefError)
				return "SquaredExponentialMomentum(1, #undef, #undef)"
			end
		end
	end
	SquaredExponentialMomentum
end

# ╔═╡ bd7cae19-a3cd-42e6-8d4f-2ad3a86bb03b
begin
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
	function Base.repr(s::SquaredExponentialGrad)
		try
			return "SquaredExponentialGrad($(s.scale))"
		catch e
			if (e isa UndefRefError)
				return "SquaredExponentialGrad(#undef)"
			end
		end
	end
	SquaredExponentialGrad
end

# ╔═╡ 5159700c-05da-490c-b14f-62e5cf5b09f0
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

# ╔═╡ 5ff78030-c6fa-4a50-be97-32c518f2418a
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

# ╔═╡ 368cc59b-0650-49bd-92b8-a8ab8ff20df6
begin
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
	
	DiminishingLRGD
end

# ╔═╡ b86794ca-a3ca-4947-adf3-6be9289e7465
available_optimiser = Dict(
	:RFD=>SquaredExponentialGrad,
	:"RFM*"=>SquaredExponentialMomentum,
	:Adam=> Flux.Optimise.Adam,
	:RMSProp => Flux.Optimise.RMSProp,
	:AdaMax => Flux.Optimise.AdaMax,
	:AdaGrad => Flux.Optimise.AdaGrad,
	:AdaDelta => Flux.Optimise.AdaDelta,
	:AMSGrad => Flux.Optimise.AMSGrad,
	:NAdam => Flux.Optimise.NAdam,
	:AdamW => Flux.Optimise.AdamW,
	:RAdam => Flux.Optimise.RAdam
)

# ╔═╡ 42fede2d-da64-4517-8db7-6fbb9a76741e
begin
	local default_opt = [:RFD, :"RFM*", :Adam, :NAdam]
	@bind ui PlutoUI.confirm(
		PlutoUI.combine() do Child
			@htl("""
			<h3>Random Function Parameters</h3>
			<p>
				dimension: $(Child(:dim, PlutoUI.NumberField(1:300, default=30)))
				Covariance-scale: $(
					Child(:scale, PlutoUI.NumberField(0.01:0.01:10, default=0.05))
				)
			</p>
			<h3>Optimization Parameters</h3>
			<p>
				steps: $(Child(:steps, PlutoUI.NumberField(1:60, default=25)))
				repeats: $(Child(:repeats, PlutoUI.NumberField(1:1000, default=10)))
			</p>
			
			<h3>Optimizers</h3>

			$(Child(
				"active_optimiser",
				PlutoUI.MultiCheckBox(
					available_optimiser |> keys |> collect,
					default=default_opt
				)
			))
			""")
		end
	)
end

# ╔═╡ 80bc99b5-677e-4073-a617-a863a6a2c4ce
md"
- LogLog: $(@bind loglog PlutoUI.CheckBox(default=false)) 
- plot sqrt: $(@bind sqrtplot PlutoUI.CheckBox(default=false)) 
- shift up by: $(@bind shift Slider(0:0.1:2, default=0,show_value=true))"

# ╔═╡ 62c798ac-79c1-4971-ab4a-ae0a50e6f9a3
begin
	gradPlot = plot(
		legend=:topright, fontfamily="Computer Modern",
		size=(300,300), yticklabelsize=5 #ticks=false
	)
	optimiser = Dict(x=>available_optimiser[x] for x in ui.active_optimiser)
	for (idx, (name, opt)) in enumerate(optimiser)
		simulations = dbfilter(
			optimiser=opt(),
			dim=ui.dim, 
			min_steps=ui.steps,
			scale=ui.scale
		) |> collect
		@progress for _ in 1:(ui.repeats-length(simulations))
			# fill database
			println("filling database")
			mongoOptimRF(
				opt(), dim=ui.dim, scale=ui.scale, steps=ui.steps
			)
		end
		for (it, sim) in enumerate(simulations)
			lstyle = ([:solid,:dash,:dashdot, :dashdotdot])[(idx-1) % 4 + 1]
			plot!(
				gradPlot, 
				sim["values"][1:ui.steps] .+ shift, 
				label=((it==1) ? String(name) : ""), 
				color=idx, 
				linestyle=lstyle, 
				lw=lstyle == :solid ? 0.9 : 1.1
			)
		end
	end
	if loglog 
		plot!(gradPlot, yaxis=:log, xaxis=:log, legend=:bottomleft)
	end
	if sqrtplot
		plot!(
			gradPlot, 
			1:ui.steps, shift ./ sqrt.(1:ui.steps), 
			linewidth=4, linestyle=:dot, label=L"\frac{%$(shift)}{\sqrt{n}}",
			linecolor=:black
		)
		# plot!(
		# 	gradPlot, 
		# 	1:ui.steps, shift ./ (1:ui.steps), 
		# 	linewidth=4, linestyle=:dot, label=L"\frac{%$(shift)}{n}"
		# )
	end
	savefig(
		"plots/$(ui.dim)dim_$(ui.steps)steps_s=$(ui.scale)_$(
			join(String.(keys(optimiser)),"_")
		).svg"
	)
	gradPlot
end

# ╔═╡ 1ad684c6-129c-449b-9eea-3a8c9dd0ac96
md"## End of Iteration Value Distribution"

# ╔═╡ edb84732-fbca-4248-b47e-4c5459df2674
@bind opt_key PlutoUI.Select([x=> String(x) for x in ui.active_optimiser])

# ╔═╡ e6114d6d-87f4-41cc-a6f8-c314a024a15f
begin
	final_vals = [
		sim["values"][ui.steps] 
		for sim in dbfilter(
			optimiser=available_optimiser[opt_key](), 
			dim=ui.dim, min_steps=ui.steps
		)
	]
	plot(
		final_vals, 
		seriestype=:histogram, normalize=:pdf, bins=length(final_vals) ÷ 10, label=String(opt_key)
	)
end

# ╔═╡ c0df62ff-d312-45d6-a001-0ac9c1b4e34b
md"## Gradient Directions"

# ╔═╡ 11a92e07-aa82-4f04-adda-d7227858061e
begin
	orthPlot = plot()
	local opt = SquaredExponentialGrad()
	simulation = first(dbfilter(optimiser=opt, dim=1000, min_steps=25))
	vals, grads = simulation["values"], simulation["gradients"]
	local grid = reshape(
		[
			dot(g1, g2)/(LinearAlgebra.norm(g2)^2)
			for g1 in grads for g2 in grads
		], 
		length(grads), :
	)
	plot!(
		orthPlot, grid, seriestype=:heatmap,
		title=latexstring(
			"Projection \$ π_{g}(\\tilde{g}\\,)=\\langle \\tilde{g}, g\\, \\rangle/\\|g\\, \\|^2\$"
		),
		ylabel=latexstring("gradient \$\\tilde{g}\$"), 
		xlabel=latexstring("gradient \$g\$"),
		fontfamily="Computer Modern"
	)
	plot(vals, label=nothing)
end

# ╔═╡ 68e7f3bf-e06e-4440-af93-b7e6fe54379d
plot!(orthPlot, size=(400,400))

# ╔═╡ 2bc50209-34d0-442c-91d3-9243749849a7
savefig(orthPlot, "plots/projections.svg")

# ╔═╡ 8208bd08-8a5f-4c14-a6a2-f1e212a27c6f
plot(map(LinearAlgebra.norm, grads))

# ╔═╡ d5a432d2-7b8e-42eb-8d2b-a4469e59dfcc
md"# Minima Distribution"

# ╔═╡ 94188b20-75fc-4fa1-b5a0-121881e1b59a
t =TracyWidom()

# ╔═╡ 4da265fa-5e46-4469-b37e-5c67c34c56d2
F(x) = cdf(t, x, beta=1)

# ╔═╡ 45bc3e50-20d5-43b0-be2b-891e12107afc
cdf_points = F.(x)

# ╔═╡ d96e1366-5302-48d6-a56e-9a1cef69aa63
pdf_points = map((x,y)-> (y-x)/0.01, cdf_points[1:end-1], cdf_points[2:end])

# ╔═╡ 9bd87493-9874-41df-b979-98d3890836af
plot(x[2:end], pdf_points, label="TracyWidom pdf")

# ╔═╡ ecd8a804-851f-4cd1-b84d-f465cbe6658c
F(-1.25)

# ╔═╡ 775e3420-6a1c-420d-bcba-7383dd35e617
md"# Appendix"

# ╔═╡ Cell order:
# ╠═4d5ceb64-18e2-40b6-b6ab-9a7befbe27b2
# ╠═42170044-fed1-4e1c-8254-93e33b21a0b7
# ╟─a5ab4c31-4a85-484b-984e-0b72311368f3
# ╟─41f523c5-c3c9-4dc9-9185-4bf3c1285345
# ╟─cde271c5-f62a-44a5-aaca-b7ebcbec1788
# ╠═4498205d-e0e6-452d-900b-ef6ca0de30cc
# ╟─7abb752e-1ea4-476f-92d1-58ea2b02511b
# ╠═310164cc-ad23-4db0-bcfe-ccf487d721ea
# ╟─51be2a30-538d-4d10-bb69-53c0aac3d92f
# ╟─63f0a57a-5b91-4518-bf3b-f5d21fcf3f0e
# ╟─80c50c25-0af3-408e-9df1-2c5a99ecb059
# ╟─8439b954-81b0-4e34-9e76-0cc4d290dd5b
# ╠═a8f77ffa-1c24-4bd9-ba43-86dd8bee4fe8
# ╟─3e33bc57-b014-4618-ace5-1d14e9f313b1
# ╟─702178e1-d0b6-4b0e-bf47-3a31acb34b77
# ╟─d85c6f84-91a1-4b90-a19a-c981ed331d5c
# ╟─5e63220a-5bec-443b-b0a1-ebb20763ca1f
# ╠═9dbbc977-7641-4a68-98bc-31d5e5847233
# ╟─601ef169-392c-4c6b-857d-eb20139d4e81
# ╠═9af5239a-7090-4f40-9ac4-96c9760a5d35
# ╟─76da78ed-c48e-45d6-bdd4-f7c2668fdab7
# ╟─c28573ad-d698-4258-9b51-cd87d4a9a84b
# ╟─8850bf86-9da2-4f20-8ec0-8fa338eb8b16
# ╟─5159700c-05da-490c-b14f-62e5cf5b09f0
# ╟─5ff78030-c6fa-4a50-be97-32c518f2418a
# ╟─86c6e77e-9301-46ac-bf22-7d5dc78db3d9
# ╠═08a40e67-33ac-424c-806c-e775e90b4bd7
# ╠═2f621535-e1f5-44fc-b64e-de2512e439b4
# ╠═f4bb022d-3857-4378-bd9c-08c39f12132f
# ╟─d329a235-fe41-4a03-a4b3-8a57c5898626
# ╟─6769a366-071e-4b3a-99b7-3db5939fe537
# ╟─bd7cae19-a3cd-42e6-8d4f-2ad3a86bb03b
# ╟─368cc59b-0650-49bd-92b8-a8ab8ff20df6
# ╠═b86794ca-a3ca-4947-adf3-6be9289e7465
# ╟─42fede2d-da64-4517-8db7-6fbb9a76741e
# ╟─80bc99b5-677e-4073-a617-a863a6a2c4ce
# ╟─62c798ac-79c1-4971-ab4a-ae0a50e6f9a3
# ╟─1ad684c6-129c-449b-9eea-3a8c9dd0ac96
# ╟─edb84732-fbca-4248-b47e-4c5459df2674
# ╟─e6114d6d-87f4-41cc-a6f8-c314a024a15f
# ╟─c0df62ff-d312-45d6-a001-0ac9c1b4e34b
# ╟─11a92e07-aa82-4f04-adda-d7227858061e
# ╠═68e7f3bf-e06e-4440-af93-b7e6fe54379d
# ╠═2bc50209-34d0-442c-91d3-9243749849a7
# ╟─8208bd08-8a5f-4c14-a6a2-f1e212a27c6f
# ╟─d5a432d2-7b8e-42eb-8d2b-a4469e59dfcc
# ╠═506140dd-5b00-4475-b367-f101260aa637
# ╠═94188b20-75fc-4fa1-b5a0-121881e1b59a
# ╠═4da265fa-5e46-4469-b37e-5c67c34c56d2
# ╠═45bc3e50-20d5-43b0-be2b-891e12107afc
# ╠═d96e1366-5302-48d6-a56e-9a1cef69aa63
# ╠═9bd87493-9874-41df-b979-98d3890836af
# ╠═ecd8a804-851f-4cd1-b84d-f465cbe6658c
# ╟─775e3420-6a1c-420d-bcba-7383dd35e617
# ╠═60e95558-aeaf-4759-9460-8da1dbc28c54
