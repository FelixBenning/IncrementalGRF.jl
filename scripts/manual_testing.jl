
using IncrementalGRF
using Plots: plot

##
k = Kernels.Matern{Float64, 1}(1.5,1.)
tay = Kernels.TaylorCovariance{1}(k)
@run Kernels._taylor1(k, [0.])

@enter res = tay(zeros(Float64,1), zeros(Float64,1))
@run rf = DifferentiableGRF(k)


x = -10:0.1:10
plot(x, vcat(rf.(x)...), show=true, label="")

##

