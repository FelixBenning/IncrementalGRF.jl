# IncrementalGRF.jl

```julia

using IncrementalGRF

rf = GaussianRandomFunction(Kernels.SquaredExponential{Float64, 2}(1))
rf([0., 0.])
rf.([
  [1., 0.]
  [0., 1.]
])

# Gradient Descent on Random Function
dim=10
diff_rf = DifferentiableGRF(Kernels.SquaredExponential{Float64, dim}(1))

steps=20
position = zeros(dim) # start at [0,...0]

lr = 0.1 # learning rate

for step in 1:steps
    val, grad = diff_rf(position) # incrementally evaluate RF at new position
    position -= lr*grad # update position
end

