# IncrementalGRF.jl

```julia

using IncrementalGRF

rf = GaussianRandomField{Float64}(Kernels.squaredExponential)
rf([0., 0.])
rf.([
  [1., 0.]
  [0., 1.]
])

# Gradient Descent on Random Field
diff_rf = DifferentiableGRF{Float64}(Kernels.sqExpKernelWithGrad)

dim=10
steps=20
position = zeros(dim) # start at [0,...0]

lr = 0.1 # learning rate

for step in 1:steps
    val, grad = diff_rf(position) # incrementally evaluate RF at new position
    position -= lr*grad # update position
end

