using KernelFunctions: Kernel, KernelFunctions as KF
using OneHotArrays: OneHotVector
import ForwardDiff as FD
import LinearAlgebra as LA

function partial(fun, dim, partials=())
	if !isnothing(local next = iterate(partials))
		ii, state = next
		return partial(
			x -> FD.derivative(0) do dx
				fun(x .+ dx * OneHotVector(ii, dim))
			end,
			dim,
			Base.rest(partials, state),
		)
	end
	return fun 
end

function partial(k, dim; partials_x=(), partials_y=())
	local f(x,y) = partial(t -> k(t,y), dim, partials_x)(x)
	return (x,y) -> partial(t -> f(x,t), dim, partials_y)(y)
end
struct Pt{Dim}
	pos::AbstractArray
	partial
end

Pt(x;partial=()) = Pt{length(x)}(x, partial)

for T in subtypes(Kernel)
	(k::T)(x::Pt{Dim}, y::Pt{Dim}) where {Dim} = evaluate_(k, x, y)
	(k::T)(x::Pt{Dim}, y) where {Dim} = evaluate_(k, x, Pt(y))
	(k::T)(x, y::Pt{Dim}) where {Dim} = evaluate_(k, Pt(x), y)
end

function evaluate_(k::T, x::Pt{Dim}, y::Pt{Dim}) where {Dim, T<:Kernel}
	return partial(
		k, Dim,
		partials_x=x.partial, partials_y=y.partial
	)(x.pos, y.pos)
end

k = KF.MaternKernel()

k([1],[1])
k([1], Pt([1], partial=(1,1)))
k(Pt([1], partial=1), [2]) # ∂ₓk(x,y)
k(Pt([1,2], partial=(1)), Pt([1,2], partial=2))# ∂ₓk(x,y)


k = KF.SEKernel()

kappa(x) = exp(- LA.dot(x,x)/2)


g = partial(k,1, partials_x=2, partials_y=0)
g(1,1)
