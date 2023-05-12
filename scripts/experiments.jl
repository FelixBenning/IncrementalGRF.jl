using KernelFunctions: Kernel, KernelFunctions as KF
using OneHotArrays: OneHotVector
import ForwardDiff as FD

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

function evaluate_(k::Kernel, x::Pt{Dim}, y::Pt{Dim}) where {Dim}
	if !isnothing(local next = iterate(x.partial))
		ii, state = next # take partial derivative in direction ii
		return FD.derivative(0) do dx
			evaluate_( # recursion
				k,  
				Pt(
					x.pos + dx * OneHotVector(ii, Dim), # directional variation
					partial=Base.rest(x.partial, state) # remaining partial derivatives
				),
				y
			)
		end
	end
	if !isnothing(local next = iterate(y.partial))
		jj, state = next # take partial derivative in direction jj
		return FD.derivative(0) do dy
			evaluate_( # recursion
				k,
				x,
				Pt(
					y.pos + dy * OneHotVector(jj, Dim), # directional variation
					partial=Base.rest(x.partial, state) # remaining partial derivatives
				)
			)
		end
	end
	k(x.pos, y.pos)
end

k = KF.MaternKernel()

k([1],[2])
k(Pt([1], partial=(1,)), [2]) # ∂ₓk(x,y)

k(Pt(1, partial=(1,)), 2)
