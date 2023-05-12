using KernelFunctions
using OneHotArrays: OneHotVector
import ForwardDiff as FD

struct Pt{Dim}
	pos::AbstractArray
	partial
end

Pt(x;partial=()) = Pt{length(x)}(x, partial)

struct TaylorKernel <: KernelFunctions.Kernel
	k::KernelFunctions.Kernel
end

function (tk::TaylorKernel)(x::Pt{Dim}, y::Pt{Dim}) where Dim
	if !isnothing(local next = iterate(x.partial))
		ii, state = next # take partial derivative in direction ii
		return FD.derivative(0) do dx
			tk( # recursion
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
			tk( # recursion
				x,
				Pt(
					y.pos + dy * OneHotVector(jj, Dim), # directional variation
					partial=Base.rest(x.partial, state) # remaining partial derivatives
				)
			)
		end
	end
	tk.k(x.pos, y.pos)
end

k = TaylorKernel(MaternKernel())

k(Pt([1]), Pt([2])) # k(x,y)  with x=1, y=2
k(Pt([1], partial=(1,)), Pt([2])) # ∂ₓk(x,y)

