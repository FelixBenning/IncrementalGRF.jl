using KernelFunctions
using OneHotArrays: OneHotVector
using ForwardDiff: derivative

struct Pt{Dim}
	pos::AbstractArray
	partial
end

Pt(x;partial=()) = Pt{length(x)}(x, partial)

struct TaylorKernel <: KernelFunctions.Kernel
	k::KernelFunctions.Kernel
end

function (tk::TaylorKernel)(x::Pt{Dim}, y::Pt{Dim}) where Dim
	k = tk.k
	for ii in x.partial
		k = (x₁,x₂) -> derivative(0) do Δx
			x₁[ii] += Δx
			return k(x₁,x₂)
		end
	end
	for jj in y.partial
		k = (x₁,x₂) -> derivative(0) do Δx
			x₂[jj] += Δx
			return k(x₁,x₂)
		end
	end
	k(x.pos, y.pos)
end

k = TaylorKernel(MaternKernel())

k(Pt([1]), Pt([2])) # k(x,y)  with x=1, y=2
k(Pt([1], partial=(1,)), Pt([2])) # ∂ₓk(x,y)

