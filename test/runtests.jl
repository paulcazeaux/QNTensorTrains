using Test
using LinearAlgebra
using QNTensorTrains

# Helper draw function
function draw!(ω)
  i = rand(ω)
  pop!(ω, i)
  return i
end

function draw_state(::Val{d}, ::Val{N}) where {d,N}
  ω = Set(1:d)
  s = ntuple(i->draw!(ω), Val(N))

  return tt_state(s, Val(d))
end

function approx_state(::Val{d},::Val{N},r,ϵ) where {N,d}
  ω = Set(1:d)
  s = ntuple(i->draw!(ω), Val(N))

  return round!(perturbation(tt_state(s, Val(d)),r,ϵ))
end

include("base/sparsecore.jl")
include("base/tensor.jl")
include("base/round.jl")
include("base/dot.jl")
include("base/matrixfree.jl")
include("base/hamiltonian.jl")
include("base/tangent.jl")
include("solvers/alsmals.jl")
include("random/round.jl")
include("random/sum.jl")
include("random/matvec.jl")
include("solvers/lanczos.jl")