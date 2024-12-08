using Test
using LinearAlgebra
using QNTensorTrains
using QNTensorTrains: Spin,Up,Dn

function randomized_state(d::Int, N::Int, Sz::Rational, ranks_range::UnitRange{Int})
    Nup = Int(N+2Sz)÷2
    Ndn = Int(N-2Sz)÷2
    r = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
    for k=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,k)
      r[k][nup,ndn] = (k in (1,d+1) ? 1 : rand(ranks_range))
    end  
    return tt_randn(Val(d),Val(Nup),Val(Ndn),r)
end

# Helper draw function
function draw!(ω)
  i = rand(ω)
  pop!(ω, i)
  return i
end

function draw_state(::Val{d}, ::Val{N}, ::Val{Sz}) where {d,N,Sz}
  ω = Set(1:d)
  Nup = Int(N+2Sz)÷2
  Ndn = Int(N-2Sz)÷2
  sup = ntuple(i->draw!(ω), Val(Nup))
  sdn = ntuple(i->draw!(ω), Val(Ndn))

  return tt_state(sup, sdn, Val(d))
end

function approx_state(::Val{d},::Val{N},::Val{Sz},ranks_range::UnitRange{Int},ϵ) where {d,N,Sz}
  ω = Set(1:d)
  Nup = Int(N+2Sz)÷2
  Ndn = Int(N-2Sz)÷2
  sup = ntuple(i->draw!(ω), Val(Nup))
  sdn = ntuple(i->draw!(ω), Val(Ndn))
  
  r = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for k=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,k)
    r[k][nup,ndn] = (k in (1,d+1) ? 1 : rand(ranks_range))
  end  

  return round!(perturbation(tt_state(sup, sdn, Val(d)),r,ϵ))
end

include("base/sparsecore.jl")
include("base/tensor.jl")
include("base/round.jl")
include("base/dot.jl")
include("base/elementaryops.jl")
include("base/hamiltonian.jl")
include("base/tangent.jl")
include("solvers/alsmals.jl")
include("solvers/lanczos.jl")
include("random/round.jl")
include("random/sum.jl")
include("random/matvec.jl")