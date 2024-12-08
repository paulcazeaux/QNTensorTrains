"""
    roundRandOrth!(x::TTvector{T,Nup,Ndn,d}, target_r::Vector{Matrix{Int}}, over::Int)

Truncates the ranks of `x` with specified ranks `target_r`.
Value of `over` represents oversampling
"""
function roundRandOrth!(x::TTvector{T,Nup,Ndn,d}, target_r::Vector{Matrix{Int}}, over::Int) where {T<:Number,Nup,Ndn,d}

  @assert all(target_r[  1] .== rank(x,1))
  @assert all(target_r[d+1] .== rank(x,d+1))

  # Use left-orthogonalized Gaussian or Rademacher randomized cores for framing
  Ω = tt_randn(Val(d),Val(Nup),Val(Ndn),target_r,orthogonal=:left)

  # Initialize tensor cores array for the result
  cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef,d)

  # Precompute partial projections W
  Fᴸ = Vector{Frame{T,Nup,Ndn,d,Matrix{T}}}(undef, d)
  Fᴸ[1] = IdFrame(Val(d), Val(Nup), Val(Ndn), 1)
  for k=1:d-1
    Fᴸ[k+1] = (adjoint(core(Ω,k)) * Fᴸ[k]) * core(x,k)
  end

  Fᴿ = IdFrame(Val(d), Val(Nup), Val(Ndn), d+1)
  for k=d:-1:2
    FᴸXₖFᴿ = Fᴸ[k] * core(x,k)
    Q, = cq!(FᴸXₖFᴿ)
    Fᴿ = core(x,k) * adjoint(Q)
    set_cores!(x, core(x,k-1) * Fᴿ, Q)
  end

  x.orthogonal = true
  x.corePosition = 1
  return x
end

function roundRandOrth(x::TTvector, target_r::Vector{Matrix{Int}}, over::Int)
  return roundRandOrth!(deepcopy(x), target_r, over::Int)
end

"""
    roundRandOrth2!(x::TTvector{T,Nup,Ndn,d}, m::Int)

Truncates the ranks of `x` with specified number `m=rmax+over` of random rank-one tensors, 
where `rmax` should be the maximum target rank and `over` an oversampling parameter.
"""
function roundRandOrth2!(x::TTvector{T,Nup,Ndn,d}, m::Int) where {T<:Number,Nup,Ndn,d}
  target_r = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for k=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,k)
    target_r[k][nup,ndn] = min(m, binomial(k-1,nup-1)*binomial(k-1,ndn-1), binomial(d+1-k,Nup+1-nup)*binomial(d+1-k,Ndn+1-ndn))
  end
  @assert target_r[  1][1,1] == rank(x,1,0)
  @assert target_r[d+1][Nup+1,Ndn+1] == rank(x,d+1,Nup+1,Ndn+1)

  # Initialize tensor cores array for the result
  cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef,d)

  # Precompute partial projections W
  Fᴸ = Vector{Frame{T,Nup,Ndn,d,Matrix{T}}}(undef, d)
  Ω₁ = randh(Val(d),Val(Nup),Val(Ndn),m)
  Fᴸ[2] = adjoint(Ω₁) * core(x, 1)
  for k=2:d-1
    # Use randomized diagonal cores (i.e. block-TT representation of m rank-one vectors) for projection
    Ωₖ = randd(Nup,Ndn,d,k,m)
    Fᴸ[k+1] = (adjoint(Ωₖ) * Fᴸ[k]) * core(x,k)
  end

  Fᴿ = IdFrame(Val(d), Val(Nup), Val(Ndn), d+1)
  for k=d:-1:2
    FᴸXₖFᴿ = Fᴸ[k] * core(x,k)
    Q, = cq!(FᴸXₖFᴿ)
    Fᴿ = core(x,k) * adjoint(Q)
    set_cores!(x, core(x,k-1) * Fᴿ, Q)
  end

  x.orthogonal = true
  x.corePosition = 1
  return x
end

function roundRandOrth2(tt::TTvector, rmax::Int, over::Int)
  return round!(roundRandOrth2!(deepcopy(tt),rmax+over), rmax=rmax)
end