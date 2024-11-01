"""
    roundRandOrth!(x::TTvector{T,N,d}, target_r::Vector{OffsetVector{Int,Vector{Int}}}, over::Int)

Truncates the ranks of `x` with specified ranks `target_r`.
Value of `over` represents oversampling
"""
function roundRandOrth!(x::TTvector{T,N,d}, target_r::Vector{OffsetVector{Int,Vector{Int}}}, over::Int) where {T<:Number,N,d}

  @assert target_r[1][0] == rank(x,1,0)
  @assert target_r[d+1][N] == rank(x,d+1,N)

  # Use left-orthogonalized Gaussian or Rademacher randomized cores for framing
  Ω = tt_randn(Val(d),Val(N),target_r,orthogonal=:left)

  # Initialize tensor cores array for the result
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef,d)

  # Precompute partial projections W
  Fᴸ = Vector{Frame{T,N,d,Matrix{T}}}(undef, d)
  Fᴸ[1] = IdFrame(Val(N), Val(d), 1)
  for k=1:d-1
    Fᴸ[k+1] = (adjoint(core(Ω,k)) * Fᴸ[k]) * core(x,k)
  end

  Fᴿ = IdFrame(Val(N), Val(d), d+1)
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

function roundRandOrth(x::TTvector, target_r::Vector{OffsetVector{Int,Vector{Int}}}, over::Int)
  return roundRandOrth!(deepcopy(x), target_r, over::Int)
end

"""
    roundRandOrth2!(x::TTvector{T,N,d}, m::Int)

Truncates the ranks of `x` with specified number `m=rmax+over` of random rank-one tensors, 
where `rmax` should be the maximum target rank and `over` an oversampling parameter.
"""
function roundRandOrth2!(x::TTvector{T,N,d}, m::Int) where {T<:Number,N,d}
  target_r = [[min(m, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]

  @assert target_r[1][0] == rank(x,1,0)
  @assert target_r[d+1][N] == rank(x,d+1,N)

  # Use left-orthogonalized Gaussian or Rademacher randomized cores for framing
  Ω = tt_randn(Val(d),Val(N),target_r,orthogonal=:left)

  # Initialize tensor cores array for the result
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef,d)

  # Precompute partial projections W
  Fᴸ = Vector{Frame{T,N,d,Matrix{T}}}(undef, d)
  Fᴸ[1] = IdFrame(Val(N), Val(d), 1)
  for k=1:d-1
    # Use randomized diagonal cores (i.e. block-TT representation of m rank-one vectors) for projection
    Ωₖ = randd(N,d,k,m)
    Fᴸ[k+1] = (adjoint(Ωₖ) * Fᴸ[k]) * core(x,k)
  end

  Fᴿ = IdFrame(Val(N), Val(d), d+1)
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