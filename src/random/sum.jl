"""
  roundRandSum(α::Vector{Number}, summands::Vector{TTvector{T,Nup,Ndn,d}}, target_r::Vector{Matrix{Int}})

Truncates the ranks of `tt` with specified ranks `target_r`.
"""
function roundRandSum(α::Vector{T1}, summands::Vector{TTvector{T,Nup,Ndn,d,M}}, target_r::Vector{Matrix{Int}}) where {T1<:Number,T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  @boundscheck @assert length(α) == length(summands)
  @assert all(all(target_r[  1] .== rank(S,  1)) for S in summands)
  @assert all(all(target_r[d+1] .== rank(S,d+1)) for S in summands)

  # Use left-orthogonalized Gaussian or Rademacher randomized cores for framing
  Ω = tt_randn(Val(d),Val(Nup),Val(Ndn),target_r,orthogonal=:left)

  # Initialize tensor cores array for the result
  cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef,d)

  # Precompute partial projections W
  Fᴸ = Matrix{Frame{T,Nup,Ndn,d,Matrix{T}}}(undef, length(summands), d)
  for s=1:length(summands)
    Fᴸ[s,1] = IdFrame(Val(d), Val(Nup), Val(Ndn), 1)
    for k=1:d-1
      Fᴸ[s,k+1] = (adjoint(core(Ω,k)) * Fᴸ[s,k]) * core(summands[s],k)
    end
  end

  SₖFᴿ = [ core(summands[s],d) * lmul!(α[s], IdFrame(Val(d), Val(Nup), Val(Ndn), d+1)) for s in axes(summands,1)]
  for k=d:-1:2
    FᴸSₖFᴿ = Fᴸ[1,k] * SₖFᴿ[1]
    for s = 2:length(summands)
      mul!(FᴸSₖFᴿ, Fᴸ[s,k], SₖFᴿ[s], 1, 1)
    end
    Q, = cq!(FᴸSₖFᴿ)
    cores[k] = Q

    if k>2
      SₖFᴿ = [ core(summands[s],k-1) * (SₖFᴿ[s] * adjoint(Q)) for s in axes(summands,1) ]
    else
      cores[1] = core(summands[1],1) * (SₖFᴿ[1] * adjoint(Q))
      for s = 2:length(summands)
        mul!(cores[1], core(summands[s],1), (SₖFᴿ[s] * adjoint(Q)), 1, 1)
      end
    end
  end

  tt = cores2tensor(cores)
  tt.orthogonal = true
  tt.corePosition = 1
  return tt
end

"""
  roundRandSum(α::Vector{Number}, summands::Vector{TTvector{T,Nup,Ndn,d}}, max_rank::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function roundRandSum(α::Vector{T1}, summands::Vector{TTvector{T,Nup,Ndn,d,M}}, rmax::Int, over::Int) where {T1<:Number,T<:Number,Nup,Ndn,d,M<:AbstractMatrix}
  target_r = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for k=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,k)
    target_r[k][nup,ndn] = over + min(rmax, binomial(k-1,nup)*binomial(k-1,ndn), binomial(d+1-k,N-nup)*binomial(d+1-k,N-ndn))
  end
  # [[(k∈(1,d+1) ? 1 : rmax+over) for n in QNTensorTrains.state_qn(Nup,Ndn,d,k)] for k=1:d+1]
  return round!(roundRandSum(α,summands,target_r), rmax=rmax)
end

"""
  roundRandSum2(α::Vector{Number}, summands::Vector{TTvector{T,Nup,Ndn,d}}, m::Int)

Truncates the ranks of `tt` with specified ranks `target_r` using a randomized projection onto `m` rank-one vectors.
"""
function roundRandSum2(α::Vector{T1}, summands::Vector{TTvector{T,Nup,Ndn,d,M}}, m::Int) where {T1<:Number,T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  target_r = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for k=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,k)
    target_r[k][nup,ndn] = min(m, binomial(k-1,nup)*binomial(k-1,ndn), binomial(d+1-k,N-nup)*binomial(d+1-k,N-ndn))
  end
  @boundscheck @assert length(α) == length(summands)
  @assert all(all(target_r[  1] .== rank(S,  1)) for S in summands)
  @assert all(all(target_r[d+1] .== rank(S,d+1)) for S in summands)

  # Initialize tensor cores for the result
  cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef,d)

  # Precompute partial projections W
  Fᴸ = Matrix{Frame{T,Nup,Ndn,d,Matrix{T}}}(undef, length(summands), d)

  Ω₁ = randh(Val(d),Val(Nup),Val(Ndn),m)
  for s=1:length(summands)
    Fᴸ[s,2] = adjoint(Ω₁) * core(summands[s],1)
  end
  for k=2:d-1
    Ωₖ = randd(Val(d),Val(Nup),Val(Ndn),k,m)
    for s=1:length(summands)
      Fᴸ[s,k+1] = adjoint(Ωₖ) * (Fᴸ[s,k] * core(summands[s],k))
    end
  end

  SₖFᴿ = [ core(summands[s],d) * lmul!(α[s], IdFrame(Val(d), Val(Nup), Val(Ndn), d+1)) for s in axes(summands,1)]
  for k=d:-1:2
    FᴸSₖFᴿ = Fᴸ[1,k] * SₖFᴿ[1]
    for s = 2:length(summands)
      mul!(FᴸSₖFᴿ, Fᴸ[s,k], SₖFᴿ[s], 1, 1)
    end
    Q, = cq!(FᴸSₖFᴿ)
    cores[k] = Q

    if k>2
      SₖFᴿ = [ core(summands[s],k-1) * (SₖFᴿ[s] * adjoint(Q)) for s in axes(summands,1)]
    else
      cores[1] = core(summands[1],1) * (SₖFᴿ[1] * adjoint(Q))
      for s = 2:length(summands)
        mul!(cores[1], core(summands[s],1), SₖFᴿ[s] * adjoint(Q), 1, 1)
      end
    end
  end

  tt = cores2tensor(cores)
  tt.orthogonal = true
  tt.corePosition = 1
  return tt
end

"""
  roundRandSum2(α::Vector{Number}, summands::Vector{TTvector{T,Nup,Ndn,d}}, max_rank::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function roundRandSum2(α::Vector{T1}, summands::Vector{TTvector{T,Nup,Ndn,d,M}}, rmax::Int, over::Int) where {T1<:Number,T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  return round!(roundRandSum2(α,summands,rmax+over), rmax=rmax)
end