"""
  roundRandSum(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, target_r::Vector{OffsetVector{Int,Vector{Int}}})

Truncates the ranks of `tt` with specified ranks `target_r`.
"""
function roundRandSum(α::Vector{T1}, summands::Vector{TTvector{T,N,d,M}}, target_r::Vector{OffsetVector{Int,Vector{Int}}}) where {T1<:Number,T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck @assert length(α) == length(summands)
  @assert all(target_r[  1][0] == rank(S,  1,0) == 1 for S in summands)
  @assert all(target_r[d+1][N] == rank(S,d+1,N) == 1 for S in summands)

  # Use left-orthogonalized Gaussian or Rademacher randomized cores for framing
  Ω = tt_randn(Val(d),Val(N),target_r,orthogonal=:left)

  # Initialize tensor cores array for the result
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef,d)

  # Precompute partial projections W
  Fᴸ = Matrix{Frame{T,N,d,Matrix{T}}}(undef, length(summands), d)
  for s=1:length(summands)
    Fᴸ[s,1] = IdFrame(Val(N), Val(d), 1)
    for k=1:d-1
      Fᴸ[s,k+1] = (adjoint(core(Ω,k)) * Fᴸ[s,k]) * core(summands[s],k)
    end
  end

  Fᴿ = [ lmul!(α[s], IdFrame(Val(N), Val(d), d+1)) for s in axes(summands,1)]
  for k=d:-1:2
    SₖFᴿ = [ core(summands[s],k) * Fᴿ[s] for s in axes(summands,1)]
    FᴸSₖFᴿ = Fᴸ[1,k] * SₖFᴿ[1]
    for s = 2:length(summands)
      mul!(FᴸSₖFᴿ, Fᴸ[s,k], SₖFᴿ[s], 1, 1)
    end
    Q, = cq!(FᴸSₖFᴿ)
    Fᴿ = [SₖFᴿ[s] * adjoint(Q) for s in axes(summands,1)]
    cores[k] = Q
  end

  cores[1] = core(summands[1],1) * Fᴿ[1]
  for s = 2:length(summands)
    mul!(cores[1], core(summands[s],1), Fᴿ[s], 1, 1)
  end

  tt = cores2tensor(cores)
  tt.orthogonal = true
  tt.corePosition = 1
  return tt
end

"""
  roundRandSum(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, max_rank::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function roundRandSum(α::Vector{T1}, summands::Vector{TTvector{T,N,d,M}}, rmax::Int, over::Int) where {T1<:Number,T<:Number,N,d,M<:AbstractMatrix}
  target_r = [[min(rmax, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  for k=2:d
    target_r[k] .+= over
  end
  # [[(k∈(1,d+1) ? 1 : rmax+over) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  return round!(roundRandSum(α,summands,target_r), rmax=rmax)
end

"""
  roundRandSum2(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, m::Int)

Truncates the ranks of `tt` with specified ranks `target_r` using a randomized projection onto `m` rank-one vectors.
"""
function roundRandSum2(α::Vector{T1}, summands::Vector{TTvector{T,N,d,M}}, m::Int) where {T1<:Number,T<:Number,N,d,M<:AbstractMatrix{T}}
  target_r = [[min(m, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]

  @boundscheck @assert length(α) == length(summands)
  @assert all(target_r[  1][0] == rank(S,  1,0) == 1 for S in summands)
  @assert all(target_r[d+1][N] == rank(S,d+1,N) == 1 for S in summands)

  # Initialize tensor cores for the result
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef,d)

  # Precompute partial projections W
  Fᴸ = Matrix{Frame{T,N,d,Matrix{T}}}(undef, length(summands), d)
  for k=1:d-1
    Ωₖ = randd(N,d,k,m)
    for s=1:length(summands)
      k == 1 && ( Fᴸ[s,1] = IdFrame(Val(N), Val(d), 1) )
      # Use randomized diagonal cores (i.e. block-TT representation of m rank-one vectors) for projection
      Fᴸ[s,k+1] = (adjoint(Ωₖ) * Fᴸ[s,k]) * core(summands[s],k)
    end
  end

  Fᴿ = [ lmul!(α[s], IdFrame(Val(N), Val(d), d+1)) for s in axes(summands,1)]
  for k=d:-1:2
    SₖFᴿ = [ core(summands[s],k) * Fᴿ[s] for s in axes(summands,1)]
    FᴸSₖFᴿ = Fᴸ[1,k] * SₖFᴿ[1]
    for s = 2:length(summands)
      mul!(FᴸSₖFᴿ, Fᴸ[s,k], SₖFᴿ[s], 1, 1)
    end
    Q, = cq!(FᴸSₖFᴿ)
    Fᴿ = [SₖFᴿ[s] * adjoint(Q) for s in axes(summands,1)]
    cores[k] = Q
  end

  cores[1] = core(summands[1],1) * Fᴿ[1]
  for s = 2:length(summands)
    mul!(cores[1], core(summands[s],1), Fᴿ[s], 1, 1)
  end

  tt = cores2tensor(cores)
  tt.orthogonal = true
  tt.corePosition = 1
  return tt
end

"""
  roundRandSum2(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, max_rank::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function roundRandSum2(α::Vector{T1}, summands::Vector{TTvector{T,N,d,M}}, rmax::Int, over::Int) where {T1<:Number,T<:Number,N,d,M<:AbstractMatrix{T}}
  return round!(roundRandSum2(α,summands,rmax+over), rmax=rmax)
end