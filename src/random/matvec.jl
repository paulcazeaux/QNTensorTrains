"""
  randRound_H_MatVec(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}, target_r::Vector{OffsetVector{Int,Vector{Int}}})

Combines the matrix-free application of two-body Hamiltonian operators, implementing the action on a given TT `core`,
and the randomized sketching and truncation of the result as a TT-vector with reduced ranks without forming the intermediate object.

The Hamiltonian is given in terms of one-electron integrals `t_{ij}` and two-electron integrals `v_{ijkl}`.
Truncates the ranks of the matrix-vector product with `x` with specified ranks `target_r`.
"""
function randRound_H_MatVec(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}, target_r::Vector{OffsetVector{Int,Vector{Int}}}) where {T<:Number,N,d}
  @assert target_r[  1][0] == rank(x,  1,0) == 1
  @assert target_r[d+1][N] == rank(x,d+1,N) == 1

  # Initialize tensor cores for the result
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef,d)

  # Use left-orthogonalized Gaussian or Rademacher randomized cores for framing
  Ω = tt_randn(Val(d),Val(N),target_r,orthogonal=:left)
  # Ω = tt_rand((-1,1), Val(d),Val(N),target_r,orthogonal=:left)
  # display(Ω.orthogonal)

  # Precompute partial projections W
  Fᴸ = Vector{Frame{T,N,d,Matrix{T}}}(undef, d)
  Fᴸ[1] = IdFrame(Val(d), Val(N), 1)
  for k=1:d-1
    Fᴸ[k+1] = adjoint(core(Ω,k)) * (Fᴸ[k] * H * core(x, k))
  end

  HXₖFᴿ = H * core(x, d) * IdFrame(Val(d), Val(N), d+1)
  for k=d:-1:2
    Q, = cq!(Fᴸ[k] * HXₖFᴿ)
    cores[k] = Q
    HXₖFᴿ = H * core(x, k-1) * (HXₖFᴿ * adjoint(Q))
  end
  cores[1] = HXₖFᴿ

  tt = cores2tensor(cores)

  tt.orthogonal = true
  tt.corePosition = 1
  return tt
end

"""
  randRound_H_MatVec(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}, rmax::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function randRound_H_MatVec(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}, rmax::Int, over::Int) where {T<:Number,N,d}
  target_r = [[min(rmax+over, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  return round_global!(randRound_H_MatVec(H, x,target_r), rmax=rmax)
end

"""
  randRound_H_MatVecAdd(α::Vector{T}, H::SparseHamiltonian{T,N,d}, x::Vector{TTvector{T,N,d}}, target_r::Vector{OffsetVector{Int,Vector{Int}}})

  Given TTvectors `x=[x[1], x[2], ... x[M]]` and coefficients `α` compute the linear combination
    α[1]⋅Hx[1] + α[2]⋅x[2] + … + α[M]⋅x[M]
  using randomized sketches.
"""
function randRound_H_MatVecAdd( α::Vector{T}, H::SparseHamiltonian{T,N,d}, summands::Vector{TTvector{T,N,d,S}},
                               target_r::Vector{OffsetVector{Int,Vector{Int}}}, to::TimerOutput) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  @assert all(target_r[  1][0] == rank(x,  1,0) == 1 for x in summands)
  @assert all(target_r[d+1][N] == rank(x,d+1,N) == 1 for x in summands)

  # Use left-orthogonalized Gaussian or Rademacher randomized cores for framing
  Ω = tt_randn(Val(d),Val(N),target_r,orthogonal=:left)
  # Ω = tt_rand((-1,1), Val(d),Val(N),target_r,orthogonal=:left)
  # display(Ω.orthogonal)

  # Initialize tensor cores array for the result
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef,d)

@timeit to "Framing" begin
  # Precompute partial projections W
  Fᴸ = Matrix{Frame{T,N,d,Matrix{T}}}(undef, length(summands), d)

  Fᴸ[1,1] = IdFrame(Val(d), Val(N), 1)
  for k=1:d-1
    Fᴸ[1,k+1] = adjoint(core(Ω,k)) * (Fᴸ[1,k] * H * core(summands[1], k))
  end
  for s=2:length(summands)
    Fᴸ[s,1] = IdFrame(Val(d), Val(N), 1)
    for k=1:d-1
      Fᴸ[s,k+1] = adjoint(core(Ω,k)) * (Fᴸ[s,k] * core(summands[s],k))
    end
  end
end
@timeit to "Core assembly" begin
  SₖFᴿ = [ s == 1 ? H * core(summands[s],d) * lmul!(α[1],IdFrame(Val(d),Val(N),d+1)) : core(summands[s],d) * lmul!(α[s],IdFrame(Val(d),Val(N),d+1)) for s in axes(summands,1)]
  for k=d:-1:2
@timeit to "Frame FᴸSₖFᴿ" begin
    FᴸSₖFᴿ = SparseCore{T,N,d}(k, Fᴸ[1,k].row_ranks, SₖFᴿ[1].col_ranks)
    for s in axes(summands,1)
      mul!(FᴸSₖFᴿ, Fᴸ[s,k], SₖFᴿ[s], 1, 1)
    end
end
@timeit to "CQ factorization" begin
    Q, = cq!(FᴸSₖFᴿ)
end
    cores[k] = Q
@timeit to "Frame SₖFᴿ" begin
    if k>2
      SₖFᴿ = [ s == 1 ? H * core(summands[s],k-1) * (SₖFᴿ[s] * adjoint(Q)) : core(summands[s],k-1) * (SₖFᴿ[s] * adjoint(Q)) for s in axes(summands,1)]
    else
      cores[1] = H * core(summands[1],1) * (SₖFᴿ[1] * adjoint(Q))
      for s = 2:length(summands)
        mul!(cores[1], core(summands[s],1), SₖFᴿ[s] * adjoint(Q), 1, 1)
      end
    end
end
  end
end

  tt = cores2tensor(cores)

  tt.orthogonal = true
  tt.corePosition = 1

  return tt
end


function randRound_H_MatVecAdd(α::Vector{T}, H::SparseHamiltonian{T,N,d}, summands::Vector{TTvector{T,N,d,S}}, rmax::Int, over::Int, to::TimerOutput) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  target_r = [[min(rmax+over, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  return randRound_H_MatVecAdd(α,H,summands,target_r,to)
end

"""
  randRound_H_MatVec2(H::SparseHamiltonian{T,N,d}, tt_in::TTvector{T,N,d}, m::Int)

Combines the matrix-free application of two-body Hamiltonian operators, implementing the action on a given TT `core`,
and the randomized sketching and truncation of the result as a TT-vector with reduced ranks without forming the intermediate object.

The Hamiltonian is given in terms of one-electron integrals `t_{ij}` and two-electron integrals `v_{ijkl}`.
Truncates the ranks of the matrix-vector product with `tt_in` using randomized projection onto `m` rank-one vectors.
"""
function randRound_H_MatVec2(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}, m::Int) where {T<:Number,N,d}
  target_r = [[min(m, binomial(k-1,n), binomial(d+1-k,N-n)) for n in occupation_qn(N,d,k)] for k=1:d+1]
  @assert target_r[  1][0] == rank(x,  1,0) == 1
  @assert target_r[d+1][N] == rank(x,d+1,N) == 1

  # Initialize tensor cores for the result
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef,d)

  # Use randomized diagonal cores (i.e. block-TT representation of m rank-one vectors) 

  # Precompute partial projections W
  Fᴸ = Vector{Frame{T,N,d,Matrix{T}}}(undef, d)
  Fᴸ[2] = adjoint(randh(Val(d),Val(N),m)) * H * core(x, 1)
  for k=2:d-1
    Fᴸ[k+1] = adjoint(randd(Val(d),Val(N),k,m)) * (Fᴸ[k] * H * core(x, k))
  end

  HXₖFᴿ = H * core(x, d) * IdFrame(Val(d), Val(N), d+1)
  for k=d:-1:2
    Q, = cq!(Fᴸ[k] * HXₖFᴿ)
    cores[k] = Q
    HXₖFᴿ = H * core(x, k-1) * (HXₖFᴿ * adjoint(Q))
  end
  cores[1] = HXₖFᴿ

  tt = cores2tensor(cores)

  tt.orthogonal = true
  tt.corePosition = 1
  return tt
end

"""
  randRound_H_MatVec2(H::SparseHamiltonian{T,N,d}, tt_in::TTvector{T,N,d}, rmax::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function randRound_H_MatVec2(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}, rmax::Int, over::Int) where {T<:Number,N,d}
  return round_global!(randRound_H_MatVec2(H, x,rmax+over), rmax=rmax)
end

"""
  randRound_H_MatVecAdd2(α::Vector{T}, H::SparseHamiltonian{T,N,d}, x::Vector{TTvector{T,N,d}}, m::Int)

  Given TTvectors `x=[x[1], x[2], ... x[M]]` and coefficients `α` compute the linear combination
    α[1]⋅Hx[1] + α[2]⋅x[2] + … + α[M]⋅x[M]
  using randomized sketches onto `m` rank-one vectors.
"""
function randRound_H_MatVecAdd2( α::Vector{T}, H::SparseHamiltonian{T,N,d}, summands::Vector{TTvector{T,N,d,S}},
                               m::Int, to::TimerOutput) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  target_r = [[min(m, binomial(k-1,n), binomial(d+1-k,N-n)) for n in occupation_qn(N,d,k)] for k=1:d+1]

  @assert all(target_r[  1][0] == rank(x,  1,0) == 1 for x in summands)
  @assert all(target_r[d+1][N] == rank(x,d+1,N) == 1 for x in summands)

  # Use randomized diagonal cores (i.e. block-TT representation of m rank-one vectors) 

  # Initialize tensor cores array for the result
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef,d)

@timeit to "Framing" begin
  # Precompute partial projections W
  Fᴸ = Matrix{Frame{T,N,d,Matrix{T}}}(undef, length(summands), d)

  Ω₁ = randh(Val(d),Val(N),m)
  Fᴸ[1,2] = adjoint(Ω₁) * H * core(summands[1], 1)
  for s=2:length(summands)
    Fᴸ[s,2] = adjoint(Ω₁) * core(summands[s],1)
  end
  for k=2:d-1
    Ωₖ = randd(Val(d),Val(N),k,m)
    Fᴸ[1,k+1] = adjoint(Ωₖ) * (Fᴸ[1,k] * H * core(summands[1], k))
    for s=2:length(summands)
      Fᴸ[s,k+1] = adjoint(Ωₖ) * (Fᴸ[s,k] * core(summands[s],k))
    end
  end
end
@timeit to "Core assembly" begin
  SₖFᴿ = [ s == 1 ? 
     H * core(summands[1],d) * lmul!(α[1],IdFrame(Val(d),Val(N),d+1)) : 
         core(summands[s],d) * lmul!(α[s],IdFrame(Val(d),Val(N),d+1))
        for s in axes(summands,1)]

  for k=d:-1:2
@timeit to "Frame FᴸSₖFᴿ" begin
    FᴸSₖFᴿ = SparseCore{T,N,d}(k, Fᴸ[1,k].row_ranks, SₖFᴿ[1].col_ranks)
    for s in axes(summands,1)
      mul!(FᴸSₖFᴿ, Fᴸ[s,k], SₖFᴿ[s], 1, 1)
    end
end
@timeit to "CQ factorization" begin
    Q, = cq!(FᴸSₖFᴿ)
end
    cores[k] = Q
@timeit to "Frame SₖFᴿ" begin
    if k>2
      SₖFᴿ = [ s == 1 ? 
     H * core(summands[1],k-1) * (SₖFᴿ[1] * adjoint(Q)) : 
         core(summands[s],k-1) * (SₖFᴿ[s] * adjoint(Q))
            for s in axes(summands,1)]
    else
      cores[1] = H * core(summands[1],1) * (SₖFᴿ[1] * adjoint(Q))
      for s = 2:length(summands)
        mul!(cores[1], core(summands[s],1), (SₖFᴿ[s] * adjoint(Q)), 1, 1)
      end
    end
end
  end
end
  tt = cores2tensor(cores)

  tt.orthogonal = true
  tt.corePosition = 1

  return tt
end


function randRound_H_MatVecAdd2(α::Vector{T}, H::SparseHamiltonian{T,N,d}, summands::Vector{TTvector{T,N,d,S}}, rmax::Int, over::Int) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  return round_global!(randRound_H_MatVecAdd2(α,H,summands,rmax+over), rmax=rmax)
end

