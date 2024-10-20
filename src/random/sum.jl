"""
  roundRandSum(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, target_r::Vector{OffsetVector{Int,Vector{Int}}})

Truncates the ranks of `tt` with specified ranks `target_r`.
"""
function roundRandSum(α::Vector{T1}, summands::Vector{TTvector{T,N,d}}, target_r::Vector{OffsetVector{Int,Vector{Int}}}) where {T1<:Number,T<:Number,N,d}
  @boundscheck @assert length(α) == length(summands)
  @assert all(target_r[  1][0] == rank(S,  1,0) == 1 for S in summands)
  @assert all(target_r[d+1][N] == rank(S,d+1,N) == 1 for S in summands)

  # Initialize tensor cores for the result
  cores = [SparseCore{T,N,d}(k) for k=1:d]

  # Use Rademacher randomized cores for framing
  Ω = tt_rand((-1,1),Val(d),Val(N),target_r,orthogonal=:right)

  # Precompute partial projections W
  W = [ [zeros(T,0,0) for n in axes(cores[k],3), S in summands] for k=1:d]
  for (s,S) in enumerate(summands)
    W[d][N,s] = [T(1);;]

    for k=d-1:-1:1
      Sₖ₊₁ = core(S,k+1)
      ωₖ₊₁ = core(Ω,k+1)
      for l in axes(Sₖ₊₁,1)
        W[k][l,s] = zeros(T,Sₖ₊₁.row_ranks[l],ωₖ₊₁.row_ranks[l])
        for r in axes(Sₖ₊₁,3)∩(l:l+1)
          if isnonzero(Sₖ₊₁,l,r) && isnonzero(ωₖ₊₁,l,r)
            X = data(Sₖ₊₁[l,r]) * W[k+1][r,s] * adjoint(data(ωₖ₊₁[l,r]))
            axpy!(factor(Sₖ₊₁[l,r])*conj(factor(ωₖ₊₁[l,r])), X, W[k][l,s])
          end
        end
      end
    end
  end

  # Start with first core
  # Initialize ranks and array for C matrices for this step
  C = [ zeros(T,0,0) for n in axes(cores[1],3), _ in summands]
  ranks = [ 0 for n in axes(cores[1],3) ]

  for n in axes(cores[1],3)
    # Implicitly contract the block-diagonal TT-sum core into new basis using frame matrices C and W
    Y = core(summands[1],1)[n,:vertical] * W[1][n,1]
    for s in 2:length(summands)
      mul!(Y, core(summands[s],1)[n,:vertical], W[1][n,s], 1, 1)
    end

    Q,_,ranks[n] = my_qc!(Y)
    for (s,S) in enumerate(summands)
      C[n,s] = α[s] * Q' * core(S,1)[n,:vertical]
    end
    cores[1][n,:vertical] = Q
  end

  for k=2:d-1

    for n in axes(cores[k],1)
      cores[k].row_ranks[n] = ranks[n]
    end

    # Contract k-th summand cores with C matrices from previous step
    V = [ C[:,s] * core(S,k) for (s,S) in enumerate(summands)]

    # Initialize ranks and array for C matrices for this step
    C = [ zeros(T,0,0) for n in axes(cores[k],3), _ in summands]
    ranks = [ 0 for n in axes(cores[k],3) ]

    for n in axes(cores[k],3)
      # Implicitly contract the block-diagonal TT-sum core into new basis using frame matrices C and W
      Y = V[1][n,:vertical] * W[k][n,1]
      for s in 2:length(V)
        mul!(Y, V[s][n,:vertical], W[k][n,s], 1, 1)
      end

      Q,_,ranks[n] = my_qc!(Y)
      for s in eachindex(V)
        C[n,s] = Q' * V[s][n,:vertical]
      end
      cores[k][n,:vertical] = Q
    end
  end

  cores[d] = C[:,1]*core(summands[1],d)
  for s=2:length(summands)
    axpy!(1, C[:,s]*core(summands[s],d), cores[d])
  end

  tt = cores2tensor(cores)

  tt.orthogonal = true
  tt.corePosition = d

  return tt
end

"""
  roundRandSum(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, max_rank::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function roundRandSum(α::Vector{T1}, summands::Vector{TTvector{T,N,d}}, rmax::Int, over::Int) where {T1<:Number,T<:Number,N,d}
  target_r = [[min(rmax, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  for k=2:d
    target_r[k] .+= over
  end
  # [[(k∈(1,d+1) ? 1 : rmax+over) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  return round!(roundRandSum(α,summands,target_r), rmax=rmax)
end

"""
  roundRandSum2(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, m::Int)

Truncates the ranks of `tt` with specified ranks `target_r` using a randomized projection onto `m` vectors.
"""
function roundRandSum2(α::Vector{T1}, summands::Vector{TTvector{T,N,d}}, m::Int) where {T1<:Number,T<:Number,N,d}
  target_r = [[min(m, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]

  @boundscheck @assert length(α) == length(summands)
  @assert all(target_r[  1][0] == rank(S,  1,0) == 1 for S in summands)
  @assert all(target_r[d+1][N] == rank(S,d+1,N) == 1 for S in summands)

  # Initialize tensor cores for the result
  cores = [SparseCore{T,N,d}(k) for k=1:d]

  Ω = randn(T,d-1,2,m)

  # Precompute partial projections W
  W = [ [zeros(T,0,0) for n in axes(cores[k],3), S in summands] for k=1:d]
  for (s,S) in enumerate(summands)
    W[d][N,s] = ones(T,1,m)

    for k=d-1:-1:1
      Sₖ₊₁ = core(S,k+1)
      for l in axes(Sₖ₊₁,1)
        W[k][l,s] = zeros(T,Sₖ₊₁.row_ranks[l],m)
        for r in axes(Sₖ₊₁,3)∩(l:l+1)
          if isnonzero(Sₖ₊₁,l,r)
            X = data(Sₖ₊₁[l,r]) * W[k+1][r,s] 
            for i=1:m
              X[:,i] .*= Ω[k,r+1-l,i]
            end
            axpy!(factor(Sₖ₊₁[l,r]), X, W[k][l,s])
          end
        end
      end

      # Truncate previous projections as needed
      for n in axes(W[k+1],1)
        W[k+1][n,s] = W[k+1][n,s][:,1:target_r[k+2][n]]
      end
    end
  end

  # Start with first core
  # Initialize ranks and array for C matrices for this step
  C = [ zeros(T,0,0) for n in axes(cores[1],3), _ in summands]
  ranks = [ 0 for n in axes(cores[1],3) ]

  for n in axes(cores[1],3)
    # Implicitly contract the block-diagonal TT-sum core into new basis using frame matrices C and W
    Y = core(summands[1],1)[n,:vertical] * W[1][n,1]
    for s in 2:length(summands)
      mul!(Y, core(summands[s],1)[n,:vertical], W[1][n,s], 1, 1)
    end

    Q,_,ranks[n] = my_qc!(Y)
    for (s,S) in enumerate(summands)
      C[n,s] = α[s] * Q' * core(S,1)[n,:vertical]
    end
    cores[1][n,:vertical] = Q
  end

  for k=2:d-1

    for n in axes(cores[k],1)
      cores[k].row_ranks[n] = ranks[n]
    end

    # Contract k-th summand cores with C matrices from previous step
    V = [ C[:,s] * core(S,k) for (s,S) in enumerate(summands)]

    # Initialize ranks and array for C matrices for this step
    C = [ zeros(T,0,0) for n in axes(cores[k],3), _ in summands]
    ranks = [ 0 for n in axes(cores[k],3) ]

    for n in axes(cores[k],3)
      # Implicitly contract the block-diagonal TT-sum core into new basis using frame matrices C and W
      Y = V[1][n,:vertical] * W[k][n,1]
      for s in 2:length(V)
        mul!(Y, V[s][n,:vertical], W[k][n,s], 1, 1)
      end

      Q,_,ranks[n] = my_qc!(Y)
      for s in eachindex(V)
        C[n,s] = Q' * V[s][n,:vertical]
      end
      cores[k][n,:vertical] = Q
    end
  end

  cores[d] = C[:,1]*core(summands[1],d)
  for s=2:length(summands)
    axpy!(1, C[:,s]*core(summands[s],d), cores[d])
  end

  tt = cores2tensor(cores)

  tt.orthogonal = true
  tt.corePosition = d

  return tt
end

"""
  roundRandSum2(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, max_rank::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function roundRandSum2(α::Vector{T1}, summands::Vector{TTvector{T,N,d}}, rmax::Int, over::Int) where {T1<:Number,T<:Number,N,d}
  return round!(roundRandSum2(α,summands,rmax+over), rmax=rmax)
end