"""
    roundRandOrth!(tt::TTvector{T,N,d}, target_r::Vector{OffsetVector{Int,Vector{Int}}}, over::Int)

Truncates the ranks of `tt` with specified ranks `target_r`.
Value of `over` represents oversampling
"""
function roundRandOrth!(tt::TTvector{T,N,d}, target_r::Vector{OffsetVector{Int,Vector{Int}}}, over::Int) where {T<:Number,N,d}

  @assert target_r[1][0] == rank(tt,1,0)
  @assert target_r[d+1][N] == rank(tt,d+1,N)
  r = rank(tt)

  # Use Rademacher cores and right-orthogonalized tt_rand
  rΩ = [target_r[k] .+ (1<k≤d ? over : 0) for k=1:d+1]
  Ω = tt_rand((-1,1), Val(d),Val(N), rΩ, orthogonal=:right)

  # Precompute partial projections W
  W = [[zeros(T,0,0) for n in axes(core(tt,k),3)] for k=1:d]
  W[d][N] = [T(1);;]

  for k=d-1:-1:1
    Xₖ₊₁ = core(tt,k+1) 
    ωₖ₊₁ = core(Ω,k+1)
    for l in axes(Xₖ₊₁,1)
      W[k][l] = zeros(T,rank(tt,k+1,l),rank(Ω,k+1,l))
      for r in axes(Xₖ₊₁,3)∩(l:l+1)
        if isnonzero(Xₖ₊₁,l,r) && isnonzero(ωₖ₊₁,l,r)
          X = data(Xₖ₊₁[l,r]) * W[k+1][r] * adjoint(data(ωₖ₊₁[l,r]))
          axpy!(factor(Xₖ₊₁[l,r])*conj(factor(ωₖ₊₁[l,r])), X, W[k][l])
        end
      end
    end
  end

  for k=1:d-1
    Xₖ = core(tt,k) 
    C = [zeros(T,0,0) for n in axes(Xₖ,3)]
    rank = [T(0) for n in axes(Xₖ,3)]

    for n in axes(Xₖ,3)
      Q,_,rank[n] = my_qc!(Xₖ[n,:vertical] * W[k][n])
      C[n] = Q' * Xₖ[n,:vertical]
      core(tt,k)[n,:vertical] = Q
    end
    lmul!(C, core(tt,k+1))
    tt.r[k+1] .= rank
  end

  tt.orthogonal = true
  tt.corePosition = d

  return tt
end

function roundRandOrth(tt::TTvector, target_r::Vector{OffsetVector{Int,Vector{Int}}}, over::Int)
  return roundRandOrth!(deepcopy(tt), target_r, over::Int)
end

"""
    roundRandOrth2!(tt::TTvector{T,N,d}, m::Int)

Truncates the ranks of `tt` with specified number `m=rmax+over` of random rank-one tensors, 
where `rmax` should be the maximum target rank and `over` an oversampling parameter.
"""
function roundRandOrth2!(tt::TTvector{T,N,d}, m::Int) where {T<:Number,N,d}
  target_r = [[min(m, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]

  # Precompute partial projections W
  W = [[zeros(T,0,0) for n in axes(core(tt,k),3)] for k=1:d]
  W[d][N] = ones(T,1,m)

  Ωₖ = zeros(T,2,m)

  for k=d-1:-1:1
    Xₖ₊₁ = core(tt,k+1)
    randn!(Ωₖ)
    for l in axes(Xₖ₊₁,1)
      W[k][l] = zeros(T,rank(tt,k+1,l),m)
      for r in axes(Xₖ₊₁,3)∩(l:l+1)
        if isnonzero(Xₖ₊₁,l,r)
          X = data(Xₖ₊₁[l,r]) * W[k+1][r]
          for i=1:m
            X[:,i] .*= Ωₖ[r+1-l,i]
          end
          axpy!(factor(Xₖ₊₁[l,r]), X, W[k][l])
        end
      end
    end

    # Truncate unnecessary projection components.
    for n in axes(W[k+1],1)
      W[k+1][n] = W[k+1][n][:,1:target_r[k+2][n]]
    end
  end

  for k=1:d-1
    Xₖ = core(tt,k) 
    C = [zeros(T,0,0) for n in axes(Xₖ,3)]
    rank = [T(0) for n in axes(Xₖ,3)]

    for n in axes(Xₖ,3)
      Q,_,rank[n] = my_qc!(Xₖ[n,:vertical] * W[k][n])
      C[n] = Q' * Xₖ[n,:vertical]
      core(tt,k)[n,:vertical] = Q
    end
    lmul!(C, core(tt,k+1))
    tt.r[k+1] .= rank
  end

  tt.orthogonal = true
  tt.corePosition = d

  return tt
end

function roundRandOrth2(tt::TTvector, rmax::Int, over::Int)
  return round!(roundRandOrth2!(deepcopy(tt),rmax+over), rmax=rmax)
end