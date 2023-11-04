"""
  randRound_H_MatVec(tt_in::TTvector{T,N,d}, t::Matrix{T}, v::Array{T,4}, target_r::Vector{OffsetVector{Int,Vector{Int}}})

Combines the matrix-free application of two-body Hamiltonian operators, implementing the action on a given TT `core`,
and the randomized sketching and truncation of the result as a TT-vector with reduced ranks without forming the intermediate object.

The Hamiltonian is given in terms of one-electron integrals `t_{ij}` and two-electron integrals `v_{ijkl}`.
Truncates the ranks of the matrix-vector product with `tt_in` with specified ranks `target_r`.
"""
function randRound_H_MatVec(tt_in::TTvector{T,N,d}, t::Matrix{T}, v::Array{T,4}, target_r::Vector{OffsetVector{Int,Vector{Int}}}) where {T<:Number,N,d}
  @assert target_r[  1][0] == rank(tt_in,  1,0) == 1
  @assert target_r[d+1][N] == rank(tt_in,d+1,N) == 1

  # Compute sparse cores for the Hamiltonian matvec
  HX = [sparse_H_matvec(core(tt_in,k), t, v) for k=1:d]

  # Initialize tensor cores for the result
  cores = [SparseCore{T,N,d}(k) for k=1:d]

  # Use Rademacher randomized cores for framing
  Ω = tt_rand((-1,1), Val(d),Val(N),target_r)

  # Precompute partial projections W
  W = Vector{OffsetVector{Matrix{T},Vector{Matrix{T}}}}(undef, d)
  W[d] = OffsetVector([n == N ? ones(T,1,1) : zeros(T,0,0) for n in 0:N], 0:N)

  for k=d-1:-1:2
    W[k] = OffsetVector([Matrix{T}(undef,0,0) for l in 0:N], 0:N)

    ωₖ₊₁ = core(Ω, k+1)
    HXₖ₊₁ = HX[k+1]

    for l in axes(ωₖ₊₁,1)
      W[k][l] = zeros(T, HXₖ₊₁[1][l], ωₖ₊₁.row_ranks[l])
      for r in axes(ωₖ₊₁,3)∩(l:l+1)
        if isnonzero(ωₖ₊₁,l,r)
          for (I,J,V) in zip(HXₖ₊₁[3],HXₖ₊₁[4],HXₖ₊₁[5])
            if isnonzero(V,l,r)
              mul!( view(W[k][l], I[l], :),
                    data(V[l,r]), 
                    W[k+1][r][J[r],:] * adjoint(data(ωₖ₊₁[l,r])),
                    factor(V[l,r]) * conj(factor(ωₖ₊₁[l,r])),
                    T(1)
                  )
            end
          end
        end
      end
    end
  end

  C = OffsetVector(Vector{Matrix{T}}(undef,N+1), 0:N)
  # Start with assembling first core.
  # No framing necessary here - ranks are going to 1.
  cores[1].row_ranks[0] = 1

  for r in axes(cores[1],3)
    if HX[1][2][r] > 0
      cores[1].col_ranks[r] = 1
      cores[1][0,r] = ones(T, 1, 1)
      C[r] = zeros(T, 1, HX[1][2][r])
      for (I,J,V) in zip(HX[1][3],HX[1][4],HX[1][5])
        if isnonzero(V,0,r)
          C[r][I[0], J[r]] = V[0,r]
        end
      end
    else
      cores[1].col_ranks[r] = 0
      cores[1][0,r] = zeros_block(T,1,0)
      C[r] = zeros(T,0,0)
    end
  end

  for k=2:d-1
    cores[k].row_ranks .= cores[k-1].col_ranks

   # Reverse because we use C[r-1] and C[r] and overwrite C[r] in the loop body
    for r in reverse(axes(cores[k],3))
      # Compute vertical unfoldings for the k-th block-sparse matvec core, 
      # contracted matrix-free style into new basis using left frame matrix C
      ql = axes(cores[k],1)∩(r-1:r)
      Y = OffsetVector([zeros(T, cores[k].row_ranks[l], HX[k][2][r]) for l in ql], ql)
      for (I,J,V) in zip(HX[k][3],HX[k][4],HX[k][5])
        for l in ql
          if isnonzero(V,l,r)
            mul!(view(Y[l],:,J[r]), view(C[l],:,I[l]), data(V[l,r]), factor(V[l,r]), 1)
          end
        end
      end
      CHX = vcat(Y...)

      # Randomized QC decomposition
      Q, = my_qc!(CHX*W[k][r])
      C[r] = adjoint(Q)*CHX
      cores[k][r,:vertical] = Q
    end
  end

  # Assemble last core by directly contracting with C matrices
  cores[d].row_ranks .= cores[d-1].col_ranks
  for l in axes(cores[d],1)
    X = zeros(T,cores[d].row_ranks[l],1)
    for (I,J,V) in zip(HX[d][3],HX[d][4],HX[d][5])
      if isnonzero(V,l,N)
        mul!(view(X,:,J[N]), view(C[l],:,I[l]), data(V[l,N]), factor(V[l,N]), 1)
      end
    end
    cores[d][l,N] = X
  end

  tt = cores2tensor(cores)

  tt.orthogonal = true
  tt.corePosition = d

  return tt
end

"""
  randRound_H_MatVec(tt_in::TTvector{T,N,d}, t::Matrix{T}, v::Array{T,4}, rmax::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function randRound_H_MatVec(tt_in::TTvector{T,N,d}, t::Matrix{T}, v::Array{T,4}, rmax::Int, over::Int) where {T<:Number,N,d}
  target_r = [[min(rmax+over, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  return round_global!(randRound_H_MatVec(tt_in,t,v,target_r), rmax=rmax)
end


"""
  randRound_H_MatVecAdd(α::Vector{T}, x::Vector{TTvector{T,N,d}}, t::Matrix{T}, v::Array{T,4}, target_r::Vector{OffsetVector{Int,Vector{Int}}})

  Given TTvectors `x=[x[1], x[2], ... x[M]]` and coefficients `α` compute the linear combination
    α[1]⋅Hx[1] + α[2]⋅x[2] + … + α[M]⋅x[M]
  using randomized sketches.
"""
function randRound_H_MatVecAdd( α::Vector{T}, x::Vector{TTvector{T,N,d}},
                               t::Matrix{T}, v::Array{T,4}, 
                               target_r::Vector{OffsetVector{Int,Vector{Int}}}) where {T<:Number,N,d}
  @assert all(target_r[  1][0] == rank(xs,  1,0) == 1 for xs in x)
  @assert all(target_r[d+1][N] == rank(xs,d+1,N) == 1 for xs in x)

  # Compute sparse cores for the Hamiltonian matvec
  HX = [sparse_H_matvec(core(x[1],k), t, v) for k=1:d]

  # Initialize tensor cores for the result
  cores = [SparseCore{T,N,d}(k) for k=1:d]

  # Use Rademacher randomized cores for framing
  Ω = tt_rand((-1,1), Val(d),Val(N),target_r)

  # Precompute partial projections W
  W = Vector{OffsetMatrix{Matrix{T},Matrix{Matrix{T}}}}(undef, d)
  for k=1:d
    w = Matrix{Matrix{T}}(undef, size(cores[k],3), length(x))
    W[k] = OffsetMatrix(w, axes(cores[k],3), axes(x,1))
  end

  # First summand: we accumulate for Hx[1]
  W[d][N,1] = [T(1);;]
  for k=d-1:-1:2
    HXₖ₊₁ = HX[k+1]
    ωₖ₊₁  = core(Ω, k+1)

    for l in axes(ωₖ₊₁,1)
      W[k][l,1] = zeros(T, HXₖ₊₁[1][l], ωₖ₊₁.row_ranks[l])
      for r in axes(ωₖ₊₁,3)∩(l:l+1)
        vW = zeros(T, HXₖ₊₁[1][l], ωₖ₊₁.col_ranks[r])
        for (I,J,v) in zip(HXₖ₊₁[3],HXₖ₊₁[4],HXₖ₊₁[5])
          if isnonzero(v,l,r)
            mul!( view(vW, I[l], :), data(v[l,r]), view(W[k+1][r,1],J[r],:),
                  factor(v[l,r]), T(1)
                )
          end
        end
        mul!(W[k][l,1], vW, adjoint(data(ωₖ₊₁[l,r])), conj(factor(ωₖ₊₁[l,r])), T(1))
      end
    end
  end
  for s=2:length(x)
    S = x[s]
    W[d][N,s] = [T(1);;]

    for k=d-1:-1:1
      Sₖ₊₁ = core(S,k+1)
      ωₖ₊₁ = core(Ω,k+1)
      for l in axes(Sₖ₊₁,1)
        W[k][l,s] = zeros(T,Sₖ₊₁.row_ranks[l],ωₖ₊₁.row_ranks[l])
        for r in axes(Sₖ₊₁,3)∩(l:l+1)
          if isnonzero(Sₖ₊₁,l,r)
            X = data(Sₖ₊₁[l,r]) * W[k+1][r,s] * adjoint(data(ωₖ₊₁[l,r]))
            axpy!(factor(Sₖ₊₁[l,r])*conj(factor(ωₖ₊₁[l,r])), X, W[k][l,s])
          end
        end
      end
    end
  end

  C = OffsetMatrix(Matrix{Matrix{T}}(undef,N+1,length(x)), 0:N, 1:length(x))

  # Start with assembling first core.
  # No framing necessary here - ranks are going to 1 or 0.
  for n in axes(cores[1],3)
    cores[1].row_ranks[0] = 1
    if HX[1][2][n] > 0 || any(rank(x[s],2)[n]>0 for s=2:length(x))
      cores[1].col_ranks[n] = 1
      cores[1][0,n] = ones(T, 1, 1)
    else
      cores[1].col_ranks[n] = 0
      cores[1][0,n] = zeros_block(T,1,0)
    end

    rn = cores[1].col_ranks[n]
    ### First summand - sparse Hx[1] representation
    if HX[1][2][n] > 0
      C[n,1] = zeros(T, 1, HX[1][2][n])
      for (I,J,V) in zip(HX[1][3],HX[1][4],HX[1][5])
        if isnonzero(V,0,n)
          C[n,1][I[0], J[n]] = V[0,n]
        end
      end
      C[n,1] .*= α[1]
    else
      C[n,1] = zeros(T,rn,0)
    end
    ### Other summands
    for s in 2:length(x)
      C[n,s] = α[s] * core(x[s],1)[0,n]
    end
  end

  for k=2:d-1
    for n in axes(cores[k],1)
      cores[k].row_ranks[n] = cores[k-1].col_ranks[n]
    end

    V = Vector{Matrix{T}}(undef, length(x))

   # Reverse because we use C[r-1,1] and C[r,1] and overwrite C[r,1] in the loop body
    for r in reverse(axes(cores[k],3))
      # Compute vertical unfoldings for the k-th block-sparse matvec core, 
      # contracted matrix-free style into new basis using left frame matrices C and right frame W
      ql = axes(cores[k],1)∩(r-1:r)
      Y = OffsetVector(Vector{Matrix{T}}(undef, length(ql)), ql)
      for l in ql
        Y[l] = zeros(T, cores[k].row_ranks[l], HX[k][2][r])
      end
      for (I,J,v) in zip(HX[k][3],HX[k][4],HX[k][5])
        for l in ql
          if isnonzero(v,l,r)
            mul!(view(Y[l],:,J[r]), view(C[l,1],:,I[l]), data(v[l,r]), factor(v[l,r]), 1)
          end
        end
      end
      if length(ql) > 1
        V[1] = vcat(Y[ql[1]],Y[ql[2]])
      else
        V[1] = Y[ql[1]]
      end
      # Remaining summands framed cores
      if length(ql) > 1
        for s=2:length(x)
          V[s] = vcat(C[ql[1],s]*core(x[s],k)[ql[1],r],
                      C[ql[2],s]*core(x[s],k)[ql[2],r])
        end
      else
        for s=2:length(x)
          V[s] = C[ql[1],s]*core(x[s],k)[ql[1],r]
        end
      end
      Y = V[1] * W[k][r,1]
      for s in 2:length(V)
        mul!(Y, V[s], W[k][r,s], 1, 1)
      end
      # Randomized QC decomposition
      Q, = my_qc!(Y)

      for s in eachindex(V)
        C[r,s] = adjoint(Q)*V[s]
      end
      cores[k][r,:vertical] = Q
    end
  end

  # Assemble last core - starting with 
  cores[d].row_ranks .= cores[d-1].col_ranks
  for l in axes(cores[d],1)
    X = zeros(T,cores[d].row_ranks[l],1)
    for (I,J,v) in zip(HX[d][3],HX[d][4],HX[d][5])
      if isnonzero(v,l,N)
        mul!(view(X,:,J[N]), view(C[l,1],:,I[l]), data(v[l,N]), factor(v[l,N]), 1)
      end
    end
    cores[d][l,N] = X
  end
  for s=2:length(x)
    axpy!(1, C[axes(cores[d],1),s]*core(x[s],d), cores[d])
  end
  tt = cores2tensor(cores)

  tt.orthogonal = true
  tt.corePosition = d

  return tt
end


function randRound_H_MatVecAdd(α::Vector{T}, tts::Vector{TTvector{T,N,d}}, t::Matrix{T}, v::Array{T,4}, rmax::Int, over::Int) where {T<:Number,N,d}
  target_r = [[min(rmax, binomial(k-1,n), binomial(d+1-k,N-n)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  for k=2:d
    target_r[k] .+= over
  end
             # [[(k∈(1,d+1) ? 1 : rmax+over) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  return randRound_H_MatVecAdd(α,tts,t,v,target_r)
end