"""
  roundRandSum(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, target_r::Vector{OffsetVector{Int,Vector{Int}}})

Truncates the ranks of `tt` with specified ranks `target_r`.
"""
function randRound_H_MatVec(tt_in::TTvector{T,N,d}, t::Matrix{T}, v::Array{T,4}, target_r::Vector{OffsetVector{Int,Vector{Int}}}) where {T<:Number,N,d}
  @assert target_r[  1][0] == rank(tt_in,  1,0) == 1
  @assert target_r[d+1][N] == rank(tt_in,d+1,N) == 1

  # Compute sparse cores for the Hamiltonian matvec
  HX = [sparse_H_matvec(core(tt_in,k), t, v) for k=1:d]

  # Initialize tensor cores for the result
  cores = [SparseCore{T,N,d}(k) for k=1:d]

  # Use Rademacher randomized cores for framing
  Ω = tt_rand((-1,1), d,N,target_r)

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
  roundRandSum(α::Vector{Number}, summands::Vector{TTvector{T,N,d}}, max_rank::Int, over::Int)

Truncates the ranks of `tt` with maximal ranks (or bond dimension) `rmax` and oversampling `over`.
"""
function randRound_H_MatVec(tt_in::TTvector{T,N,d}, t::Matrix{T}, v::Array{T,4}, rmax::Int, over::Int) where {T<:Number,N,d}
  target_r = [[(k∈(1,d+1) ? 1 : rmax+over) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  return round_global!(randRound_H_MatVec(tt_in,t,v,target_r), rmax=rmax)
end