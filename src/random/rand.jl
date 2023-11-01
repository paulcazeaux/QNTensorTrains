"""
  A = randn(N::Int, d::Int, k::Int,
                    row_ranks::OffsetVector{Int, Vector{Int}}, 
                    col_ranks::OffsetVector{Int, Vector{Int}},
                    [T=Float64])

  Compute a core with randomized entries within the sparse block structure allowed
  for the `k`-th core of a `d`-dimensional TT-tensor with `N` total occupation number,
  with given ranks `row_ranks` and `col_ranks`.
"""
function Base.randn(N::Int, d::Int, k::Int,
                    row_ranks::OffsetVector{Int, Vector{Int}}, 
                    col_ranks::OffsetVector{Int, Vector{Int}},
                    ::Type{T}=Float64) where T<:Number
  A = SparseCore{T,N,d}(k)
  @boundscheck A.row_qn == axes(row_ranks, 1)
  @boundscheck A.col_qn == axes(col_ranks, 1)
  A.row_ranks .= row_ranks
  A.col_ranks .= col_ranks

  # (  sum(row_ranks[n]*col_ranks[n  ] for n in axes(A.unoccupied, 1))
  #      + sum(row_ranks[n]*col_ranks[n+1] for n in axes(A.occupied,   1))
  #     )^(-1/2)

  for n in axes(A.unoccupied, 1)
    if row_ranks[n]>0 && col_ranks[n]>0
      σ = ( row_ranks[n]*col_ranks[n] )^(-1/2) # * binomial(d,N)/(binomial(k-1,n)*binomial(d-k,N-n)))^(-1/2)
    else
      σ = 1
    end
    A[n,1,n] = Block(σ, randn(T,row_ranks[n],col_ranks[n]))
    # @show norm(A[n,1,n])
  end
  for n in axes(A.occupied, 1)
    if row_ranks[n]>0 && col_ranks[n+1]>0
      σ = ( row_ranks[n]*col_ranks[n+1] )^(-1/2) # * binomial(d,N)/(binomial(k-1,n)*binomial(d-k,N-n-1)))^(-1/2)
    else
      σ = 1
    end
    A[n,2,n+1] = Block(σ, randn(T,row_ranks[n],col_ranks[n+1]))
    # @show norm(A[n,2,n+1])
  end
  return A
end

"""
  A = rand(S, N::Int, d::Int, k::Int,
                    row_ranks::OffsetVector{Int, Vector{Int}}, 
                    col_ranks::OffsetVector{Int, Vector{Int}})

  Compute a core with randomized entries in the indexable collection `S` within the sparse block structure allowed
  for the `k`-th core of a `d`-dimensional TT-tensor with `N` total occupation number,
  with given ranks `row_ranks` and `col_ranks`.
"""
function Base.rand(S, N::Int, d::Int, k::Int,
                    row_ranks::OffsetVector{Int, Vector{Int}}, 
                    col_ranks::OffsetVector{Int, Vector{Int}},
                    ::Type{T}=Float64) where {T<:Number}

  @assert eltype(S) <: Number
  @assert T == float(eltype(S))


  A = SparseCore{T,N,d}(k)
  @boundscheck A.row_qn == axes(row_ranks, 1)
  @boundscheck A.col_qn == axes(col_ranks, 1)
  A.row_ranks .= row_ranks
  A.col_ranks .= col_ranks

  # (  sum(row_ranks[n]*col_ranks[n  ] for n in axes(A.unoccupied, 1))
  #      + sum(row_ranks[n]*col_ranks[n+1] for n in axes(A.occupied,   1))
  #     )^(-1/2)
  for n in axes(A.unoccupied, 1)
    # if row_ranks[n]>0 && col_ranks[n]>0
    #   σ = ( row_ranks[n]*col_ranks[n] )^(-1/2) # * binomial(d,N)/(binomial(k-1,n)*binomial(d-k,N-n)))^(-1/2)
    # else
    #   σ = 1
    # end
    a = float(rand(S,row_ranks[n],col_ranks[n]))
    r = minimum(size(a))
    if r < 50 # Because the matrix is small, we double check the numerical rank of the sketch matrix
      while rank(a) < r
        a = float(rand(S,row_ranks[n],col_ranks[n]))
      end
    end
    A[n,1,n] = Block(a)
    # @show norm(A[n,1,n])
  end
  for n in axes(A.occupied, 1)
    # if row_ranks[n]>0 && col_ranks[n+1]>0
    #   σ = ( row_ranks[n]*col_ranks[n+1] )^(-1/2) # * binomial(d,N)/(binomial(k-1,n)*binomial(d-k,N-n-1)))^(-1/2)
    # else
    #   σ = 1
    # end
    a = float(rand(S,row_ranks[n],col_ranks[n+1]))
    r = minimum(size(a))
    if r < 50 # Because the matrix is small, we double check the numerical rank of the sketch matrix
      if n ∈ axes(A.unoccupied, 1) && n+1 ∈ axes(A.unoccupied, 2)
        while rank(hcat(data(A.unoccupied[n]),a)) < min(row_ranks[n], col_ranks[n]+col_ranks[n+1]) &&
              rank(vcat(data(A.unoccupied[n+1]),a)) < min(row_ranks[n]+row_ranks[n+1], col_ranks[n+1])
          a = float(rand(S,row_ranks[n],col_ranks[n+1]))
        end
      elseif n ∈ axes(A.unoccupied, 1)
        while rank(hcat(data(A.unoccupied[n]),a)) < min(row_ranks[n], col_ranks[n]+col_ranks[n+1])
          a = float(rand(S,row_ranks[n],col_ranks[n+1]))
        end
      elseif n+1 ∈ axes(A.unoccupied, 2)
        while rank(vcat(data(A.unoccupied[n+1]),a)) < min(row_ranks[n]+row_ranks[n+1],col_ranks[n+1])
          a = float(rand(S,row_ranks[n],col_ranks[n+1]))
        end
      else
        while rank(a) < r
          a = float(rand(S,row_ranks[n],col_ranks[n+1]))
        end
      end
    end
    A[n,2,n+1] = Block(a)
    # @show norm(A[n,2,n+1])
  end
  return A
end



"""
    tt = tt_rand(S, Val(d), Val(N), r::Vector{OffsetVector{Int,Vector{Int}}}, [T=Float64])

Compute a d-dimensional TT-tensor with ranks `r` and entries drawn uniformly from the indexable collection `S` for the cores.
"""
function tt_rand(S, ::Val{d}, ::Val{N},  r::Vector{OffsetVector{Int,Vector{Int}}},
                    ::Type{T}=Float64) where {T<:Number,N,d}
  @assert eltype(S) <: Number
  @assert T == float(eltype(S))

  @boundscheck (length(r) == d+1) || (length(r) == d-1)
  for k=1:d+1
    @boundscheck axes(r[k],1) == occupation_qn(N,d,k)
  end

  tt = tt_zeros(Val(d),Val(N),T)
  if length(r) == d+1
    for k=1:d
      @boundscheck axes(r[k],1) == axes(core(tt,k),1)
    end
    @boundscheck axes(r[d+1],1) == axes(core(tt,d),3)
    tt.r .= r
    for k=1:d
      C = rand(S, N,d,k,rank(tt,k),rank(tt,k+1))
      set_core!(tt, C)
    end

  else # length(r) == d-1
    for k=1:d-1
      @boundscheck axes(r[k],1) == axes(core(tt,k+1),1)
    end
    tt.r[2:d] .= r
    for k=1:d
      set_core!(tt, rand(S, N,d,k,rank(tt,k),rank(tt,k+1)))
    end
  end

  check(tt)

  return tt
end

"""
    tt = tt_randn(Val(d), Val(N), r::Vector{OffsetVector{Int,Vector{Int}}}, [T=Float64])

Compute a d-dimensional TT-tensor with ranks `r` and Gaussian distributed entries for the cores.
"""
function tt_randn(::Val{d}, ::Val{N}, r::Vector{OffsetVector{Int,Vector{Int}}}, ::Type{T}=Float64) where {T<:Number,N,d}
  @boundscheck (length(r) == d+1) || (length(r) == d-1)
  for k=1:d+1
    @boundscheck axes(r[k],1) == occupation_qn(N,d,k)
  end

  tt = tt_zeros(Val(d),Val(N),T)
  if length(r) == d+1
    for k=1:d
      @boundscheck axes(r[k],1) == axes(core(tt,k),1)
    end
    @boundscheck axes(r[d+1],1) == axes(core(tt,d),3)
    tt.r .= r
    for k=1:d
      C = randn(N,d,k,rank(tt,k),rank(tt,k+1),T)
      set_core!(tt, C)
    end

  else # length(r) == d-1
    for k=1:d-1
      @boundscheck axes(r[k],1) == axes(core(tt,k+1),1)
    end
    tt.r[2:d] .= r
    for k=1:d
      set_core!(tt, randn(N,d,k,rank(tt,k),rank(tt,k+1),T))
    end
  end

  check(tt)

  return tt
end

"""
    tt_out = perturbation(tt::TTvector{T,N,d}, r::Vector{OffsetVector{Int,Vector{Int}}}, [ϵ::Float64 = 1e-4])

Compute a d-dimensional TT-tensor perturbation of `tt` with ranks at least `r` and Gaussian distributed entries for the cores.
"""
function perturbation(tt::TTvector{T,N,d}, r::Vector{OffsetVector{Int,Vector{Int}}}, ϵ::Float64 = 1e-4) where {T<:Number,N,d}
  @boundscheck (length(r) == d+1) || (length(r) == d-1)
  for k=1:d+1
    @boundscheck axes(r[k],1) == occupation_qn(N,d,k)
  end

  rp = deepcopy(rank(tt))

  if length(r) == d+1
    for k=1:d+1
      @boundscheck axes(r[k],1) == axes(rank(tt,k),1)
      for n in axes(r[k],1)
        rp[k][n] = max(1, r[k][n] - rank(tt,k,n))
      end
    end
  else # length(r) == d-1
    for k=2:d
      @boundscheck axes(r[k-1],1) == axes(rank(tt,k),1)
      for n in axes(r[k-1],1)
        rp[k][n] = max(1, r[k-1][n] - rank(tt,k,n))
      end
    end
  end
  p = tt_randn(Val(d),Val(N),rp)
  lmul!(ϵ*norm(tt)/norm(p), p)
  return tt + p
end

"""
    tt_out = perturbation(tt::TTvector{T,N,d}, r::Int, [ϵ::Float64 = 1e-4])

Compute a d-dimensional TT-tensor perturbation of `tt` with all block ranks at least `r` and Gaussian distributed entries for the cores.
"""
function perturbation(tt::TTvector{T,N,d}, r::Int, ϵ::Float64 = 1e-4) where {T<:Number,N,d}
  r = [ [(2≤k≤d ? r : 1) for n in axes(rank(tt,k),1)] for k=1:d+1]
  return perturbation(tt,r,ϵ)
end