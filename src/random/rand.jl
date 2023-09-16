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

  # σ = (  sum(row_ranks[n]*col_ranks[n  ] for n in axes(A.unoccupied, 1))
  #      + sum(row_ranks[n]*col_ranks[n+1] for n in axes(A.occupied,   1))
  #     )^(-1/2)
  for n in axes(A.unoccupied, 1)
    if row_ranks[n]>0 && col_ranks[n]>0
      σ = ( row_ranks[n]*col_ranks[n] * binomial(d,N)/(binomial(k-1,n)*binomial(d-k,N-n)))^(-1/2)
    else
      σ = 1
    end
    A[n,1,n] = Block(σ, randn(T,row_ranks[n],col_ranks[n]))
    # @show norm(A[n,1,n])
  end
  for n in axes(A.occupied, 1)
    if row_ranks[n]>0 && col_ranks[n+1]>0
      σ = ( row_ranks[n]*col_ranks[n+1] * binomial(d,N)/(binomial(k-1,n)*binomial(d-k,N-n-1)))^(-1/2)
    else
      σ = 1
    end
    A[n,2,n+1] = Block(σ, randn(T,row_ranks[n],col_ranks[n+1]))
    # @show norm(A[n,2,n+1])
  end
  return A
end


"""
    tt = tt_rand(d::Int, N::Int, r::Vector{OffsetVector{Int,Vector{Int}}}, [T=Float64])

Compute a d-dimensional TT-tensor with ranks `r` and Gaussian distributed entries for the cores.
"""
function tt_rand(d::Int, N::Int, r::Vector{OffsetVector{Int,Vector{Int}}}, ::Type{T}=Float64) where {T<:Number}
  @boundscheck (length(r) == d+1) || (length(r) == d-1)
  for k=1:d+1
    @boundscheck axes(r[k],1) == occupation_qn(N,d,k)
  end

  tt = tt_zeros(d,N,T)
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