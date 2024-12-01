"""
    round!(tt::TTvector{T,Nup,Ndn,d}, [ϵ=1e-14][; rmax = 0])

Truncates the ranks of `tt` with specified accuracy ϵ (default 1e-14) and maximal rank rmax.
Implements TT-rounding algorithm.
"""
function round!(tt::TTvector{T,Nup,Ndn,d}, ϵ::Float64=1e-14; rmax::Union{Int,Vector{Int}} = 0) where {T<:Number,Nup,Ndn,d}
  if typeof(rmax)==Int
    if rmax > 0
      max_ranks = zeros(Int, d+1)
      for k=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,k)
        max_ranks[k] += min(rmax, binomial(k-1,nup-1)*binomial(k-1,ndn-1), binomial(d+1-k,Nup+1-nup)*binomial(d+1-k,Ndn+1-ndn))
      end
      rmax = max_ranks
    elseif rmax == 0
      rmax = zeros(Int, d+1)
      for k=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,k)
        rmax[k] += min(binomial(k-1,nup-1)*binomial(k-1,ndn-1), binomial(d+1-k,Nup+1-nup)*binomial(d+1-k,Ndn+1-ndn))
      end
      if any(rmax .> typemax(Int))
        @warn "Risk of Int overflow - theoretical maximum rank above $(typemax(Int))"
        rmax = min.(rmax, typemax(Int))
      end
    end
  end
  @assert length(rmax) == d+1

  leftOrthogonalize!(tt)

  ep = norm(tt)^2/sqrt(d-1) * ϵ
  for k = d:-1:2
    U, sval, Vt = svd_horizontal(core(tt,k))
    rmul!(U, sval)
    ranks = chop(sval, ep)
    ranks = min.(ranks, rmax[k])

    Xₖ = SparseCore{T,Nup,Ndn,d}(k, ranks, rank(tt,k+1))
    C = Frame{T,Nup,Ndn,d}(k, U.row_ranks, ranks)
    for (nup,ndn) in row_qn(core(tt,k))
      Xₖ[(nup,ndn),:horizontal] = view(Vt[nup,ndn],   1:ranks[nup,ndn], :)
      copyto!(block(C,nup,ndn), 1:rank(tt,k,(nup,ndn)), 1:ranks[nup,ndn], 
              block(U,nup,ndn), 1:rank(tt,k,(nup,ndn)), 1:ranks[nup,ndn])
    end

    set_cores!(tt, core(tt,k-1) * C, Xₖ)
  end

  tt.orthogonal = true
  tt.corePosition = 1
  check(tt)

  return tt
end

"""
    round!(tt::TTvector{T,Nup,Ndn,d}, [ϵ=1e-14][; rmax = 0])

Truncates the ranks of `tt` to the specified maximal ranks.
Implements TT-rounding algorithm.
"""
function round!(tt::TTvector{T,Nup,Ndn,d}, ranks::Vector{Matrix{Int}}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert length(ranks) == d+1
    for k=1:d+1
      @assert findall(ranks[k].>0) ⊆ CartesianIndex.(state_qn(Nup,Ndn,d,k))
    end
  end

  leftOrthogonalize!(tt)
  tt.r = ranks

  for k = d:-1:2
    U, sval, Vt = svd_horizontal(core(tt,k))
    rmul!(U, sval)

    Xₖ = SparseCore{T,Nup,Ndn,d}(k, ranks[k],ranks[k+1])
    C = Frame{T,Nup,Ndn,d}(k, U.row_ranks, ranks[k])

    for (nup,ndn) in row_qn(core(tt,k))
      Xₖ[(nup,ndn),:horizontal] = view(Vt[nup,ndn],   1:ranks[k][nup,ndn], :)
      copyto!(block(C,nup,ndn), 1:rank(tt,k,(nup,ndn)), 1:ranks[k][nup,ndn], 
              block(U,nup,ndn), 1:rank(tt,k,(nup,ndn)), 1:ranks[k][nup,ndn])
    end

    set_core!(tt, Xₖ)
    set_core!(tt, core(tt,k-1) * C)
    tt.r[k] = ranks[k]
  end

  tt.orthogonal = true
  tt.corePosition = 1
  check(tt)

  return tt
end

"""
    round(tt::TTvector{T,Nup,Ndn,d}, [ϵ=1e-14][; rmax = 0])

Create an approximation of `tt` with specified accuracy ϵ (default 1e-14) and maximal rank rmax.
Implements TT-rounding algorithm.
"""
function Base.round(tt::TTvector{T,Nup,Ndn,d}, ϵ::Float64=1e-14; rmax::Union{Int,Array{Int,1}} = 0, Global::Bool=false) where {T<:Number,Nup,Ndn,d}
  tt1 = deepcopy(tt)
  if (Global)
    round_global!(tt1,ϵ,rmax=rmax)
  else
    round!(tt1, ϵ, rmax=rmax)
  end
  return tt1
end

"""
    round(tt::TTvector{T,Nup,Ndn,d}, rmax::Vector{OffsetVector{Int,Vector{Int}}})

Create an approximation of `tt` with specified maximal ranks rmax.
Implements TT-rounding algorithm.
"""
function Base.round(tt::TTvector{T,Nup,Ndn,d}, rmax::Vector{Matrix{Int}}) where {T<:Number,Nup,Ndn,d}
  return round!(deepcopy(tt), rmax)
end

"""
    round_global!(tt::TTvector{T,Nup,Ndn,d}, [ϵ=1e-14][; rmax = 0])

Truncates the ranks of `tt` with specified accuracy ϵ (default 1e-14) and maximal rank rmax.
Implements TT-rounding algorithm with global chopping strategy.
"""
function round_global!(tt::TTvector{T,Nup,Ndn,d}, ϵ::Float64=1e-14; rmax::Int = 0) where {T<:Number,Nup,Ndn,d}
  max_ranks = [zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  if rmax > 0
    for k=1:d+1, (nup,ndn) in state_qn(Nup,Ndn,d,k)
      max_ranks[k][nup,ndn] = min(rmax, binomial(k-1,nup-1)*binomial(k-1,ndn-1), binomial(d+1-k,Nup+1-nup)*binomial(d+1-k,Ndn+1-ndn))
    end
  else
    for k=1:d+1, (nup,ndn) in state_qn(Nup,Ndn,d,k)
      max_ranks[k][nup,ndn] = min(binomial(k-1,nup-1)*binomial(k-1,ndn-1), binomial(d+1-k,Nup+1-nup)*binomial(d+1-k,Ndn+1-ndn))
    end
  end
  return round_global!(tt, max_ranks, ϵ)
end

function round_global!(tt::TTvector{T,Nup,Ndn,d}, rmax::Vector{Matrix{Int}}, ϵ::Float64=1e-14) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert length(rmax) == d+1
    for k=1:d+1
      @assert findall(rmax[k].>0) ⊆ CartesianIndex.(state_qn(Nup,Ndn,d,k))
    end
  end

  svals = svdvals(tt)
  ranks = chop(svals, norm(tt)*ϵ)

  for k=1:d-1
    ranks[k] = min.(ranks[k], rmax[k+1])
    rank(tt,k+1) .= ranks[k]
    set_core!!(tt, reduce_ranks(core(tt,k), rank(tt,k), rank(tt,k+1)))
  end
  let k=d
    set_core!!(tt, reduce_ranks(core(tt,k), rank(tt,k), rank(tt,k+1)))
  end

  tt.orthogonal = false
  check(tt)

  return tt
end

"""
    round_global(tt::TTvector{T,Nup,Ndn,d}, [ϵ=1e-14][; rmax = 0])

Create an approximation of `tt` with specified accuracy ϵ (default 1e-14) and maximal rank rmax.
Implements TT-rounding algorithm with global chopping strategy.
"""
@inline function round_global(tt, ϵ; rmax=0)
  return round_global!(deepcopy(tt), ϵ, rmax=rmax)
end

@inline function round_global(tt, rmax, ϵ)
  return round_global!(deepcopy(tt), rmax, ϵ)
end

"""
    svdvals(tt::TTvector{T,Nup,Ndn,d})

Compute all nonzero singular values of all matrix unfoldings of `tt`, returns result as an array of arrays.
This function modifies the internal state of the TT-tensor by orthogonalizing it.
"""
function LinearAlgebra.svdvals(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}

  tt1 = deepcopy(tt)
  leftOrthogonalize!(tt)

  svals = Vector{Frame{T,Nup,Ndn,d,Diagonal{T,Vector{T}}}}(undef, d-1)
  for k = d:-1:2
    U, S, Cₖ = svd_horizontal_to_core(core(tt,k))
    rmul!(U,S)
    svals[k-1] = S
    set_cores!(tt, core(tt,k-1) * U, Cₖ)
  end

  tt.orthogonal = true
  tt.corePosition = 1

  return svals
end

"""
    qr(tt::TTvector{T,Nup,Ndn,d}, op::Symbol)

Compute the full left or right orthogonalization of `tt`.
Return `R` triangular matrix for the block case, when trailing ranks `r[1]` or `r[d+1]` are not equal to `1`.

`op` may be
  * `:LR` for left-right orthogonalization,
  * `:RL` for right-left orthogonalization.
"""
function LinearAlgebra.qr(tt::TTvector{T,Nup,Ndn,d}, op::Symbol) where {T<:Number,Nup,Ndn,d}
  tt1 = copy(tt)
  R = qr!(tt1, op)
  return tt1, R
end

"""
    qr!(tt::TTvector{T,Nup,Ndn,d}, op::Symbol)

Like [`qr`](@ref), but acts in-place on `tt`.
"""
function LinearAlgebra.qr!(tt::TTvector{T,Nup,Ndn,d}, op::Symbol) where {T<:Number,Nup,Ndn,d}
  @assert op == :LR || op == :RL

  if op == :LR     # Orthogonalization from left to right
    leftOrthogonalize!(tt, keepRank=true)
    R = qr!(tt.cores[d])
    return R[N]
  else # op == :RL # Orthogonalization from right to left
    rightOrthogonalize!(tt, keepRank=true)
    L = lq!(tt.cores[1])
    return L[0]
  end
end

function step_core_right!(tt::TTvector{T,Nup,Ndn,d}, k::Int; keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  @assert 1 ≤ k ≤ d-1

  if (keepRank)
    R = qr!(core(tt,k))
    lmul!(R, core(tt,k+1))
  else
    Qk, C, ranks = qc!(core(tt,k))
    set_cores!(tt, Qk, C * core(tt,k+1))
  end

  if tt.orthogonal && (tt.corePosition == k)
    tt.corePosition = k+1
  end
  return tt
end

function step_core_left!(tt::TTvector{T,Nup,Ndn,d}, k::Int; keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  @assert 2 ≤ k ≤ d

  if (keepRank)
    L = lq!(core(tt,k))
    rmul!(core(tt,k-1), L)
  else
    Qk, C, ranks = cq!(core(tt,k))
    set_cores!(tt, core(tt,k-1) * C, Qk)
  end
  
  if tt.orthogonal && (tt.corePosition == k)
    tt.corePosition = k-1
  end
  return tt
end

"""
    move_core!(tt::TTvector{T,Nup,Ndn,d}, corePosition::Int; [keepRank::Bool=false])

Move the orthogonalization center of `tt` from its current position to mode index `i` using rank-revealing QR-decompositions, reducing accordingly the rank of `tt`.
Optionally you can specify `keepRank=true` to make sure the rank is not reduced, which may be needed e.g. for ALS.
If `tt` is not orthogonalized, it will be after the call.
"""
function move_core!(tt::TTvector{T,Nup,Ndn,d}, corePosition::Int; keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  @assert 1 ≤ corePosition ≤ d

  if tt.orthogonal
    if tt.corePosition > corePosition
      for k=tt.corePosition:-1:corePosition+1
        step_core_left!(tt,k,keepRank=keepRank)
      end
    elseif tt.corePosition < corePosition
      for k=tt.corePosition:corePosition-1
        step_core_right!(tt,k,keepRank=keepRank)
      end
    end
  else
    for k=1:corePosition-1
      step_core_right!(tt,k,keepRank=keepRank)
    end
    for k=d:-1:corePosition+1
      step_core_left!(tt,k,keepRank=keepRank)
    end
    tt.orthogonal = true
    tt.corePosition = corePosition
  end
  return tt
end

"""
    leftOrthogonalize!(tt::TTvector{T,Nup,Ndn,d}; keepRank::Bool=false)

Compute the full left orthogonalization of `tt`, where the vertical unfolding 
of the cores 1,...,d-1 have orthogonal columns.
"""
function leftOrthogonalize!(tt::TTvector{T,Nup,Ndn,d}; keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  move_core!(tt, d; keepRank=keepRank)
  return tt
end

"""
    rightOrthogonalize!(tt::TTvector{T,Nup,Ndn,d}; keepRank::Bool=false)

Compute the full right orthogonalization of `tt`, where the horizontal unfolding 
of the cores 2,...,d have orthogonal rows.
"""
function rightOrthogonalize!(tt::TTvector{T,Nup,Ndn,d}; keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  move_core!(tt, 1; keepRank=keepRank)
  return tt
end
