"""
    round!(tt::TTvector{T,N,d}, [ϵ=1e-14][; rmax = 0])

Truncates the ranks of `tt` with specified accuracy ϵ (default 1e-14) and maximal rank rmax.
Implements TT-rounding algorithm.
"""
function round!(tt::TTvector{T,N,d}, ϵ::Float64=1e-14; rmax::Union{Int,Vector{Int}} = 0) where {T<:Number,N,d}

  @boundscheck if rmax == 0
    rmax = [min(2^(k-1), 2^(d+1-k)) for k=1:d+1]
    if any(rmax .> typemax(Int))
      @warn "Risk of Int overflow - theoretical maximum rank above $(typemax(Int))"
      rmax = min.(rmax, typemax(Int))
    end
  elseif length(rmax) == 1
    rmax = [min(rmax, 2^(k-1), 2^(d+1-k)) for k=1:d+1]
  else
    @assert length(rmax) == d+1
  end

  leftOrthogonalize!(tt)

  ep = norm(tt)^2/sqrt(d-1) * ϵ
  for k = d:-1:2
    U, sval, Vt = svd(core(tt,k), :horizontal)
    ranks = chop(sval, ep)
    ranks = min.(ranks, rmax[k])

    for n in axes(core(tt,k),1)
      r = ranks[n]
      tt.r[k][n] = r
      core(tt,k)[n,:horizontal] = Vt[n][1:r,:]
      U[n] = U[n][:,1:r] * Diagonal(sval[n][1:r])
    end
    rmul!(core(tt,k-1), U)
  end

  tt.orthogonal = true
  tt.corePosition = 1
  check(tt)

  return tt
end

"""
    round!(tt::TTvector{T,N,d}, [ϵ=1e-14][; rmax = 0])

Truncates the ranks of `tt` to the specified maximal ranks.
Implements TT-rounding algorithm.
"""
function round!(tt::TTvector{T,N,d}, ranks::Vector{OffsetVector{Int,Vector{Int}}}) where {T<:Number,N,d}
  @assert length(ranks) == d+1
  if typeof(ranks) <: Vector{OffsetVector{Int,Vector{Int}}}
    for k=1:d+1
      @assert axes(ranks[k],1) == occupation_qn(N,d,k)
    end
  end

  leftOrthogonalize!(tt)
  tt.r = ranks

  for k = d:-1:2
    U, sval, Vt = svd(core(tt,k), :horizontal)

    for n in axes(core(tt,k),1)
      r = ranks[k][n]
      if length(sval[n]) < r
        # Pad with zeros to achieve desired ranks
        U[n] = hcat(U[n], zeros(T,size(U[n],1),r-size(U[n],2)))
        sval[n] = vcat(sval[n], zeros(T,r-length(sval[n])))
        Vt[n] = vcat(Vt[n], zeros(T,r-size(Vt[n],1),size(Vt[n],2)))
      end
      core(tt,k)[n,:horizontal] = Vt[n][1:r,:]
      U[n] = U[n][:,1:r] * Diagonal(sval[n][1:r])
    end
    rmul!(core(tt,k-1), U)
  end

  tt.orthogonal = true
  tt.corePosition = 1
  check(tt)

  return tt
end

"""
    round(tt::TTvector{T,N,d}, [ϵ=1e-14][; rmax = 0])

Create an approximation of `tt` with specified accuracy ϵ (default 1e-14) and maximal rank rmax.
Implements TT-rounding algorithm.
"""
function Base.round(tt::TTvector{T,N,d}, ϵ::Float64=1e-14; rmax::Union{Int,Array{Int,1}} = 0, Global::Bool=false) where {T<:Number,N,d}
  tt1 = deepcopy(tt)
  if (Global)
    round_global!(tt1,ϵ,rmax=rmax)
  else
    round!(tt1, ϵ, rmax=rmax)
  end
  return tt1
end

"""
    round(tt::TTvector{T,N,d}, rmax::Vector{OffsetVector{Int,Vector{Int}}})

Create an approximation of `tt` with specified maximal ranks rmax.
Implements TT-rounding algorithm.
"""
function Base.round(tt::TTvector{T,N,d}, rmax::Vector{OffsetVector{Int,Vector{Int}}}) where {T<:Number,N,d}
  return round!(deepcopy(tt), rmax)
end

"""
    round_global!(tt::TTvector{T,N,d}, [ϵ=1e-14][; rmax = 0])

Truncates the ranks of `tt` with specified accuracy ϵ (default 1e-14) and maximal rank rmax.
Implements TT-rounding algorithm with global chopping strategy.
"""
function round_global!(tt::TTvector{T,N,d}, ϵ::Float64=1e-14; rmax::Union{Int,Array{Int,1}} = 0) where {T<:Number,N,d}

  if rmax == 0
    rmax = [min(2^(k-1), 2^(d+1-k)) for k=1:d+1]
    if any(rmax .> typemax(Int))
      @warn "Risk of Int overflow - theoretical maximum rank above $(typemax(Int))"
      rmax = min.(rmax, typemax(Int))
    end
    rmax = convert(Vector{Int}, rmax)
  elseif length(rmax) == 1
    sizes = convert(NTuple{d,BigInt}, size(tt))
    rmax = [min(rmax, 2^(k-1), 2^(d+1-k)) for k=1:d+1]
    rmax = convert(Vector{Int}, rmax)
  else
    @assert length(rmax) == d+1
  end

  svals = svdvals(tt)

  ranks = chop(svals, norm(tt)*ϵ)
  for k=1:d-1
    for n in axes(ranks[k],1)
      ranks[k][n] = min(ranks[k][n], rmax[k])
    end
    tt.r[k+1] = ranks[k]
  end

  for k=1:d
    reduce_ranks!(core(tt,k), tt.r[k], tt.r[k+1])
  end

  tt.orthogonal = false
  check(tt)

  return tt
end

"""
    round_global(tt::TTvector{T,N,d}, [ϵ=1e-14][; rmax = 0])

Create an approximation of `tt` with specified accuracy ϵ (default 1e-14) and maximal rank rmax.
Implements TT-rounding algorithm with global chopping strategy.
"""
function round_global(tt::TTvector{T,N,d}, ϵ::Float64=1e-14; rmax::Union{Int,Array{Int,1}} = 0) where {T<:Number,N,d}
  tt1 = deepcopy(tt)
  round2!(tt1, ϵ, rmax=rmax)
  return tt1
end

"""
    svdvals(tt::TTvector{T,N,d})

Compute all nonzero singular values of all matrix unfoldings of `tt`, returns result as an array of arrays.
This function modifies the internal state of the TT-tensor by orthogonalizing it.
"""
function LinearAlgebra.svdvals(tt::TTvector{T,N,d}) where {T<:Number,N,d}

  leftOrthogonalize!(tt)

  svals = [ [T[] for n in axes(core(tt,k), 1)] for k=2:d]
  for k = d:-1:2
    U, S = svd!(core(tt,k), :horizontal)

    svals[k-1] .= S
    rmul!(core(tt,k-1), U.*Diagonal.(S))
  end

  tt.orthogonal = true
  tt.corePosition = 1
  return svals
end

"""
    qr(tt::TTvector{T,N,d}, op::Symbol)

Compute the full left or right orthogonalization of `tt`.
Return `R` triangular matrix for the block case, when trailing ranks `r[1]` or `r[d+1]` are not equal to `1`.

`op` may be
  * `:LR` for left-right orthogonalization,
  * `:RL` for right-left orthogonalization.
"""
function LinearAlgebra.qr(tt::TTvector{T,N,d}, op::Symbol) where {T<:Number,N,d}
  tt1 = deepcopy(tt)
  R = qr!(tt1, op)
  return tt1, R
end

"""
    qr!(tt::TTvector{T,N,d}, op::Symbol)

Like [`qr`](@ref), but acts in-place on `tt`.
"""
function LinearAlgebra.qr!(tt::TTvector{T,N,d}, op::Symbol) where {T<:Number,N,d}
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

function step_core_right!(tt::TTvector{T,N,d}, k::Int; keepRank::Bool=false) where {T<:Number,N,d}
  @assert 1 ≤ k ≤ d-1

  if (keepRank)
    R = qr!(core(tt,k))
    lmul!(R, core(tt,k+1))
  else
    C, ranks = qc!(core(tt,k))
    lmul!(C, core(tt,k+1))
    tt.r[k+1] = ranks
  end

  if tt.orthogonal && (tt.corePosition == k)
    tt.corePosition = k+1
  end
  return tt
end

function step_core_left!(tt::TTvector{T,N,d}, k::Int; keepRank::Bool=false) where {T<:Number,N,d}
  @assert 2 ≤ k ≤ d

  if (keepRank)
    L = lq!(core(tt,k))
    rmul!(core(tt,k-1), L)
  else
    C, ranks = cq!(core(tt,k))
    rmul!(core(tt,k-1), C)
    tt.r[k] .= ranks
  end
  
  if tt.orthogonal && (tt.corePosition == k)
    tt.corePosition = k-1
  end
  return tt
end

"""
    move_core!(tt::TTvector{T,N,d}, corePosition::Int; [keepRank::Bool=false])

Move the orthogonalization center of `tt` from its current position to mode index `i` using rank-revealing QR-decompositions, reducing accordingly the rank of `tt`.
Optionally you can specify `keepRank=true` to make sure the rank is not reduced, which may be needed e.g. for ALS.
If `tt` is not orthogonalized, it will be after the call.
"""
function move_core!(tt::TTvector{T,N,d}, corePosition::Int; keepRank::Bool=false) where {T<:Number,N,d}
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
    leftOrthogonalize!(tt::TTvector{T,N,d}; keepRank::Bool=false)

Compute the full left orthogonalization of `tt`, where the vertical unfolding 
of the cores 1,...,d-1 have orthogonal columns.
"""
function leftOrthogonalize!(tt::TTvector{T,N,d}; keepRank::Bool=false) where {T<:Number,N,d}
  move_core!(tt, d; keepRank=keepRank)
  return tt
end

"""
    rightOrthogonalize!(tt::TTvector{T,N,d}; keepRank::Bool=false)

Compute the full right orthogonalization of `tt`, where the horizontal unfolding 
of the cores 2,...,d have orthogonal rows.
"""
function rightOrthogonalize!(tt::TTvector{T,N,d}; keepRank::Bool=false) where {T<:Number,N,d}
  move_core!(tt, 1; keepRank=keepRank)
  return tt
end
