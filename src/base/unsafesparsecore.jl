"""
  UnsafeSparseCore{T<:Number,N,d,k}

Minimal structure holding references to blocks in a SparseCore, for use in matrix free applications.
No size or rank information for the block structure is checked or retained. 

The special function `isnonzero(X,l,r)` `isnonzero(X,l,s,r)` should be used to check for existence of any given block.

N is the total number of electrons and d is the overall tensor order; dictates structure.
"""
struct UnsafeSparseCore{T<:Number,N,d}
  unoccupied::OffsetVector{Block{T}, Vector{Block{T}}}
  occupied::OffsetVector{Block{T}, Vector{Block{T}}}
end

function UnsafeSparseCore{T,N,d}(ax_unocc::UnitRange{Int64}, ax_occ::UnitRange{Int64}) where {T<:Number,N,d}
  return UnsafeSparseCore{T,N,d}(
            OffsetVector(Vector{Block{T}}(undef, length(ax_unocc)), ax_unocc), 
            OffsetVector(Vector{Block{T}}(undef, length(ax_occ  )), ax_occ  ))
end

function Base.similar(A::UnsafeSparseCore{T,N,d}) where {T<:Number,N,d}
  return UnsafeSparseCore{T,N,d}(similar(A.unoccupied), similar(A.occupied))
end

@inline function Base.getindex(A::UnsafeSparseCore, l::Int, r::Int)
  if l == r # Unoccupied state
    return A.unoccupied[l]
  elseif l+1 == r # Occupied state
    return A.occupied[l]
  else # Forbidden
    throw(BoundsError(A, (l,r-l+1,r)))
  end
end

@inline function Base.getindex(A::UnsafeSparseCore, l::Int, s::Int, r::Int)
  if s==1 && l == r # Unoccupied state
    return A.unoccupied[l]
  elseif s==2 && l+1 == r # Occupied state
    return A.occupied[l]
  else # Forbidden
    throw(BoundsError(A, (l,s,r)))
  end
end

@inline function Base.setindex!(A::UnsafeSparseCore{T}, X::Block{T}, l::Int, r::Int) where T<:Number
  if l == r # Unoccupied state
    A.unoccupied[l] = X
  elseif l+1 == r # Occupied state
    A.occupied[l] = X
  else # Forbidden
    throw(BoundsError(A, (l,r-l+1,r)))
  end
end

@inline function Base.setindex!(A::UnsafeSparseCore{T}, X::Block{T}, l::Int, s::Int, r::Int) where T<:Number
  @boundscheck @assert l+s == r+1
  if s==1 && l == r # Unoccupied state
    A.unoccupied[l] = X
  elseif s==2 && l+1 == r # Occupied state
    A.occupied[l] = X
  else # Forbidden
    throw(BoundsError(A, (l,s,r)))
  end
end

@inline function isnonzero(A::UnsafeSparseCore, l::Int, r::Int)
  return l   == r && l ∈ axes(A.unoccupied,1) && isnonzero(A.unoccupied[l]) ? true :
         l+1 == r && l ∈ axes(A.occupied,  1) && isnonzero(A.occupied[l]  ) ? true : 
         false
end

@inline function isnonzero(A::UnsafeSparseCore, l::Int, s::Int, r::Int)
  return s==1 && l   == r && l ∈ axes(A.unoccupied,1) && isnonzero(A.unoccupied[l]) ? true :
         s==2 && l+1 == r && l ∈ axes(A.occupied,  1) && isnonzero(A.occupied[l]  ) ? true : 
         false
end

@inline function shift_qn(qn::AbstractRange, flux::Int, nl::Int, nr::Int, N::Int)
  start = min(max(nl,   first(qn)+(flux>0 ? flux : 0)), last(qn )+1)
  stop  = max(min(N-nr, last(qn ) +(flux<0 ? flux : 0)),first(qn)-1)
  return start:stop
end


"""
  SparseCore(k::Int, row_ranks::OffsetVector{Int, Vector{Int}}, col_ranks::OffsetVector{Int, Vector{Int}}, A::UnsafeSparseCore{T<:Number,N,d})

Convert an UnsafeSparseCore back into a standard SparseCore object with input of the missing rank information.
"""

function SparseCore(k::Int, row_ranks::OffsetVector{Int, Vector{Int}}, col_ranks::OffsetVector{Int, Vector{Int}}, A::UnsafeSparseCore{T,N,d}) where {T<:Number,N,d}
  B = SparseCore{T,N,d}(k)

  B.row_ranks .= row_ranks
  B.col_ranks .= col_ranks
  for l in axes(B.unoccupied,1)
    if l ∈ axes(A.unoccupied,1)
      @boundscheck @assert size(A.unoccupied[l]) == (row_ranks[l],col_ranks[l])
      B.unoccupied[l] = A.unoccupied[l]
    else
      B.unoccupied[l] = zeros_block(T,row_ranks[l],col_ranks[l])
    end
  end
  for l in axes(B.occupied,1)
    if l ∈ axes(A.occupied,1)
      @boundscheck @assert size(A.occupied[l]) == (row_ranks[l],col_ranks[l+1])
      B.occupied[l] = A.occupied[l]
    else
      B.occupied[l] = zeros_block(T,row_ranks[l],col_ranks[l+1])
    end
  end
  return B
end
