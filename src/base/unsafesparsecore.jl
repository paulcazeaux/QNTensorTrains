"""
  UnsafeSparseCore{T<:Number,N,d,k}

Minimal structure holding references to blocks in a SparseCore, for use in matrix free applications.
No size or rank information for the block structure is checked or retained. 

The special function `isnonzero(X,l,r)` `isnonzero(X,l,s,r)` should be used to check for existence of any given block.

N is the total number of electrons and d is the overall tensor order; dictates structure.
"""
mutable struct UnsafeSparseCore{T<:Number,N,d,S<:AbstractMatrix{T}}
  jw::Bool
  factor::T
  unoccupied::OffsetVector{S, Vector{S}}
  occupied::OffsetVector{S, Vector{S}}
end

function UnsafeSparseCore{T,N,d,S}(; 
          unoccupied = missing, occupied   = missing,
          jw::Bool = false,     factor::Number = 1
                                   ) where {T,N,d,S}
  
  return UnsafeSparseCore{T,N,d,S}(jw,T(factor), 
            ismissing(unoccupied) ? OffsetVector(Vector{S}(undef,0),1:0) : OffsetVector(unoccupied[2],unoccupied[1]), 
            ismissing(occupied)   ? OffsetVector(Vector{S}(undef,0),1:0) : OffsetVector(occupied[2],occupied[1]))
end

@inline function factor(A::UnsafeSparseCore)
  return A.factor
end

@inline function jw(A::UnsafeSparseCore)
  return A.jw
end

@inline @propagate_inbounds function factor(A::UnsafeSparseCore, l::Int, r::Int)
  @boundscheck @assert r==l || r==l+1
  return (jw(A) && r==l+1 ? -factor(A) : factor(A))
end

@inline function shift_qn(qn::AbstractRange, flux::Int, nl::Int, nr::Int, N::Int)
  start = min(max(nl,   first(qn)+(flux>0 ? flux : 0)), last(qn )+1)
  stop  = max(min(N-nr, last(qn ) +(flux<0 ? flux : 0)),first(qn)-1)
  return start:stop
end

@propagate_inbounds function LinearAlgebra.lmul!(a::Number, B::UnsafeSparseCore)
  B.factor *= a
  return B
end
"""

        --- 
          |
C =       B
      |   |
    --A-- -

"""
@propagate_inbounds @inline function LinearAlgebra.mul!(
                C::SparseCore{T,N,d}, 
                A::UnsafeSparseCore{T,N,d}, 
                B::Frame{T,N,d}, 
                α::Number, β::Number) where {T<:Number,N,d}
  @boundscheck begin
    @assert axes(C,3) == axes(B,2)
    @assert axes(A.unoccupied,1) ⊆ axes(C.unoccupied,1)
    @assert axes(A.occupied,1)   ⊆ axes(C.occupied,1)
  end
  for l in axes(C.unoccupied,1)
    if l ∈ axes(A.unoccupied,1) 
      mul!(C.unoccupied[l],A.unoccupied[l],B.blocks[l],α*factor(A),β)
    elseif β ≢ 1
      rmul!(C.unoccupied[l],β)
    end
  end
  for l in axes(C.occupied,1)
    if l ∈ axes(A.occupied,1)
      mul!(C.occupied[l],A.occupied[l],B.blocks[l+1],(jw(A) ? -α*factor(A) : α*factor(A)),β)
    elseif β ≉ 1
      rmul!(C.occupied[l],β)
    end
  end
  return C
end

"""
       --
       |
C =    A
       |    |
       -- --B--
"""
@propagate_inbounds @inline function LinearAlgebra.mul!(
                C::SparseCore{T,N,d}, 
                A::Frame{T,N,d},
                B::UnsafeSparseCore{T,N,d}, 
                α::Number, β::Number) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(A) == site(C)
    @assert axes(B.unoccupied,1) ⊆ axes(C.unoccupied,1)
    @assert axes(B.occupied,1)   ⊆ axes(C.occupied,1)
  end
  for l in axes(C.unoccupied,1)
    if l ∈ axes(B.unoccupied,1) 
      mul!(C.unoccupied[l],A.blocks[l],B.unoccupied[l],α*factor(B),β)
    elseif β ≢ 1
      rmul!(C.unoccupied[l],β)
    end
  end
  for l in axes(C.occupied,1)
    if l ∈ axes(B.occupied,1)
      mul!(C.occupied[l],A.blocks[l],B.occupied[l],(jw(B) ? -α*factor(B) : α*factor(B)),β)
    elseif β ≉ 1
      rmul!(C.occupied[l],β)
    end
  end
  return C
end


"""
    --B----
      |   |
C =       |
      |   |
    --A-- -
"""
@propagate_inbounds @inline function LinearAlgebra.mul!(
                C::Frame{T,N,d}, 
                A::UnsafeSparseCore{T,N,d}, 
                B::AdjointSparseCore{T,N,d},
                α::Number, β::Number) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(B) == site(C)
    @assert axes(A.unoccupied,1) ⊆ axes(C.blocks,1)
    @assert axes(A.occupied,1)   ⊆ axes(C.blocks,1)
  end
  for l in axes(C,1)
    if l ∈ axes(A.unoccupied,1)∩axes(A.occupied,1) 
      mul!(C.blocks[l],A.unoccupied[l],adjoint(parent(B).unoccupied[l]), α*factor(A),                         β)
      mul!(C.blocks[l],A.occupied[l],  adjoint(parent(B).occupied[l]  ), (jw(A) ? -α*factor(A) : α*factor(A)),1)
    elseif l ∈ axes(A.unoccupied,1)
      mul!(C.blocks[l],A.unoccupied[l],adjoint(parent(B).unoccupied[l]), α*factor(A),                         β)
    elseif l ∈ axes(A.occupied,1)
      mul!(C.blocks[l],A.occupied[l],  adjoint(parent(B).occupied[l]  ), (jw(A) ? -α*factor(A) : α*factor(A)),β)
    elseif β ≢ 1
      rmul!(C.blocks[l],β)
    end
  end
  return C
end

"""
       ----A--
       |   |
C =    |
       |   |
       - --B--
"""
@propagate_inbounds @inline function LinearAlgebra.mul!(
                C::Frame{T,N,d}, 
                A::AdjointSparseCore{T,N,d},
                B::UnsafeSparseCore{T,N,d}, 
                α::Number, β::Number) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(A) == site(C)-1
    @assert axes(B.unoccupied,1) ⊆ axes(parent(A).unoccupied,1)
    @assert axes(B.occupied,1) ⊆ axes(parent(A).occupied,1)
  end
  for r in axes(C,1)
    if r ∈ axes(B.unoccupied,1) && r-1 ∈ axes(B.occupied,1) 
      mul!(C.blocks[r], adjoint(parent(A).unoccupied[r]), B.unoccupied[r], α*factor(B),                         β)
      mul!(C.blocks[r], adjoint(parent(A).occupied[r-1]), B.occupied[r-1], (jw(B) ? -α*factor(B) : α*factor(B)),1)
    elseif r ∈ axes(B.unoccupied,1)
      mul!(C.blocks[r], adjoint(parent(A).unoccupied[r]), B.unoccupied[r], α*factor(B),                         β)
    elseif r-1 ∈ axes(B.occupied,1)
      mul!(C.blocks[r], adjoint(parent(A).occupied[r-1]), B.occupied[r-1], (jw(B) ? -α*factor(B) : α*factor(B)),β)
    elseif β ≢ 1
      rmul!(C.blocks[l],β)
    end
  end
  return C
end

"""
  SparseCore(k::Int, row_ranks::OffsetVector{Int, Vector{Int}}, col_ranks::OffsetVector{Int, Vector{Int}}, A::UnsafeSparseCore{T<:Number,N,d})

Convert an UnsafeSparseCore back into a standard SparseCore object with input of the missing rank information.
"""

function SparseCore(k::Int, row_ranks::OffsetVector{Int, Vector{Int}}, col_ranks::OffsetVector{Int, Vector{Int}}, A::UnsafeSparseCore{T,N,d,S}) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  B = SparseCore{T,N,d}(k, row_ranks, col_ranks)

  for l in axes(B.unoccupied,1) ∩ axes(A.unoccupied,1)
    @boundscheck @assert size(A.unoccupied[l]) == (row_ranks[l],col_ranks[l])
    B.unoccupied[l] .= factor(A).*A.unoccupied[l]
  end
  for l in axes(B.occupied,1) ∩ axes(A.occupied,1)
    @boundscheck @assert size(A.occupied[l]) == (row_ranks[l],col_ranks[l+1])
    B.occupied[l] .= (jw(A) ? -factor(A) : factor(A)).*A.occupied[l]
  end
  return B
end


@propagate_inbounds function Base.copyto!(dest::SparseCore{T,N,d}, src::UnsafeSparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert axes(src.unoccupied,1) ⊆ axes(dest.unoccupied,1)
    @assert axes(src.occupied,  1) ⊆ axes(dest.occupied,  1)
  end
  for l in axes(src.unoccupied,1)
    dest.unoccupied[l] .= factor(src) .* src.unoccupied[l]
  end
  for l in axes(src.occupied,1)
    dest.occupied[l]   .= (jw(src) ? -factor(src) : factor(src)) .* src.occupied[l]
  end
  return dest
end
