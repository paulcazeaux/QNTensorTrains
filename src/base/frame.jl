
"""
  Frame{T<:Number,Nup,Ndn,d,S<:Side}

Special block diagonal sparse structure obtained by 
* left-to-right framing:
    <--
    |
    F
    |
    --> k

or right-to-left  framing:
      <--
        |
        F
        |
  k-1 -->

Nup, Ndn are the total number of spin up/down electrons and d is the overall tensor order; dictates structure
"""
struct Frame{T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}} <: AbstractFrame{T,Nup,Ndn,d}
  k         :: Int        # Framed core index
  n         :: Int        # column size

  qn        :: Vector{Tuple{Int,Int}}

  row_ranks :: Matrix{Int}
  col_ranks :: Matrix{Int}

  blocks    :: Matrix{M}

  mem       :: Memory{T}

  function Frame{T,Nup,Ndn,d}(k::Int,
                        row_ranks::Matrix{Int}, 
                        col_ranks::Matrix{Int},
                        mem::Memory{T}, offset::Int=0) where {T<:Number,Nup,Ndn,d}
    @boundscheck begin
      0≤Nup≤d && 0≤Ndn≤d || throw(DimensionMismatch("Total number of electrons per spin Nup=$Nup,Ndn=$Ndn cannot be larger than dimension $d"))
      1 ≤ k ≤ d+1 || throw(BoundsError())
    end

    qn = state_qn(Nup,Ndn,d,k)
    n = length(qn)

    @boundscheck begin
      @assert size(row_ranks) == size(col_ranks) == (Nup+1,Ndn+1)
      findall(row_ranks.>0) ⊆ CartesianIndex.(qn) || throw(DimensionMismatch("Unexpected quantum number indices for given row ranks array"))
      findall(col_ranks.>0) ⊆ CartesianIndex.(qn) || throw(DimensionMismatch("Unexpected quantum number indices for given column ranks array"))
    end

    sz = sum(row_ranks[nup,ndn]*col_ranks[nup,ndn] for (nup,ndn) in qn )

    @boundscheck begin
      @assert length(mem) ≥ offset+sz
    end

    blocks     = Matrix{Matrix{T}}(undef, Nup+1,Ndn+1)

    mem[1+offset:sz+offset] .= T(0)
    idx = 1+offset
    for (nup,ndn) in qn
      blocks[nup,ndn] = Block(row_ranks[nup,ndn],col_ranks[nup,ndn],mem,idx)
      idx += length(blocks[nup,ndn])
    end

    return new{T,Nup,Ndn,d,Matrix{T}}(k,n,qn,
                        deepcopy(row_ranks),deepcopy(col_ranks),
                        blocks, mem)
  end

  function Frame{T,Nup,Ndn,d}(k::Int, blocks::Matrix{M}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
    @boundscheck begin
      0≤Nup≤d && 0≤Ndn≤d || throw(DimensionMismatch("Total number of electrons per spin Nup=$Nup,Ndn=$Ndn cannot be larger than dimension $d"))
      1 ≤ k ≤ d+1             || throw(BoundsError())
    end

    qn = state_qn(Nup,Ndn,d,k)
    n = length(qn)

    @boundscheck begin
      @assert size(blocks) == (Nup+1,Ndn+1) 
      issetequal(findall(idx->isassigned(blocks,idx), keys(blocks)), CartesianIndex.(qn)) || throw(DimensionMismatch("Unexpected quantum number indices for given blocks array"))
    end
    row_ranks = zeros(Int,Nup+1,Ndn+1)
    col_ranks = zeros(Int,Nup+1,Ndn+1)
    for (nup,ndn) in qn
      row_ranks[nup,ndn], col_ranks[nup,ndn] = size(blocks[nup,ndn])
    end

    return new{T,Nup,Ndn,d,M}(k,n,qn,row_ranks,col_ranks,blocks)
  end
end

function Frame{T,Nup,Ndn,d}(k::Int,
                      row_ranks::Matrix{Int}, 
                      col_ranks::Matrix{Int}) where {T<:Number,Nup,Ndn,d}
  mem = Memory{T}(undef,sum(row_ranks.*col_ranks))
  return Frame{T,Nup,Ndn,d}(k,row_ranks,col_ranks,mem)
end


function Base.similar(A::Frame{T,Nup,Ndn,d,M}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  return Frame{T,Nup,Ndn,d}(A.k,A.row_ranks,A.col_ranks)
end

function Base.convert(::Type{Frame{T,Nup,Ndn,d,Matrix{T}}}, A::Frame{T,Nup,Ndn,d,M}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  B = Frame{T,Nup,Ndn,d}(A.k,A.row_ranks,A.col_ranks)
  for (nup,ndn) in qn(A)
    block(B,nup,ndn) .= block(A,nup,ndn)
  end
  return B
end

@inline function Base.size(A::Frame)
  return (A.n, A.n)
end

@inline function Base.length(A::Frame)
  return A.n*A.n
end

@inline function Base.axes(A::Frame)
  return (1:A.n, 1:A.n)
end

@inline function site(A::Frame)
  return A.k
end

@inline function qn(A::Frame)
  return A.qn
end

@inline function row_ranks(A::Frame)
  return A.row_ranks
end
@inline function row_rank(A::Frame,l::Int)
  return A.row_ranks[l]
end
@inline function col_ranks(A::Frame)
  return A.col_ranks
end
@inline function col_rank(A::Frame, r::Int)
  return A.col_ranks[r]
end

@inline function blocks(A::Frame)
  return A.blocks
end

@inline function block(A::Frame, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in qn(A)
  return blocks(A)[nup,ndn]
end
@inline function block(A::Frame, qn::Tuple{Int,Int})
  return block(A,qn[1],qn[2])
end
@inline function block(A::Frame, l::Int)
  return block(A,qn(A)[l])
end

@inline @propagate_inbounds function Base.getindex(A::AbstractFrame, l::Tuple{Int,Int}, r::Tuple{Int,Int})
  @boundscheck begin
    @assert l in qn(A) && r in qn(A)
    l==r || throw(BoundsError(A, (l,r)))
  end
  return @inbounds block(A,l)
end

@inline @propagate_inbounds function Base.getindex(A::AbstractFrame, l::Int, r::Int)
  @boundscheck begin
    checkbounds(A,l,r)
    l==r || throw(BoundsError(A, (l,r)))
  end
  return @inbounds block(A,l)
end

@inline @propagate_inbounds function Base.getindex(A::AbstractFrame, n::Int)
  @boundscheck checkbounds(A,n,n)
  return block(A,n)
end

@inline @propagate_inbounds function Base.setindex!(A::AbstractFrame, X, l::Int, r::Int)
  if r == l # Diagonal state
    A[l] = X
  else # Forbidden
    throw(BoundsError(A, (l,r)))
  end
end

@inline @propagate_inbounds function Base.setindex!(A::Frame{T}, X::M, n::Int) where {T<:Number,M<:AbstractMatrix{T}}
  @boundscheck begin
    checkbounds(A, n, n)
    !isdefined(A, :mem) || size(X) == (A.row_ranks[n],A.col_ranks[n]) || 
      throw(DimensionMismatch("Trying to assign block of size $(size(X)) to a block of prescribed ranks $((A.row_ranks[n], A.col_ranks[n]))"))
  end
  if isdefined(A, :mem)
    copyto!(block(A,n), X)
  else
    A.blocks[n] = X
    A.row_ranks[n], A.col_ranks[n] = size(X)
  end
end

@propagate_inbounds function LinearAlgebra.lmul!(α::Number, B::Frame)
  for (nup,ndn) in qn(B)
    lmul!(α, block(B,nup,ndn))
  end
  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(A::Frame, β::Number)
  for (nup,ndn) in qn(A)
    rmul!(block(A,nup,ndn), β)
  end
  return A
end

@propagate_inbounds function LinearAlgebra.lmul!(A::Frame{T,Nup,Ndn,d,<:AbstractMatrix{T}}, B::Frame{T,Nup,Ndn,d,Matrix{T}}) where {T<:Number,Nup,Ndn,d}
  @boundscheck site(A) == site(B)
  for (nup,ndn) in qn(A)
    lmul!(block(A,nup,ndn), block(B,nup,ndn))
  end
  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(A::Frame{T,Nup,Ndn,d,Matrix{T}}, B::Frame{T,Nup,Ndn,d,<:AbstractMatrix{T}}) where {T<:Number,Nup,Ndn,d}
  @boundscheck site(A) == site(B)
  for (nup,ndn) in qn(A)
    rmul!(block(A,nup,ndn), block(B,nup,ndn))
  end
  return A
end

function LinearAlgebra.norm(A::Frame)
  return sqrt(sum(sum(abs2,block(A,qn)) for qn in qn(A)))
end

@propagate_inbounds function Base.copyto!(dest::Frame{T,Nup,Ndn,d}, src::Frame{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert site(dest.parent) == site(src)
    @assert dest.row_ranks == src.row_ranks && dest.col_ranks == src.col_ranks
  end
  for (nup,ndn) in qn(A)
    copyto!(block(A,nup,ndn), block(B,nup,ndn))
  end
  return dest
end


