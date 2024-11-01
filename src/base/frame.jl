"""
  Frame{T<:Number,N,d,S<:Side}

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

N is the total number of electrons and d is the overall tensor order; dictates structure
"""
struct Frame{T<:Number,N,d,M<:AbstractMatrix{T}} <: AbstractArray{AbstractMatrix{T},2}
  k::Int        # Framed core index
  n::Int        # column size

  qn::OffsetArrays.IdOffsetRange{Int64,UnitRange{Int64}}

  row_ranks::OffsetVector{Int, Vector{Int}}
  col_ranks::OffsetVector{Int, Vector{Int}}

  blocks::OffsetVector{M, Vector{M}}

  mem::Memory{T}

  function Frame{T,N,d}(k::Int,
                        row_ranks::OffsetVector{Int, Vector{Int}}, 
                        col_ranks::OffsetVector{Int, Vector{Int}},
                        mem::Memory{T}, offset::Int=0) where {T<:Number,N,d}
    @boundscheck begin
      N ≤ d                   || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
      1 ≤ k ≤ d+1             || throw(BoundsError())
    end

    n = min(k-1, N) - min(max(N+k-1-d, 0), N) + 1
    qn = occupation_qn(N,d,k)
    @boundscheck begin
      axes(row_ranks,1) == qn || throw(DimensionMismatch("Unexpected quantum number indices for given row ranks array"))
      axes(col_ranks,1) == qn || throw(DimensionMismatch("Unexpected quantum number indices for given column ranks array"))
    end

    sz = sum(row_ranks[n]*col_ranks[n] for n ∈ qn )
    @boundscheck begin
      @assert length(mem) ≥ offset+sz
    end

    blocks     = OffsetVector(Vector{Matrix{T}}(undef, length(qn)), qn)

    mem[1+offset:sz+offset] .= T(0)
    idx = 1+offset
    for n in qn
      blocks[n] = Block(row_ranks[n],col_ranks[n],mem,idx)
      idx += length(blocks[n])
    end

    return new{T,N,d,Matrix{T}}(k,n,qn,
                        deepcopy(row_ranks),deepcopy(col_ranks),
                        blocks, mem)
  end

  function Frame{T,N,d}(k::Int, blocks::OffsetVector{M,Vector{M}}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
    @boundscheck begin
      N ≤ d                   || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
      1 ≤ k ≤ d+1             || throw(BoundsError())
    end

    n = min(k-1, N) - min(max(N+k-1-d, 0), N) + 1
    qn = occupation_qn(N,d,k)
    @boundscheck begin
      axes(blocks,1) == qn || throw(DimensionMismatch("Unexpected quantum number indices for given blocks array"))
    end
    row_ranks = [size(blocks[n],1) for n in qn]
    col_ranks = [size(blocks[n],2) for n in qn]
    return new{T,N,d,M}(k,n,qn,row_ranks,col_ranks,blocks)
  end
end

function Frame{T,N,d}(k::Int,
                      row_ranks::OffsetVector{Int, Vector{Int}}, 
                      col_ranks::OffsetVector{Int, Vector{Int}}) where {T<:Number,N,d}
  mem = Memory{T}(undef,sum(row_ranks.*col_ranks))
  return Frame{T,N,d}(k,row_ranks,col_ranks,mem)
end


function Base.similar(A::Frame{T,N,d,M}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  return Frame{T,N,d}(A.k,A.row_ranks,A.col_ranks)
end

function Base.convert(::Type{Frame{T,N,d,Matrix{T}}}, A::Frame{T,N,d,M}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  B = Frame{T,N,d}(A.k,A.row_ranks,A.col_ranks)
  for (a,b) in zip(A.blocks, B.blocks)
    copyto!(b,a)
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
  return (A.qn, A.qn)
end

@inline function site(A::Frame)
  return A.k
end

@inline @propagate_inbounds function Base.getindex(A::Frame{T,N,d,M}, l::Int, r::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck checkbounds(A,l,r)
  if l == r
    @inbounds a = A.blocks[l]
  else
    a = M(undef, A.row_ranks[l], A.col_ranks[r])
    fill!(a, 0)
  end
  return a
end

@inline @propagate_inbounds function Base.getindex(A::Frame, n::Int)
  @boundscheck checkbounds(A,n,n)
  return A.blocks[n]
end

@inline @propagate_inbounds function Base.setindex!(A::Frame, X, l::Int, r::Int)
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
    copyto!(A.blocks[n], X)
  else
    A.blocks[n] = X
    A.row_ranks[n], A.col_ranks[n] = size(X)
  end
end

@propagate_inbounds function LinearAlgebra.lmul!(α::Number, B::Frame)
  for b in B.blocks
    lmul!(α,b)
  end
  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(A::Frame, β::Number)
  for a in A.blocks
    rmul!(a,β)
  end
  return A
end

@propagate_inbounds function LinearAlgebra.lmul!(A::Frame{T,N,d,<:AbstractMatrix{T}}, B::Frame{T,N,d,Matrix{T}}) where {T<:Number,N,d}
  @boundscheck site(A) == site(B)
  for (a,b) in zip(A.blocks, B.blocks)
    lmul!(a,b)
  end
  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(A::Frame{T,N,d,Matrix{T}}, B::Frame{T,N,d,<:AbstractMatrix{T}}) where {T<:Number,N,d}
  @boundscheck site(A) == site(B)
  for (a,b) in zip(A.blocks, B.blocks)
    rmul!(a,b)
  end
  return A
end

function LinearAlgebra.norm(A::Frame)
  return sqrt(sum(b->sum(abs2,b), A.blocks))
end


@propagate_inbounds function Base.getindex(A::Frame{T,N,d},  ::Colon, ::Colon) where {T<:Number,N,d}
  return Frame{T,N,d}(A.k, [ view(A.blocks[n],:,:) for n in axes(A,1) ] )
end

@propagate_inbounds function Base.getindex(
    A::Frame{T,N,d}, 
    I::OffsetVector{UnitRange{Int},Vector{UnitRange{Int}}},
    ::Colon) where {T<:Number,N,d}
  @boundscheck begin
    @assert axes(A,1) == axes(A,2) == axes(I,1)
    @assert all(I[n] ⊆ 1:A.row_ranks[n] for n in axes(A,1))
  end

  return Frame{T,N,d}(A.k, [ view(A.blocks[n],I[n],:) for n in axes(A,1) ])
end

@propagate_inbounds function Base.getindex(
    A::Frame{T,N,d}, 
    ::Colon,
    J::OffsetVector{UnitRange{Int},Vector{UnitRange{Int}}}) where {T<:Number,N,d}
  @boundscheck begin
    @assert axes(A,1) == axes(A,2) == axes(J,1)
    @assert all(J[n] ⊆ 1:A.col_ranks[n] for n in axes(A,2))
  end
  return Frame{T,N,d}(A.k, [ view(A.blocks[n],:,J[n]) for n in axes(A,2) ] )
end


@propagate_inbounds function Base.getindex(
    A::Frame{T,N,d}, 
    I::OffsetVector{UnitRange{Int},Vector{UnitRange{Int}}},
    J::OffsetVector{UnitRange{Int},Vector{UnitRange{Int}}}) where {T<:Number,N,d}
  @boundscheck begin
    @assert axes(A,1) == axes(A,2) == axes(I,1) == axes(J,1)
    @assert all(I[n] ⊆ 1:A.row_ranks[n] for n in axes(A,1))
    @assert all(J[n] ⊆ 1:A.col_ranks[n] for n in axes(A,3))
  end
  return Frame{T,N,d}(A.k, [ view(A.blocks[n],I[n],J[n]) for n in axes(A,1) ])
end

@propagate_inbounds function Base.copyto!(dest::Frame{T,N,d}, src::Frame{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(dest.parent) == site(src)
    @assert dest.row_ranks == src.row_ranks && dest.col_ranks == src.col_ranks
  end
  for (a,b) in zip(dest.blocks, src.blocks)
    copyto!(a,b)
  end
  return dest
end


