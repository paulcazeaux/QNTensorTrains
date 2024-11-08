abstract type SparseCoreView{T,N,d} <: AbstractCore{T,N,d} end

# Concrete SparseCoreView types

struct SparseCoreNoView{T,N,d,P} <: SparseCoreView{T,N,d}
  parent::SparseCore{T,N,d,P}
  function SparseCoreNoView(parent::SparseCore{T,N,d,P})  where {T,N,d,P<:AbstractMatrix{T}}
    return new{T,N,d,P}(parent)
  end
end

struct SparseCoreRowView{T,N,d,P,I} <: SparseCoreView{T,N,d}
  parent::SparseCore{T,N,d,P}
  i::OffsetVector{I, Vector{I}}
  function SparseCoreRowView(parent::SparseCore{T,N,d,P}, i::OffsetVector{I, Vector{I}}) where {T,N,d,P<:AbstractMatrix{T},I}
    return new{T,N,d,P,I}(parent,i)
  end
end

struct SparseCoreColView{T,N,d,P,J} <: SparseCoreView{T,N,d}
  parent::SparseCore{T,N,d,P}
  j::OffsetVector{J, Vector{J}}
  function SparseCoreColView(parent::SparseCore{T,N,d,P}, j::OffsetVector{J, Vector{J}}) where {T,N,d,P<:AbstractMatrix{T},J}
    return new{T,N,d,P,J}(parent,j)
  end
end

struct SparseCoreBlockView{T,N,d,P,I,J} <: SparseCoreView{T,N,d}
  parent::SparseCore{T,N,d,P}
  i::OffsetVector{I, Vector{I}}
  j::OffsetVector{J, Vector{J}}
  function SparseCoreBlockView(parent::SparseCore{T,N,d,P}, i::OffsetVector{I, Vector{I}}, j::OffsetVector{J, Vector{J}}) where {T,N,d,P<:AbstractMatrix{T},I,J}
    return new{T,N,d,P,I,J}(parent,i,j)
  end
end

# view implementation

@inline @propagate_inbounds function Base.view(A::SparseCore, I...)
  return A[I...]
end

@inline @propagate_inbounds function Base.getindex(
  parent::SparseCore{T,N,d,P}, 
  i::OffsetVector{I,Vector{I}},
  ::Colon) where {T<:Number,N,d,P<:AbstractMatrix{T},I<:AbstractRange{Int}}
  @boundscheck begin
    @assert axes(i,1) == parent.row_qn
    @assert all( i[n] ⊆ 1:parent.row_ranks[n] for n in parent.row_qn )
  end
  return SparseCoreRowView(parent, i)
end

function Base.getindex(
  parent::SparseCore{T,N,d,P}, 
  ::Colon,
  j::OffsetVector{J,Vector{J}}) where {T<:Number,N,d,P<:AbstractMatrix{T},J<:AbstractRange{Int}}
  @boundscheck begin
    @assert axes(j,1) == parent.col_qn
    @assert all( j[n] ⊆ 1:parent.col_ranks[n] for n in parent.col_qn )
  end
  return SparseCoreColView(parent, j)
end

function Base.getindex(
  parent::SparseCore{T,N,d,P}, 
  i::OffsetVector{I,Vector{I}},
  j::OffsetVector{J,Vector{J}}) where {T<:Number,N,d,P<:AbstractMatrix{T},I<:AbstractRange{Int},J<:AbstractRange{Int}}

  @boundscheck begin
    @assert axes(i,1) == parent.row_qn
    @assert axes(j,1) == parent.col_qn
    @assert all( i[n] ⊆ 1:parent.row_ranks[n] for n in parent.row_qn )
    @assert all( j[n] ⊆ 1:parent.col_ranks[n] for n in parent.col_qn )
  end
  return SparseCoreBlockView(parent,i,j)
end

function Base.getindex(
  parent::SparseCore{T,N,d,P}, 
  ::Colon, ::Colon) where {T<:Number,N,d,P<:AbstractMatrix{T}}
  return SparseCoreNoView(parent)
end

# Basic functionality

@inline function parent(A::SparseCoreNoView)
  return A.parent
end
@inline function parent(A::SparseCoreRowView)
  return A.parent
end
@inline function parent(A::SparseCoreColView)
  return A.parent
end
@inline function parent(A::SparseCoreBlockView)
  return A.parent
end

@inline function Base.size(A::SparseCoreView)
  return size(parent(A))
end

@inline function Base.axes(A::SparseCoreView)
  return axes(parent(A))
end

@inline function Base.length(A::SparseCoreView)
  return length(parent(A))
end

@inline function site(A::SparseCoreView)
  return site(parent(A))
end

@inline function row_ranks(A::SparseCoreNoView)
  return row_ranks(parent(A))
end
@inline function row_ranks(A::SparseCoreRowView)
  return length.(A.i)
end
@inline function row_ranks(A::SparseCoreColView)
  return row_ranks(parent(A))
end
@inline function row_ranks(A::SparseCoreBlockView)
  return length.(A.i)
end

@inline function col_ranks(A::SparseCoreNoView)
  return col_ranks(parent(A))
end
@inline function col_ranks(A::SparseCoreRowView)
  return col_ranks(parent(A))
end
@inline function col_ranks(A::SparseCoreColView)
  return length.(A.j)
end
@inline function col_ranks(A::SparseCoreBlockView)
  return length.(A.j)
end

@inline function row_rank(A::SparseCoreNoView, r::Int)
  return row_rank(parent(A),r)
end
@inline function row_rank(A::SparseCoreRowView, l::Int)
  return length(A.i[l])
end
@inline function row_rank(A::SparseCoreColView, r::Int)
  return row_rank(parent(A),r)
end
@inline function row_rank(A::SparseCoreBlockView, l::Int)
  return length(A.i[l])
end

@inline function col_rank(A::SparseCoreNoView, l::Int)
  return col_rank(parent(A),l)
end
@inline function col_rank(A::SparseCoreRowView, l::Int)
  return col_rank(parent(A),l)
end
@inline function col_rank(A::SparseCoreColView, r::Int)
  return length(A.j[r])
end
@inline function col_rank(A::SparseCoreBlockView, r::Int)
  return length(A.j[r])
end

@propagate_inbounds function Base.getindex(A::SparseCoreView, l::Int, s::Int, r::Int)
  @boundscheck checkbounds(A, l,s,r)

  if s==1 && r == l # Unoccupied state
    return unoccupied(A,l)
  elseif s==2 && l+1 == r # Occupied state
    return occupied(A,l)
  else # Forbidden
    throw(BoundsError(A, (l,s,r)))
  end
end


@inline @propagate_inbounds function unoccupied(A::SparseCoreNoView, l::Int)
  return view(unoccupied(parent(A),l),:,:)
end
@inline @propagate_inbounds function unoccupied(A::SparseCoreRowView, l::Int)
  return view(unoccupied(parent(A),l),A.i[l],:)
end
@inline @propagate_inbounds function unoccupied(A::SparseCoreColView, l::Int)
  return view(unoccupied(parent(A),l),:, A.j[l])
end
@inline @propagate_inbounds function unoccupied(A::SparseCoreBlockView, l::Int)
  return view(unoccupied(parent(A),l),A.i[l], A.j[l])
end

@inline @propagate_inbounds function occupied(A::SparseCoreNoView, l::Int)
  return view(occupied(parent(A),l),:,:)
end
@inline @propagate_inbounds function occupied(A::SparseCoreRowView, l::Int)
  return view(occupied(parent(A),l),A.i[l],:)
end
@inline @propagate_inbounds function occupied(A::SparseCoreColView, l::Int)
  return view(occupied(parent(A),l),:, A.j[l+1])
end
@inline @propagate_inbounds function occupied(A::SparseCoreBlockView, l::Int)
  return view(occupied(parent(A),l),A.i[l], A.j[l+1])
end



###############################################

abstract type FrameView{T,N,d} <: AbstractFrame{T,N,d} end

# Concrete FrameView types

struct FrameNoView{T,N,d,M} <: FrameView{T,N,d}
  parent::Frame{T,N,d,M}
  function FrameNoView(parent::Frame{T,N,d,M})  where {T,N,d,M<:AbstractMatrix{T}}
    return new{T,N,d,M}(parent)
  end
end

struct FrameRowView{T,N,d,M,I} <: FrameView{T,N,d}
  parent::Frame{T,N,d,M}
  i::OffsetVector{I, Vector{I}}
  function FrameRowView(parent::Frame{T,N,d,M}, i::OffsetVector{I, Vector{I}}) where {T,N,d,M<:AbstractMatrix{T},I}
    return new{T,N,d,M,I}(parent,i)
  end
end

struct FrameColView{T,N,d,M,J} <: FrameView{T,N,d}
  parent::Frame{T,N,d,M}
  j::OffsetVector{J, Vector{J}}
  function FrameColView(parent::Frame{T,N,d,M}, j::OffsetVector{J, Vector{J}}) where {T,N,d,M<:AbstractMatrix{T},J}
    return new{T,N,d,M,J}(parent,j)
  end
end

struct FrameBlockView{T,N,d,M,I,J} <: FrameView{T,N,d}
  parent::Frame{T,N,d,M}
  i::OffsetVector{I, Vector{I}}
  j::OffsetVector{J, Vector{J}}
  function FrameBlockView(parent::Frame{T,N,d,M}, i::OffsetVector{I, Vector{I}}, j::OffsetVector{J, Vector{J}}) where {T,N,d,M<:AbstractMatrix{T},I,J}
    return new{T,N,d,M,I,J}(parent,i,j)
  end
end


# view implementation

@inline @propagate_inbounds function Base.view(A::Frame, I...)
  return A[I...]
end

@inline @propagate_inbounds function Base.getindex(
  parent::Frame, 
  i::OffsetVector{I,Vector{I}},
  ::Colon) where {I<:AbstractRange{Int}}
  @boundscheck begin
    @assert axes(i,1) == parent.qn
    @assert all( i[n] ⊆ 1:parent.row_ranks[n] for n in parent.qn )
  end
  return FrameRowView(parent, i)
end

function Base.getindex(
  parent::Frame, 
  ::Colon,
  j::OffsetVector{J,Vector{J}}) where {J<:AbstractRange{Int}}
  @boundscheck begin
    @assert axes(j,1) == parent.qn
    @assert all( j[n] ⊆ 1:parent.col_ranks[n] for n in parent.qn )
  end
  return FrameColView(parent, j)
end

function Base.getindex(
  parent::Frame, 
  i::OffsetVector{I,Vector{I}},
  j::OffsetVector{J,Vector{J}}) where {I<:AbstractRange{Int},J<:AbstractRange{Int}}

  @boundscheck begin
    @assert axes(i,1) == parent.qn
    @assert axes(j,1) == parent.qn
    @assert all( i[n] ⊆ 1:parent.row_ranks[n] for n in parent.qn )
    @assert all( j[n] ⊆ 1:parent.col_ranks[n] for n in parent.qn )
  end
  return FrameBlockView(parent,i,j)
end

function Base.getindex(
  parent::Frame, 
  ::Colon, ::Colon)
  return FrameNoView(parent)
end

# Array implementation

@inline function parent(A::FrameNoView)
  return A.parent
end
@inline function parent(A::FrameRowView)
  return A.parent
end
@inline function parent(A::FrameColView)
  return A.parent
end
@inline function parent(A::FrameBlockView)
  return A.parent
end

@inline function Base.size(A::FrameView)
  return size(parent(A))
end

@inline function Base.axes(A::FrameView)
  return axes(parent(A))
end

@inline function Base.length(A::FrameView)
  return length(parent(A))
end

@inline function site(A::FrameView)
  return site(parent(A))
end

@inline @propagate_inbounds function block(A::FrameNoView, l::Int)
  return view(block(parent(A),l),:,:)
end
@inline @propagate_inbounds function block(A::FrameRowView, l::Int)
  return view(block(parent(A),l),A.i[l],:)
end
@inline @propagate_inbounds function block(A::FrameColView, l::Int)
  return view(block(parent(A),l),:, A.j[l])
end
@inline @propagate_inbounds function block(A::FrameBlockView, l::Int)
  return view(block(parent(A),l),A.i[l], A.j[l])
end

@propagate_inbounds function Base.getindex(A::FrameView, l::Int, r::Int)
  @boundscheck checkbounds(A,l,r)
  if l==r
    return block(A,l)
  else
    throw(BoundsError(A, (l,r)))
  end
end
