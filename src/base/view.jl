# Concrete SparseCoreView types

struct SparseCoreNoView{T,Nup,Ndn,d,P} <: SparseCoreView{T,Nup,Ndn,d}
  parent :: SparseCore{T,Nup,Ndn,d,P}
  function SparseCoreNoView(parent::SparseCore{T,Nup,Ndn,d,P})  where {T,Nup,Ndn,d,P<:AbstractMatrix{T}}
    return new{T,Nup,Ndn,d,P}(parent)
  end
end

struct SparseCoreRowView{T,Nup,Ndn,d,P,I} <: SparseCoreView{T,Nup,Ndn,d}
  parent :: SparseCore{T,Nup,Ndn,d,P}
  i      :: Matrix{I}
  function SparseCoreRowView(parent::SparseCore{T,Nup,Ndn,d,P}, i::Matrix{I}) where {T,Nup,Ndn,d,P<:AbstractMatrix{T},I}
    return new{T,Nup,Ndn,d,P,I}(parent,i)
  end
end

struct SparseCoreColView{T,Nup,Ndn,d,P,J} <: SparseCoreView{T,Nup,Ndn,d}
  parent :: SparseCore{T,Nup,Ndn,d,P}
  j      :: Matrix{J}
  function SparseCoreColView(parent::SparseCore{T,Nup,Ndn,d,P}, j::Matrix{J}) where {T,Nup,Ndn,d,P<:AbstractMatrix{T},J}
    return new{T,Nup,Ndn,d,P,J}(parent,j)
  end
end

struct SparseCoreBlockView{T,Nup,Ndn,d,P,I,J} <: SparseCoreView{T,Nup,Ndn,d}
  parent :: SparseCore{T,Nup,Ndn,d,P}
  i      :: Matrix{I}
  j      :: Matrix{J}
  function SparseCoreBlockView(parent::SparseCore{T,Nup,Ndn,d,P}, i::Matrix{I}, j::Matrix{J}) where {T,Nup,Ndn,d,P<:AbstractMatrix{T},I,J}
    return new{T,Nup,Ndn,d,P,I,J}(parent,i,j)
  end
end

# view implementation

@inline @propagate_inbounds function Base.view(A::SparseCore, I...)
  return A[I...]
end

@inline @propagate_inbounds function Base.getindex(
  parent::SparseCore{T,Nup,Ndn,d,P}, 
  i::Matrix{I},
  ::Colon) where {T<:Number,Nup,Ndn,d,P<:AbstractMatrix{T},I<:AbstractRange{Int}}
  @boundscheck begin
    @assert axes(i) == (1:Nup+1,1:Ndn+1)
    @assert issetequal(findall(idx->isassigned(i,idx), keys(i)), row_qn(parent))
    @assert all( i[nup,ndn] ⊆ 1:row_rank(parent,nup,ndn) for (nup,ndn) in row_qn(parent) )
  end
  return SparseCoreRowView(parent, i)
end

function Base.getindex(
  parent::SparseCore{T,Nup,Ndn,d,P}, 
  ::Colon,
  j::Matrix{J}) where {T<:Number,Nup,Ndn,d,P<:AbstractMatrix{T},J<:AbstractRange{Int}}
  @boundscheck begin
    @assert axes(j) == (Nup+1,Ndn+1)
    @assert issetequal(findall(idx->isassigned(j,idx), keys(j)), col_qn(parent))
    @assert all( j[n] ⊆ 1:col_rank(parent,nup,ndn) for (nup,ndn) in col_qn(parent) )
  end
  return SparseCoreColView(parent, j)
end

function Base.getindex(
  parent::SparseCore{T,Nup,Ndn,d,P}, 
  i::Matrix{I},
  j::Matrix{J}) where {T<:Number,Nup,Ndn,d,P<:AbstractMatrix{T},I<:AbstractRange{Int},J<:AbstractRange{Int}}

  @boundscheck begin
    @assert axes(i) == axes(j) == (Nup+1,Ndn+1)
    @assert issetequal(findall(idx->isassigned(i,idx), keys(i)), row_qn(parent))
    @assert issetequal(findall(idx->isassigned(j,idx), keys(j)), col_qn(parent))
    @assert all( i[nup,ndn] ⊆ 1:row_rank(parent,nup,ndn) for (nup,ndn) in row_qn(parent) )
    @assert all( j[nup,ndn] ⊆ 1:col_rank(parent,nup,ndn) for (nup,ndn) in col_qn(parent) )
  end
  return SparseCoreBlockView(parent,i,j)
end

function Base.getindex(
  parent::SparseCore{T,Nup,Ndn,d,P}, 
  ::Colon, ::Colon) where {T<:Number,Nup,Ndn,d,P<:AbstractMatrix{T}}
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

@inline function row_qn(A::SparseCoreView)
  return row_qn(parent(A))
end
@inline function col_qn(A::SparseCoreView)
  return col_qn(parent(A))
end

@inline function row_ranks(A::SparseCoreView)
  ranks = zeros(Int,Nup+1,Ndn+1)
  for (nup,ndn) in row_qn(A)
    ranks[nup,ndn] = row_rank(A,nup,ndn)
  end
  return ranks
end
@inline function col_ranks(A::SparseCoreColView)
  ranks = zeros(Int,Nup+1,Ndn+1)
  for (nup,ndn) in col_qn(A)
    ranks[nup,ndn] = col_rank(A,nup,ndn)
  end
  return ranks
end

@inline function row_rank(A::SparseCoreNoView,    lup::Int, ldn::Int)
  @boundscheck @assert (lup,ldn) in row_qn(A)
  return row_rank(parent(A),lup,ldn)
end
@inline function row_rank(A::SparseCoreRowView,   lup::Int, ldn::Int)
  @boundscheck @assert (lup,ldn) in row_qn(A)
  return length(A.i[lup,ldn])
end
@inline function row_rank(A::SparseCoreColView,   lup::Int, ldn::Int)
  @boundscheck @assert (lup,ldn) in row_qn(A)
  return row_rank(parent(A),lup,ldn)
end
@inline function row_rank(A::SparseCoreBlockView, lup::Int, ldn::Int)
  @boundscheck @assert (lup,ldn) in row_qn(A)
  return length(A.i[lup,ldn])
end

@inline function col_rank(A::SparseCoreNoView,    rup::Int, rdn::Int)
  @boundscheck @assert (rup,rdn) in col_qn(A)
  return col_rank(parent(A),rup,rdn)
end
@inline function col_rank(A::SparseCoreRowView,   rup::Int, rdn::Int)
  @boundscheck @assert (rup,rdn) in col_qn(A)
  return col_rank(parent(A),rup,rdn)
end
@inline function col_rank(A::SparseCoreColView,   rup::Int, rdn::Int)
  @boundscheck @assert (rup,rdn) in col_qn(A)
  return length(A.j[rup,rdn])
end
@inline function col_rank(A::SparseCoreBlockView, rup::Int, rdn::Int)
  @boundscheck @assert (rup,rdn) in col_qn(A)
  return length(A.j[rup,rdn])
end

@inline @propagate_inbounds function ○○(A::SparseCoreNoView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn) in col_qn(A)
  return view(○○(parent(A),nup, ndn),:,:)
end
@inline @propagate_inbounds function ○○(A::SparseCoreRowView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn) in col_qn(A)
  return view(○○(parent(A),nup, ndn),A.i[nup, ndn],:)
end
@inline @propagate_inbounds function ○○(A::SparseCoreColView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn) in col_qn(A)
  return view(○○(parent(A),nup, ndn),:, A.j[nup, ndn])
end
@inline @propagate_inbounds function ○○(A::SparseCoreBlockView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn) in col_qn(A)
  return view(○○(parent(A),nup, ndn),A.i[nup, ndn], A.j[nup, ndn])
end

@inline @propagate_inbounds function up(A::SparseCoreNoView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup+1,ndn) in col_qn(A)
  return view(up(parent(A),nup,ndn),:,:)
end
@inline @propagate_inbounds function up(A::SparseCoreRowView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup+1,ndn) in col_qn(A)
  return view(up(parent(A),nup,ndn),A.i[nup,ndn],:)
end
@inline @propagate_inbounds function up(A::SparseCoreColView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup+1,ndn) in col_qn(A)
  return view(up(parent(A),nup,ndn),:, A.j[nup+1,ndn])
end
@inline @propagate_inbounds function up(A::SparseCoreBlockView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup+1,ndn) in col_qn(A)
  return view(up(parent(A),nup,ndn),A.i[nup,ndn], A.j[nup+1,ndn])
end

@inline @propagate_inbounds function dn(A::SparseCoreNoView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn+1) in col_qn(A)
  return view(dn(parent(A),nup,ndn),:,:)
end
@inline @propagate_inbounds function dn(A::SparseCoreRowView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn+1) in col_qn(A)
  return view(dn(parent(A),nup,ndn),A.i[nup,ndn],:)
end
@inline @propagate_inbounds function dn(A::SparseCoreColView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn+1) in col_qn(A)
  return view(dn(parent(A),nup,ndn),:, A.j[nup,ndn+1])
end
@inline @propagate_inbounds function dn(A::SparseCoreBlockView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn+1) in col_qn(A)
  return view(dn(parent(A),nup,ndn),A.i[nup,ndn], A.j[nup,ndn+1])
end

@inline @propagate_inbounds function ●●(A::SparseCoreNoView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup+1,ndn+1) in col_qn(A)
  return view(●●(parent(A),nup,ndn),:,:)
end
@inline @propagate_inbounds function ●●(A::SparseCoreRowView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup+1,ndn+1) in col_qn(A)
  return view(●●(parent(A),nup,ndn),A.i[nup,ndn],:)
end
@inline @propagate_inbounds function ●●(A::SparseCoreColView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup+1,ndn+1) in col_qn(A)
  return view(●●(parent(A),nup,ndn),:, A.j[nup+1,ndn+1])
end
@inline @propagate_inbounds function ●●(A::SparseCoreBlockView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup+1,ndn+1) in col_qn(A)
  return view(●●(parent(A),nup,ndn),A.i[nup,ndn], A.j[nup+1,ndn+1])
end

###############################################

# Concrete FrameView types

struct FrameNoView{T,Nup,Ndn,d,M} <: FrameView{T,Nup,Ndn,d}
  parent :: Frame{T,Nup,Ndn,d,M}
  function FrameNoView(parent::Frame{T,Nup,Ndn,d,M})  where {T,Nup,Ndn,d,M<:AbstractMatrix{T}}
    return new{T,Nup,Ndn,d,M}(parent)
  end
end

struct FrameRowView{T,Nup,Ndn,d,M,I} <: FrameView{T,Nup,Ndn,d}
  parent :: Frame{T,Nup,Ndn,d,M}
  i      :: Matrix{I}
  function FrameRowView(parent::Frame{T,Nup,Ndn,d,M}, i::Matrix{I}) where {T,Nup,Ndn,d,M<:AbstractMatrix{T},I}
    return new{T,Nup,Ndn,d,M,I}(parent,i)
  end
end

struct FrameColView{T,Nup,Ndn,d,M,J} <: FrameView{T,Nup,Ndn,d}
  parent :: Frame{T,Nup,Ndn,d,M}
  j      :: Matrix{J}
  function FrameColView(parent::Frame{T,Nup,Ndn,d,M}, j::Matrix{J}) where {T,Nup,Ndn,d,M<:AbstractMatrix{T},J}
    return new{T,Nup,Ndn,d,M,J}(parent,j)
  end
end

struct FrameBlockView{T,Nup,Ndn,d,M,I,J} <: FrameView{T,Nup,Ndn,d}
  parent :: Frame{T,Nup,Ndn,d,M}
  i      :: Matrix{I}
  j      :: Matrix{J}
  function FrameBlockView(parent::Frame{T,Nup,Ndn,d,M}, i::Matrix{I}, j::Matrix{J}) where {T,Nup,Ndn,d,M<:AbstractMatrix{T},I,J}
    return new{T,Nup,Ndn,d,M,I,J}(parent,i,j)
  end
end


# view implementation

@inline @propagate_inbounds function Base.view(A::Frame, I...)
  return A[I...]
end

@inline @propagate_inbounds function Base.getindex(
  parent::Frame, 
  i::Matrix{I},
  ::Colon) where {I<:AbstractRange{Int}}
  @boundscheck begin
    @assert size(i) == (Nup+1,Ndn+1)
    @assert issetequal(findall(idx->isassigned(i,idx), keys(i)), qn(parent))
    @assert all( i[nup,ndn] ⊆ 1:row_rank(parent,nup,ndn) for (nup,ndn) in qn(parent) )
  end
  return FrameRowView(parent, i)
end

function Base.getindex(
  parent::Frame, 
  ::Colon,
  j::Matrix{J}) where {J<:AbstractRange{Int}}
  @boundscheck begin
    @assert size(j) == (Nup+1,Ndn+1)
    @assert issetequal(findall(idx->isassigned(j,idx), keys(j)), qn(parent))
    @assert all( j[nup,ndn] ⊆ 1:col_rank(parent,nup,ndn) for (nup,ndn) in qn(parent) )
  end
  return FrameColView(parent, j)
end

function Base.getindex(
  parent::Frame, 
  i::Matrix{I},
  j::Matrix{J}) where {I<:AbstractRange{Int},J<:AbstractRange{Int}}

  @boundscheck begin
    @assert size(i) == size(j) == (Nup+1,Ndn+1)
    @assert issetequal(findall(idx->isassigned(i,idx), keys(i)), qn(parent)) 
    @assert issetequal(findall(idx->isassigned(j,idx), keys(j)), qn(parent))
    @assert all( i[nup,ndn] ⊆ 1:row_rank(parent,nup,ndn) for (nup,ndn) in qn(parent) )
    @assert all( j[nup,ndn] ⊆ 1:col_rank(parent,nup,ndn) for (nup,ndn) in qn(parent) )
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

@inline function qn(A::FrameView)
  return qn(parent(A))
end
@inline function site(A::FrameView)
  return site(parent(A))
end

@inline @propagate_inbounds function block(A::FrameNoView,    nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn) in col_qn(A)
  return view(block(parent(A),nup,ndn),:,:)
end
@inline @propagate_inbounds function block(A::FrameRowView,   nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn) in col_qn(A)
  return view(block(parent(A),nup,ndn), A.i[nup,ndn],:)
end
@inline @propagate_inbounds function block(A::FrameColView,   nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn) in col_qn(A)
  return view(block(parent(A),nup,ndn),:,A.j[nup,ndn])
end
@inline @propagate_inbounds function block(A::FrameBlockView, nup::Int, ndn::Int)
  @boundscheck @assert (nup,ndn) in row_qn(A) && (nup,ndn) in col_qn(A)
  return view(block(parent(A),nup,ndn), A.i[nup,ndn], A.j[nup,ndn])
end

@propagate_inbounds function Base.getindex(A::FrameView, l::Int, r::Int)
  @boundscheck checkbounds(A,l,r)
  if l==r
    return block(A,l)
  else
    throw(BoundsError(A, (l,r)))
  end
end
