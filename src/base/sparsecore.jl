function state_qn(Nup::Int,Ndn::Int, d::Int, k::Int)
  @boundscheck begin
    1 ≤ k ≤ d+1 || throw(BoundsError())
    0≤Nup≤d && 0≤Ndn≤d || throw(DimensionMismatch("Total number of electrons per spin Nup=$Nup,Ndn=$Ndn cannot be larger than dimension $d"))
  end
  return [ (nup+1,ndn+1) for nup=0:Nup, ndn=0:Ndn if max(Nup+(k-1-d),0) ≤ nup ≤ min(k-1,Nup) && max(Ndn+(k-1-d),0) ≤ ndn ≤ min(k-1,Ndn)]
end

function in_state_qn(nup::Int, ndn::Int, Nup::Int,Ndn::Int, d::Int, k::Int)
  return max(Nup+(k-1-d),0) ≤ nup-1 ≤ min(k-1,Nup) && max(Ndn+(k-1-d),0) ≤ ndn-1 ≤ min(k-1,Ndn)
end

"""
  SparseCore{T<:Number,Nup,Ndn,d,S<:AbstractMatrix{T}}

Special bidiagonal sparse structure,
where diagonal  correspond to modal index = 1, 
and  just above the diagonal correspond to modal index 2

Nup is the total number of spin-upelectrons, Ndn number of spin-down electrons, and d is the overall tensor order; dictates structure
"""
struct SparseCore{T,Nup,Ndn,d,S<:AbstractMatrix{T}} <: AbstractCore{T,Nup,Ndn,d}
  k         :: Int        # Core index
  m         :: Int        # row size
  n         :: Int        # column size

  row_qn    :: Vector{Tuple{Int,Int}}
  col_qn    :: Vector{Tuple{Int,Int}}

  row_ranks :: Matrix{Int}
  col_ranks :: Matrix{Int}

  ○○        :: Matrix{S}
  up        :: Matrix{S}
  dn        :: Matrix{S}
  ●●        :: Matrix{S}

  mem       :: Memory{T}

  # Main constructor - single memory allocation
  function SparseCore{T,Nup,Ndn,d}(k::Int, row_ranks::Matrix{Int}, col_ranks::Matrix{Int},
                                mem::Memory{T}, offset::Int=0) where {T<:Number,Nup,Ndn,d}
    @boundscheck begin
      1 ≤ k ≤ d          || throw(BoundsError())
      0≤Nup≤d && 0≤Ndn≤d || throw(DimensionMismatch("Total number of electrons per spin Nup=$Nup,Ndn=$Ndn cannot be larger than dimension $d"))
    end

    row_qn = state_qn(Nup,Ndn,d,k)
    col_qn = state_qn(Nup,Ndn,d,k+1)

    m = length(row_qn)
    n = length(col_qn)

    @boundscheck begin
      size(row_ranks) == size(col_ranks) == (Nup+1,Ndn+1)
      findall(row_ranks.>0) ⊆ CartesianIndex.(row_qn) || throw(DimensionMismatch("Unexpected quantum number indices for given row ranks array"))
      findall(col_ranks.>0) ⊆ CartesianIndex.(col_qn) || throw(DimensionMismatch("Unexpected quantum number indices for given column ranks array"))
    end

    sz = sum(row_ranks[nup,ndn]*col_ranks[nup  ,ndn  ]   for (nup,ndn) in it_○○(row_qn,col_qn); init=0 ) +
         sum(row_ranks[nup,ndn]*col_ranks[nup+1,ndn  ]   for (nup,ndn) in it_up(row_qn,col_qn); init=0 ) +
         sum(row_ranks[nup,ndn]*col_ranks[nup  ,ndn+1]   for (nup,ndn) in it_dn(row_qn,col_qn); init=0 ) +
         sum(row_ranks[nup,ndn]*col_ranks[nup+1,ndn+1]   for (nup,ndn) in it_●●(row_qn,col_qn); init=0 )

    @boundscheck begin
      @assert length(mem) ≥ offset+sz
    end

    ○○ = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
    up = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
    dn = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
    ●● = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)

    mem[1+offset:sz+offset] .= T(0)
    idx = 1+offset

    for (nup,ndn) in row_qn
      (nup  ,ndn  ) in col_qn && ( ○○[nup,ndn] = Block(row_ranks[nup,ndn],col_ranks[nup  ,ndn  ],mem,idx); idx += length(○○[nup,ndn]) )
      (nup+1,ndn  ) in col_qn && ( up[nup,ndn] = Block(row_ranks[nup,ndn],col_ranks[nup+1,ndn  ],mem,idx); idx += length(up[nup,ndn]) )
      (nup  ,ndn+1) in col_qn && ( dn[nup,ndn] = Block(row_ranks[nup,ndn],col_ranks[nup  ,ndn+1],mem,idx); idx += length(dn[nup,ndn]) )
      (nup+1,ndn+1) in col_qn && ( ●●[nup,ndn] = Block(row_ranks[nup,ndn],col_ranks[nup+1,ndn+1],mem,idx); idx += length(●●[nup,ndn]) )
    end
    return new{T,Nup,Ndn,d,Matrix{T}}(k,m,n,row_qn,col_qn,
                        deepcopy(row_ranks),deepcopy(col_ranks),
                        ○○, up, dn, ●●, mem)
  end

  # Partial initialization without memory field when fully populated block arrays are provided
  function SparseCore{T,Nup,Ndn,d}(k::Int, ○○::Matrix{S}, up::Matrix{S}, dn::Matrix{S}, ●●::Matrix{S}) where {T<:Number,Nup,Ndn,d,S<:AbstractMatrix{T}}
    @boundscheck begin
      N ≤ d                       || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
      1 ≤ k ≤ d                   || throw(BoundsError())
    end
    m = length(row_qn)
    n = length(col_qn)

    row_qn = state_qn(Nup,Ndn,d,k)
    col_qn = state_qn(Nup,Ndn,d,k+1)
    @boundscheck begin
      @assert size(○○) == size(up) == size(dn) == size(●●) == (Nup+1,Ndn+1)
      @assert issetequal(findall(i->isassigned(○○,i), keys(○○)), CartesianIndex.(collect(it_○○(row_qn,col_qn))))
      @assert issetequal(findall(i->isassigned(up,i), keys(up)), CartesianIndex.(collect(it_up(row_qn,col_qn))))
      @assert issetequal(findall(i->isassigned(dn,i), keys(dn)), CartesianIndex.(collect(it_dn(row_qn,col_qn))))
      @assert issetequal(findall(i->isassigned(●●,i), keys(●●)), CartesianIndex.(collect(it_●●(row_qn,col_qn))))
    end
    row_ranks = zeros(Int,Nup+1,Ndn+1)
    col_ranks = zeros(Int,Nup+1,Ndn+1)
    for (nup,ndn) in row_qn
      if (nup,ndn) in col_qn
        @boundscheck begin
          (nup+1,ndn  ) in col_qn && @assert size(○○[nup,ndn],1)==size(up[nup,ndn],1)
          (nup  ,ndn+1) in col_qn && @assert size(○○[nup,ndn],1)==size(dn[nup,ndn],1)
          (nup+1,ndn+1) in col_qn && @assert size(○○[nup,ndn],1)==size(●●[nup,ndn],1)
        end
        row_ranks[nup,ndn] = size(○○[nup,ndn],1)
      elseif (nup+1,ndn) in col_qn
        @boundscheck begin
          (nup  ,ndn+1) in col_qn && @assert size(up[nup,ndn],1)==size(dn[nup,ndn],1)
          (nup+1,ndn+1) in col_qn && @assert size(up[nup,ndn],1)==size(●●[nup,ndn],1)
        end
        row_ranks[nup,ndn] = size(up[nup,ndn],1)
      elseif (nup,ndn+1) in col_qn
        @boundscheck begin
          (nup+1,ndn+1) in col_qn && @assert size(dn[nup,ndn],1)==size(●●[nup,ndn],1)
        end
        row_ranks[nup,ndn] = size(up[nup,ndn],1)
      elseif (nup+1,ndn+1) in col_qn
        row_ranks[nup,ndn] = size(●●[nup,ndn],1)
      end
    end

    for (nup,ndn) in col_qn
      if (nup,ndn) in row_qn
        @boundscheck begin
          (nup-1,ndn  ) in col_qn && @assert size(○○[nup  ,ndn  ],2)==size(up[nup-1,ndn  ],2)
          (nup  ,ndn-1) in col_qn && @assert size(○○[nup  ,ndn  ],2)==size(dn[nup  ,ndn-1],2)
          (nup-1,ndn-1) in col_qn && @assert size(○○[nup  ,ndn  ],2)==size(●●[nup-1,ndn-1],2)
        end
        col_ranks[nup,ndn] = size(○○[nup,ndn],2)
      elseif (nup-1,ndn) in row_qn
        @boundscheck begin
          (nup  ,ndn+1) in col_qn && @assert size(up[nup-1,ndn  ],2)==size(dn[nup  ,ndn-1],2)
          (nup+1,ndn+1) in col_qn && @assert size(up[nup-1,ndn  ],2)==size(●●[nup-1,ndn-1],2)
        end
        col_ranks[nup,ndn] = size(up[nup-1,ndn],2)
      elseif (nup,ndn-1) in row_qn
        @boundscheck begin
          (nup-1,ndn-1) in col_qn && @assert size(dn[nup  ,ndn-1],2)==size(●●[nup-1,ndn-1],2)
        end
        col_ranks[nup,ndn] = size(up[nup,ndn-1],2)
      elseif (nup-1,ndn-1) in row_qn
        col_ranks[nup,ndn] = size(●●[nup-1,ndn-1],2)
      end
    end
    return new{T,Nup,Ndn,d,S}(k, m, n, row_qn, col_qn, row_ranks, col_ranks, ○○, up, dn, ●●)
  end
end

function SparseCore{T,Nup,Ndn,d}(k::Int, row_ranks::Matrix{Int}, 
                                      col_ranks::Matrix{Int}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert size(row_ranks) == size(col_ranks) == (Nup+1,Ndn+1)
  end

  row_qn = state_qn(Nup,Ndn,d,k)
  col_qn = state_qn(Nup,Ndn,d,k+1)

  sz = sum(row_ranks[nup,ndn]*col_ranks[nup  ,ndn  ]   for (nup,ndn) in it_○○(row_qn,col_qn); init=0 ) +
       sum(row_ranks[nup,ndn]*col_ranks[nup+1,ndn  ]   for (nup,ndn) in it_up(row_qn,col_qn); init=0 ) +
       sum(row_ranks[nup,ndn]*col_ranks[nup  ,ndn+1]   for (nup,ndn) in it_dn(row_qn,col_qn); init=0 ) +
       sum(row_ranks[nup,ndn]*col_ranks[nup+1,ndn+1]   for (nup,ndn) in it_●●(row_qn,col_qn); init=0 )

  mem = Memory{T}(undef,sz)
  return SparseCore{T,Nup,Ndn,d}(k,row_ranks,col_ranks,mem)
end

@inline function in_row_qn(nup::Int, ndn::Int, A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return in_state_qn(nup,ndn,Nup,Ndn,d,A.k)
end
@inline function in_col_qn(nup::Int, ndn::Int, A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return in_state_qn(nup,ndn,Nup,Ndn,d,A.k+1)
end

@inline function it_○○(A)
  return Iterators.filter(qn->in_col_qn(qn[1],qn[2],A), row_qn(A))
end
@inline function it_○○(row_qn::Vector{Tuple{Int,Int}}, col_qn::Vector{Tuple{Int,Int}})
  return Iterators.filter(in(col_qn), row_qn)
end
@inline function it_up(A)
  return Iterators.filter(qn->in_col_qn(qn[1]+1,qn[2],A), row_qn(A))
end
@inline function it_up(row_qn::Vector{Tuple{Int,Int}}, col_qn::Vector{Tuple{Int,Int}})
  return Iterators.filter(qn->(qn[1]+1,qn[2]) in col_qn, row_qn)
end
@inline function it_dn(A)
  return Iterators.filter(qn->in_col_qn(qn[1],qn[2]+1,A), row_qn(A))
end
@inline function it_dn(row_qn::Vector{Tuple{Int,Int}}, col_qn::Vector{Tuple{Int,Int}})
  return Iterators.filter(qn->(qn[1],qn[2]+1) in col_qn, row_qn)
end
@inline function it_●●(A)
  return Iterators.filter(qn->in_col_qn(qn[1]+1,qn[2]+1,A), row_qn(A))
end
@inline function it_●●(row_qn::Vector{Tuple{Int,Int}}, col_qn::Vector{Tuple{Int,Int}})
  return Iterators.filter(qn->(qn[1]+1,qn[2]+1) in col_qn, row_qn)
end

@inline function Block(row_rank::Int,col_rank::Int,mem::Memory{T},idx::Int) where T<:Number
  return row_rank>0 && col_rank>0 ? Base.wrap(Array, memoryref(mem,idx), (row_rank, col_rank)) : zeros(T, row_rank, col_rank)
end

"""
  SparseCore{T,Nup,Ndn,d}(sup::Bool, nup::Int, sdn::Bool, ndn::Int, k::Int)

Core initialization for a pure, non-entangled state
  where sup,sdn ∈ {true, false} denotes occupation or not of the corresponding spin state,
  nup,ndn are the number of occupied up/down states on cores 1...k-1,
  N is the total number of particles
"""
function SparseCore{T,Nup,Ndn,d}(sup::Bool, nup::Int, sdn::Bool, ndn::Int, k::Int) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    1 ≤ k ≤ d       || throw(BoundsError())
    0≤Nup≤d && 0≤Ndn≤d || throw(DimensionMismatch("Total number of electrons per spin Nup=$Nup,Ndn=$Ndn cannot be larger than dimension $d"))
    0≤nup≤min(k-1,Nup) && 0≤ndn≤min(k-1,Ndn) || throw(DimensionMismatch("Number of occupied states on cores 1...$(k-1) cannot be larger than $(k-1)"))
    Nup-nup-sup ≤ (d-k)     || throw(DimensionMismatch("Number of occupied up-spin states on cores $(k+1)...$d cannot be larger than $(d-k)"))
    Ndn-ndn-sdn ≤ (d-k)     || throw(DimensionMismatch("Number of occupied down-spin states on cores $(k+1)...$d cannot be larger than $(d-k)"))
  end

  row_qn = state_qn(Nup,Ndn,d,k)
  col_qn = state_qn(Nup,Ndn,d,k+1)
  nup,ndn = nup+1,ndn+1

  row_ranks = zeros(Int,Nup+1,Ndn+1)
  col_ranks = zeros(Int,Nup+1,Ndn+1)
  row_ranks[nup    ,ndn    ] = 1
  col_ranks[nup+sup,ndn+sdn] = 1
  new_core = SparseCore{T,Nup,Ndn,d}(k, row_ranks, col_ranks)
  if sup && sdn
    new_core.●●[nup,ndn] .= 1
  elseif sup
    new_core.up[nup,ndn] .= 1
  elseif sdn
    new_core.dn[nup,ndn] .= 1
  else
    new_core.○○[nup,ndn] .= 1
  end

  return new_core
end

@inline function Base.similar(A::SparseCore{T,Nup,Ndn,d,Matrix{T}}) where {T<:Number,Nup,Ndn,d}
  return SparseCore{T,Nup,Ndn,d}(site(A),row_ranks(A),col_ranks(A))
end

@inline function Base.size(A::SparseCore)
  return (A.m, 4, A.n)
end

@inline function Base.axes(A::SparseCore)
  return (1:A.m, 1:4, 1:A.n)
end

@inline function Base.length(A::SparseCore)
  return A.m*4*A.n
end

@inline function site(A::SparseCore)
  return A.k
end

@inline function row_qn(A::SparseCore)
  return A.row_qn
end
@inline function row_qn(A::SparseCore, qn::Int)
  return A.row_qn[qn]
end

@inline function col_qn(A::SparseCore)
  return A.col_qn
end
@inline function col_qn(A::SparseCore, qn::Int)
  return A.col_qn[qn]
end

@inline function ○○(A::SparseCore)
  return A.○○
end
@inline function ○○(A::AbstractCore, nup::Int, ndn::Int)
  @boundscheck @assert in_row_qn(nup,ndn,A) && in_col_qn(nup,ndn,A)
  return ○○(A)[nup,ndn]
end
@inline function ○○(A::AbstractCore, qn::Tuple{Int,Int})
  return ○○(A,qn[1],qn[2])
end
@inline function ○○(A::AbstractCore, qn::Int)
  return ○○(A, row_qn(A)[qn])
end

@inline function up(A::SparseCore)
  return A.up
end
@inline function up(A::AbstractCore, nup::Int, ndn::Int)
  @boundscheck @assert in_row_qn(nup,ndn,A) && in_col_qn(nup+1,ndn,A)
  return up(A)[nup,ndn]
end
@inline function up(A::AbstractCore, qn::Tuple{Int,Int})
  return up(A,qn[1],qn[2])
end
@inline function up(A::AbstractCore, qn::Int)
  return up(A, row_qn(A,qn))
end

@inline function dn(A::SparseCore)
  return A.dn
end
@inline function dn(A::AbstractCore, nup::Int, ndn::Int)
  @boundscheck @assert in_row_qn(nup,ndn,A) && in_col_qn(nup,ndn+1,A)
  return dn(A)[nup,ndn]
end
@inline function dn(A::AbstractCore, qn::Tuple{Int,Int})
  return dn(A,qn[1],qn[2])
end
@inline function dn(A::AbstractCore, qn::Int)
  return dn(A, row_qn(A,qn))
end

@inline function ●●(A::SparseCore)
  return A.●●
end
@inline function ●●(A::AbstractCore, nup::Int, ndn::Int)
  @boundscheck @assert in_row_qn(nup,ndn,A) && in_col_qn(nup+1,ndn+1,A)
  return ●●(A)[nup,ndn]
end
@inline function ●●(A::AbstractCore, qn::Tuple{Int,Int})
  return ●●(A,qn[1],qn[2])
end
@inline function ●●(A::AbstractCore, qn::Int)
  return ●●(A, row_qn(A,qn))
end

@inline function row_ranks(A::SparseCore)
  return A.row_ranks
end
@inline function row_rank(A::SparseCore, nup::Int, ndn::Int)
  return row_ranks(A)[nup,ndn]
end
@inline function row_rank(A::AbstractCore, qn::Tuple{Int,Int})
  return row_rank(A,qn[1],qn[2])
end
@inline function row_rank(A::AbstractCore, qn::Int)
  return row_rank(A, row_qn(A,qn))
end

@inline function col_ranks(A::SparseCore)
  return A.col_ranks
end
@inline function col_rank(A::SparseCore, nup::Int, ndn::Int)
  return col_ranks(A)[nup,ndn]
end
@inline function col_rank(A::AbstractCore, qn::Tuple{Int,Int})
  return col_rank(A,qn[1],qn[2])
end
@inline function col_rank(A::AbstractCore, qn::Int)
  return col_rank(A, col_qn(A,qn))
end

@propagate_inbounds function Base.getindex(A::AbstractCore, l::Int, s::Int, r::Int)
  @boundscheck checkbounds(A, l,s,r)

  if s==1 && row_qn(A,l) == col_qn(A,r)            # Unoccupied state
    return ○○(A,l)
  elseif s==2 && row_qn(A,l).+(1,0) == col_qn(A,r) # Up-spin state
    return up(A,l)
  elseif s==3 && row_qn(A,l).+(0,1) == col_qn(A,r) # Down-spin state
    return dn(A,l)
  elseif s==4 && row_qn(A,l).+(1,1) == col_qn(A,r) # Doubly occupied state
    return ●●(A,l)
  else # Forbidden
    throw(BoundsError(A, (l,s,r)))
  end
end

@propagate_inbounds function Base.getindex(A::AbstractCore, l::Int, r::Int)
  return A[row_qn(A,l),col_qn(A,r)]
end

@propagate_inbounds function Base.getindex(A::AbstractCore, l::Tuple{Int,Int}, r::Tuple{Int,Int})
  @boundscheck @assert in_row_qn(l...,A) && in_col_qn(r..., A)
  lup,ldn = l; rup,rdn = r
  if lup==rup && ldn==rdn
    return ○○(A,lup,ldn)
  elseif lup+1==rup && ldn==rdn
    return up(A,lup,ldn)
  elseif lup==rup && ldn+1==rdn
    return dn(A,lup,ldn)
  elseif lup+1==rup && ldn+1==rdn
    return ●●(A,lup,ldn)
  else
    throw(BoundsError(A, (l,r)))
  end
end

@propagate_inbounds function Base.getindex(A::AbstractCore, n::Int, unfolding::Symbol)
  @assert unfolding == :horizontal || unfolding == :vertical ||
          unfolding == :R          || unfolding == :L

  if unfolding == :horizontal || unfolding == :R
    return A[row_qn(A,n), :horizontal]
  else # unfolding == :vertical || unfolding == :L
    return A[col_qn(A,n), :vertical]
  end
end

@propagate_inbounds function Base.getindex(A::AbstractCore, qn::Tuple{Int,Int}, unfolding::Symbol)
  @assert unfolding == :horizontal || unfolding == :vertical ||
          unfolding == :R          || unfolding == :L

  if unfolding == :horizontal || unfolding == :R
    lup, ldn = qn
    @boundscheck @assert in_row_qn(lup,ldn,A) || throw(BoundsError(A, (qn,:,:)))
    return reduce(hcat, A[(lup,ldn),(rup,rdn)] for (rup,rdn) in ( (lup,ldn), (lup+1,ldn), (lup,ldn+1), (lup+1,ldn+1) ) if in_col_qn(rup,rdn,A))
  else # unfolding == :vertical || unfolding == :L
    rup,rdn = qn
    @boundscheck @assert in_col_qn(rup,rdn,A) || throw(BoundsError(A, (:,:,qn)))
    return reduce(vcat, A[(lup,ldn),(rup,rdn)] for (lup,ldn) in ( (rup,rdn), (rup-1,rdn), (rup,rdn-1), (rup-1,rdn-1) ) if in_row_qn(lup,ldn,A))
  end
end

@propagate_inbounds function Base.setindex!(A::SparseCore{T}, X::S, l::Int,r::Int) where {T<:Number, S<:AbstractMatrix{T}}
  Base.setindex!(A,X,row_qn(A,l),col_qn(A,r))
end

@propagate_inbounds function Base.setindex!(A::SparseCore{T}, X::S, l::Tuple{Int,Int}, r::Tuple{Int,Int}) where {T<:Number, S<:AbstractMatrix{T}}
  lup,ldn = l; rup,rdn = r
  @boundscheck begin
    @assert in_row_qn(lup,ldn,A) && in_row_qn(rup,rdn,A)
    size(X) == (row_rank(A,l),col_rank(A,r)) || 
      throw(DimensionMismatch("Trying to assign block of size $(size(X)) to a block of prescribed ranks $((row_rank(A,l), col_rank(A,r)))"))
  end

  if lup==rup && ldn==rdn
    copyto!(○○(A,lup,ldn), X)
  elseif lup+1==rup && ldn==rdn
    copyto!(up(A,lup,ldn), X)
  elseif lup==rup && ldn+1==rdn
    copyto!(dn(A,lup,ldn), X)
  elseif lup+1==rup && ldn+1==rdn
    copyto!(●●(A,lup,ldn), X)
  else
    throw(BoundsError(A, (l,r)))
  end
end

@propagate_inbounds function Base.setindex!(A::SparseCore{T}, X::S, l::Int,s::Int,r::Int) where {T<:Number, S<:AbstractMatrix{T}}
  @boundscheck begin
    checkbounds(A, l,s,r)
    size(X) == (row_rank(A,l),col_rank(A,r)) || 
      throw(DimensionMismatch("Trying to assign block of size $(size(X)) to a block of prescribed ranks $((row_rank(A,l), col_rank(A,r)))"))
  end

  if s==1 && row_qn(A,l) == col_qn(A,r)            # Unoccupied state
    copyto!(○○(A,l), X)
  elseif s==2 && row_qn(A,l) == col_qn(A,r).+(1,0) # Up-spin state
    copyto!(up(A,l), X)
  elseif s==3 && row_qn(A,l) == col_qn(A,r).+(0,1) # Down-spin state
    copyto!(dn(A,l), X)
  elseif s==4 && row_qn(A,l) == col_qn(A,r).+(1,1) # Doubly occupied state
    copyto!(●●(A,l), X)
  else # Forbidden
    throw(BoundsError(A, (l,s,r)))
  end
end

@propagate_inbounds function Base.setindex!(A::SparseCore{T,Nup,Ndn,d,Matrix{T}}, X::S, n::Int, unfolding::Symbol) where {T<:Number,Nup,Ndn,d,S<:AbstractMatrix{T}}
  @assert unfolding == :horizontal || unfolding == :vertical ||
          unfolding == :R          || unfolding == :L

  if unfolding == :horizontal || unfolding == :R
    return setindex!(A, X, row_qn(A,n), :horizontal)
  elseif unfolding == :vertical || unfolding == :L
    return setindex!(A, X, col_qn(A,n), :vertical)
  else
    throw(BoundsError(A, unfolding))
  end
end

@propagate_inbounds function Base.setindex!(A::SparseCore{T,Nup,Ndn,d,Matrix{T}}, X::S, qn::Tuple{Int,Int}, unfolding::Symbol) where {T<:Number,Nup,Ndn,d,S<:AbstractMatrix{T}}
  @assert unfolding == :horizontal || unfolding == :vertical ||
          unfolding == :R          || unfolding == :L

  if unfolding == :horizontal || unfolding == :R
    lup,ldn = qn
    @boundscheck @assert in_row_qn(lup,ldn,A) || throw(BoundsError(A, (qn,:,:)))
    R = collect(r for r in ( (lup,ldn), (lup+1,ldn), (lup,ldn+1), (lup+1,ldn+1) ) if r ∈ col_qn(A))
    @boundscheck begin
      @assert size(X,1) == row_rank(A,qn)
      @assert size(X,2) == sum(col_rank(A,r) for r in R)
    end
    Rs = cumsum(col_rank(A,r) for r in R)
    for (i,r) in enumerate(R)
      copyto!(A[qn,r], view(X, :, (i==1 ? 1 : Rs[i-1]+1):Rs[i]))
    end
  elseif unfolding == :vertical || unfolding == :L
    rup,rdn = qn
    @boundscheck @assert in_col_qn(rup,rdn,A) || throw(BoundsError(A, (:,:,qn)))
    L = collect(l for l in ( (rup,rdn), (rup-1,rdn), (rup,rdn-1), (rup-1,rdn-1) ) if l ∈ row_qn(A))
    @boundscheck begin
      @assert size(X,1) == sum(row_rank(A,l) for l in L)
      @assert size(X,2) == col_rank(A,qn)
    end
    Ls = cumsum(row_rank(A,l) for l in L)
    for (i,l) in enumerate(L)
      copyto!(A[l,qn], view(X, (i==1 ? 1 : Ls[i-1]+1):Ls[i], :))
    end
  else
    throw(BoundsError(A, unfolding))
  end
end

@propagate_inbounds function Base.copyto!(dest::AbstractCore{T,Nup,Ndn,d}, src::AbstractCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert site(src) == site(dest)
    @assert row_ranks(dest) == row_ranks(src) && col_ranks(dest) == col_ranks(src)
  end
  for (lup,ldn) in row_qn(A)
    in_col_qn(lup  ,ldn  ,A) && copyto!(○○(dest,l),○○(src,l))
    in_col_qn(lup+1,ldn  ,A) && copyto!(up(dest,l),up(src,l))
    in_col_qn(lup  ,ldn+1,A) && copyto!(dn(dest,l),dn(src,l))
    in_col_qn(lup+1,ldn+1,A) && copyto!(●●(dest,l),(●●src,l))
  end
  return dest
end

function Base.show(io::IO, ::MIME"text/plain", A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  if get(io, :compact, true)
    str = "SparseCore{$T,$N,$d} with $(A.m)x2x$(A.n) block shape"
  else
    # Manage some formatting and padding
    strr = ["r[$(qn)]=$(row_ranks(A)[qn])" for qn in axes(A,1)]
    len = max(length.(strr)...)
    padr = len .- length.(strr)
    strr_rows = ["$(s)"*" "^(pad+len+3)          for (s,pad) in zip(strr, padr)]

    strr = ["r[$(qn)]=$(col_ranks(A)[qn])" for qn in axes(A,3)]
    len = max(length.(strr)...)
    padr = len .- length.(strr)
    strr_cols = ["$(s)"*" "^(pad+len+3)          for (s,pad) in zip(strr, padr)]

    str = string("SparseCore{$T,$N,$d} with $(A.m)x2x$(A.n) block shape and index $(site(A)).",
      "\nRow ranks are ", strr_rows..., 
      "\nColumn ranks are ", strr_cols)
  end
  print(io, str)
end

# function Base.show(io::IO, A::SparseCore)

#     (size(A,1) == 0 || size(A,3) == 0) && return show(io, MIME("text/plain"), A)
#     row_str = [" $(qn)‹$(row_ranks(A)[qn]) " for qn in axes(A,1)]
#     col_str = [" $(qn)‹$(col_ranks(A)[qn]) " for qn in axes(A,3)]

#     rw = maximum(length.(row_str))
#     cw = maximum(length.(col_str))
#     cpad(s,w) = rpad(lpad(s, div(w-length(s),2)+length(s)),w)
#     row_str .= cpad.(row_str, rw)
#     col_str .= cpad.(col_str, cw)

#     Grid = fill(UInt16(10240), size(A,3)*(cw+1)+2+rw, 2size(A,1))
#     Grid[rw+1,2:end] .= '⎢'
#     Grid[end-1,2:end] .= '⎥'
#     if size(A,1)>1
#       Grid[rw+1,2] = '⎡'
#       Grid[rw+1,end] = '⎣'
#       Grid[end-1,2] = '⎤'
#       Grid[end-1,end] = '⎦'
#     else
#       Grid[rw+1,2] = '['
#       Grid[end-1,end] = ']'
#     end
#     Grid[end, :] .= '\n'

#     for (j,n) in enumerate(axes(A,1))
#       Grid[1:rw,2j]       .= collect(row_str[n])
#     end
#     for (i,n) in enumerate(axes(A,3))
#       Grid[rw+1+(cw+1)*(i-1).+(1:cw),1]       .= collect(col_str[n])
#     end

#     for (j,n) in enumerate(axes(A,1))
#       if n ∈ axes(A,3)
#         i = findfirst(isequal(n), OffsetArrays.no_offset_view(axes(A,3)))
#         if (row_rank(A,n) > 0) && (col_rank(A,n) > 0)
#           Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j]   .= collect(cpad('○',cw))
#         end
#         if n+1 ∈ axes(A,3)
#           Grid[rw+1+(cw+1)*i,2j] = '⎢'
#           if n+1 ∈ axes(A,1)
#             Grid[rw+1+(cw+1)*i,2j+1] = '+'
#           end
#         end
#         if n-1 ∈ axes(A,3)
#           Grid[rw+1+(cw+1)*(i-1),2j] = '⎢'
#         end
#         if n+1 ∈ axes(A,1)
#           Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j+1] .= collect(cpad('⎯',cw))
#         end
#       end
#       if n+1 ∈ axes(A,3)
#         i = findfirst(isequal(n+1), OffsetArrays.no_offset_view(axes(A,3)))
#         if (row_rank(A,n) > 0) && (col_ranks(A)[n+1] > 0)
#           Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j]   .= collect(cpad('●',cw))
#         end
#         if n+1 ∈ axes(A,1)
#           Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j+1]       .= collect(cpad('⎯',cw))
#           if n+2 ∈ axes(A,3)
#             Grid[rw+1+(cw+1)*i,2j+1] = '+'
#           end
#         end
#         if n-1 ∈ axes(A,1)
#           Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j-1]       .= collect(cpad('⎯',cw))
#         end
#         if n+2 ∈ axes(A,3)
#           Grid[rw+1+(cw+1)*i,2j] = '⎢'
#         end
#       end
#     end

#     foreach(c -> print(io, Char(c)), @view Grid[1:end-1])
# end

function Base.summary(io::IO, A::SparseCore)
  show(io, summary(A))
end

function Base.summary(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return string("[$(axes(A,1)[begin]):$(axes(A,1)[end])]x$(size(A,2))x[$(axes(A,3)[begin]):$(axes(A,3)[end])] SparseCore{$(T),$(N),$(d)})")
end


struct AdjointCore{T<:Number,Nup,Ndn,d,C<:AbstractCore{T,Nup,Ndn,d}} <: AbstractAdjointCore{T,Nup,Ndn,d}
  parent::C
end

@inline function LinearAlgebra.adjoint(A::C) where {T<:Number,Nup,Ndn,d,C<:AbstractCore{T,Nup,Ndn,d}}
  return AdjointCore{T,Nup,Ndn,d,C}(A)
end

@inline function LinearAlgebra.adjoint(A::AdjointCore)
  return parent(A)
end

@inline function Base.parent(A::AdjointCore)
  return A.parent
end

@inline function Base.size(A::AdjointCore)
  return reverse(size(parent(A)))
end

@inline function Base.size(A::AdjointCore, i)
  return size(A)[i]
end

@inline function Base.length(A::AdjointCore)
  return length(parent(A))
end

@inline function Base.axes(A::AdjointCore)
  return reverse(axes(parent(A)))
end

@inline function Base.axes(A::AdjointCore, i)
  return axes(A)[i]
end

@inline function site(A::AdjointCore)
  return site(parent(A))
end

@inline function ○○(A::AdjointCore, nup::Int, ndn::Int)
  return adjoint(○○(parent(A), nup, ndn))
end
@inline function ○○(A::AdjointCore, r::Int)
  return ○○(A, row_qn(A, r))
end

@inline function up(A::AdjointCore, nup::Int, ndn::Int)
  return adjoint(up(parent(A), nup-1, ndn))
end
@inline function up(A::AdjointCore, r::Int)
  return up(A, row_qn(A, r))
end

@inline function dn(A::AdjointCore, nup::Int, ndn::Int)
  return adjoint(dn(parent(A), nup, ndn-1))
end
@inline function dn(A::AdjointCore, r::Int)
  return dn(A, row_qn(A, r))
end

@inline function ●●(A::AdjointCore, nup::Int, ndn::Int)
  return adjoint(●●(parent(A), nup-1, ndn-1))
end
@inline function ●●(A::AdjointCore, r::Int)
  return ●●(A, row_qn(A, r))
end

@inline function row_qn(A::AdjointCore)
  return col_qn(parent(A))
end
@inline function col_qn(A::AdjointCore)
  return row_qn(parent(A))
end

@inline function row_ranks(A::AdjointCore)
  return col_ranks(parent(A))
end
@inline function row_rank(A::AdjointCore, r...)
  return col_rank(parent(A), r...)
end

@inline function col_ranks(A::AdjointCore)
  return row_ranks(parent(A))
end
@inline function col_rank(A::AdjointCore, l...)
  return row_rank(parent(A), l...)
end

@inline @propagate_inbounds function Base.getindex(A::AdjointCore, I, J)
  return adjoint(parent(A)[J,I])
end

@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,Nup,Ndn,d}, A::SparseCore{T,Nup,Ndn,d}, b::Number) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert row_ranks(C) == row_ranks(A)
  @boundscheck @assert col_ranks(C) == col_ranks(A)

  for (lup,ldn) in row_qn(A)
    in_col_qn(lup  ,ldn  ,A) && mul!(○○(C,lup,ldn), T(b), ○○(A,lup,ldn))
    in_col_qn(lup+1,ldn  ,A) && mul!(up(C,lup,ldn), T(b), up(A,lup,ldn))
    in_col_qn(lup  ,ldn+1,A) && mul!(dn(C,lup,ldn), T(b), dn(A,lup,ldn))
    in_col_qn(lup+1,ldn+1,A) && mul!(●●(C,lup,ldn), T(b), ●●(A,lup,ldn))
  end
  return C
end

@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,Nup,Ndn,d}, a::Number, B::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return mul!(C,B,a)
end

function Base.:*(a::Number, B::SparseCore)
  return lmul!(a,deepcopy(B))
end

Base.:*(A::SparseCore, b::Number) = b*A

function Base.:+(A::SparseCore)
  return deepcopy(A)
end

function Base.:-(A::SparseCore)
  return lmul!(-1, deepcopy(A))
end

@propagate_inbounds function LinearAlgebra.lmul!(a::Number, B::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  for (lup,ldn) in row_qn(B)
    in_col_qn(lup  ,ldn  ,B) && lmul!(T(a), ○○(B,lup,ldn))
    in_col_qn(lup+1,ldn  ,B) && lmul!(T(a), up(B,lup,ldn))
    in_col_qn(lup  ,ldn+1,B) && lmul!(T(a), dn(B,lup,ldn))
    in_col_qn(lup+1,ldn+1,B) && lmul!(T(a), ●●(B,lup,ldn))
  end

  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(B::SparseCore{T,Nup,Ndn,d}, a::Number) where {T<:Number,Nup,Ndn,d}
  return lmul!(a,B)
end

@propagate_inbounds function LinearAlgebra.lmul!(A::AbstractMatrix{T}, B::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert site(B) == 1
    @assert size(A,1) == size(A,2) == row_rank(B,0,0)
  end

  for (lup,ldn) in row_qn(B)
    in_col_qn(lup  ,ldn  ,B) && lmul!(A, ○○(B,lup,ldn))
    in_col_qn(lup+1,ldn  ,B) && lmul!(A, up(B,lup,ldn))
    in_col_qn(lup  ,ldn+1,B) && lmul!(A, dn(B,lup,ldn))
    in_col_qn(lup+1,ldn+1,B) && lmul!(A, ●●(B,lup,ldn))
  end
  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(A::SparseCore{T,Nup,Ndn,d}, B::Mat) where {T<:Number,Nup,Ndn,d,Mat<:AbstractMatrix{T}}
  @assert site(A) == d
  @assert col_rank(A,1) == size(B,1) == size(B,2)

  for (lup,ldn) in row_qn(A)
    in_col_qn(lup  ,ldn  ,A) && lmul!(○○(A,lup,ldn),B)
    in_col_qn(lup+1,ldn  ,A) && lmul!(up(A,lup,ldn),B)
    in_col_qn(lup  ,ldn+1,A) && lmul!(dn(A,lup,ldn),B)
    in_col_qn(lup+1,ldn+1,A) && lmul!(●●(A,lup,ldn),B)
  end
  return A
end

@propagate_inbounds function LinearAlgebra.lmul!(A::Frame{T,Nup,Ndn,d,Mat}, B::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d,Mat<:AbstractMatrix{T}}
  @boundscheck begin
    site(A) == site(B) && axes(A,1) == axes(B,1) || throw(DimensionMismatch("Axes mismatch between matrices $(summary(axis(A,1))) and core row indices $(summary(axis(B,1)))"))
    @assert row_ranks(A) == col_ranks(A) == row_ranks(B)
  end

  for (lup,ldn) in row_qn(B)
    in_col_qn(lup  ,ldn  ,B) && lmul!(block(A,lup,ldn), ○○(B,lup,ldn))
    in_col_qn(lup+1,ldn  ,B) && lmul!(block(A,lup,ldn), up(B,lup,ldn))
    in_col_qn(lup  ,ldn+1,B) && lmul!(block(A,lup,ldn), dn(B,lup,ldn))
    in_col_qn(lup+1,ldn+1,B) && lmul!(block(A,lup,ldn), ●●(B,lup,ldn))
  end

  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(A::SparseCore{T,Nup,Ndn,d}, B::Frame{T,Nup,Ndn,d,Mat}) where {T<:Number,Nup,Ndn,d,Mat<:AbstractMatrix{T}}
  @boundscheck begin
    axes(A,3) == axes(B,1)|| throw(DimensionMismatch("Axes mismatch between core column indices $(summary(axis(B,3))) and matrices $(summary(axis(A,1)))"))
    for n in axes(A,3)
      @assert col_rank(A,n) == size(B[n],1) == size(B[n],2)
    end
  end

  for (lup,ldn) in row_qn(A)
    in_col_qn(lup  ,ldn  ,A) && rmul!(○○(A,lup,ldn), block(B,lup  ,ldn  ))
    in_col_qn(lup+1,ldn  ,A) && rmul!(up(A,lup,ldn), block(B,lup+1,ldn  ))
    in_col_qn(lup  ,ldn+1,A) && rmul!(dn(A,lup,ldn), block(B,lup  ,ldn+1))
    in_col_qn(lup+1,ldn+1,A) && rmul!(●●(A,lup,ldn), block(B,lup+1,ldn+1))
  end

  return A
end

"""
LinearAlgebra.mul!(C::SparseCore, A::Frame, B::SparseCore, α=1, β=0)
      --
      |
C =   A
      |    |
      -- --B--
"""
@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,Nup,Ndn,d}, A::Frame{T,Nup,Ndn,d,M}, B::SparseCore{T,Nup,Ndn,d}, α::Number=1, β::Number=0) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  @boundscheck begin
    @assert site(A) == site(B)
    @assert row_ranks(C) == row_ranks(A)
    @assert col_ranks(A) == row_ranks(B)
    @assert col_ranks(C) == col_ranks(B)
  end

  for (lup,ldn) in row_qn(C)
    in_col_qn(lup  ,ldn  ,C) && mul!(○○(C,lup,ldn), block(A,lup,ldn), ○○(B,lup,ldn), α, β)
    in_col_qn(lup+1,ldn  ,C) && mul!(up(C,lup,ldn), block(A,lup,ldn), up(B,lup,ldn), α, β)
    in_col_qn(lup  ,ldn+1,C) && mul!(dn(C,lup,ldn), block(A,lup,ldn), dn(B,lup,ldn), α, β)
    in_col_qn(lup+1,ldn+1,C) && mul!(●●(C,lup,ldn), block(A,lup,ldn), ●●(B,lup,ldn), α, β)
  end
  return C
end


"""
LinearAlgebra.mul!(C::SparseCore, A::SparseCore, B::Frame, α=1, β=0)
           --
            | 
C =         B 
       |    | 
     --A-- --
"""
@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,Nup,Ndn,d}, A::SparseCore{T,Nup,Ndn,d}, B::Frame{T,Nup,Ndn,d,M}, α::Number=1, β::Number=0) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  @boundscheck begin
    @assert site(A)+1 == site(B)
    @assert row_ranks(C) == row_ranks(A)
    @assert col_ranks(A) == row_ranks(B)
    @assert col_ranks(C) == col_ranks(B)
  end
  for (lup,ldn) in row_qn(C)
    in_col_qn(lup  ,ldn  ,C) && mul!(○○(C,lup,ldn), ○○(A,lup,ldn), block(B,lup  ,ldn  ), α, β)
    in_col_qn(lup+1,ldn  ,C) && mul!(up(C,lup,ldn), up(A,lup,ldn), block(B,lup+1,ldn  ), α, β)
    in_col_qn(lup  ,ldn+1,C) && mul!(dn(C,lup,ldn), dn(A,lup,ldn), block(B,lup  ,ldn+1), α, β)
    in_col_qn(lup+1,ldn+1,C) && mul!(●●(C,lup,ldn), ●●(A,lup,ldn), block(B,lup+1,ldn+1), α, β)
  end
  return C
end

"""
LinearAlgebra.mul!(C::AdjointCore, A::Frame, B::AdjointCore, α=1, β=0)
    -- --A--
    |    |   
C = B 
    | 
    --
"""
@propagate_inbounds function LinearAlgebra.mul!(C::AdjointCore{T,Nup,Ndn,d}, A::AdjointCore{T,Nup,Ndn,d}, B::Frame{T,Nup,Ndn,d,Mat}, α::Number=1, β::Number=0) where {T<:Number,Nup,Ndn,d,Mat<:AbstractMatrix{T}}
  tA = parent(A)
  tC = parent(C)
  @boundscheck begin
    @assert site(A) == site(B) == site(C)
    @assert row_ranks(tC) == col_ranks(B)
    @assert row_ranks(B) == row_ranks(tA)
    @assert col_ranks(tC) == col_ranks(tA)
  end
  for (lup,ldn) in row_qn(tC)
    in_col_qn(lup  ,ldn  ,tC) && mul!(○○(tC,lup,ldn), adjoint(block(B,lup,ldn)), ○○(tA,lup,ldn), α, β)
    in_col_qn(lup+1,ldn  ,tC) && mul!(up(tC,lup,ldn), adjoint(block(B,lup,ldn)), up(tA,lup,ldn), α, β)
    in_col_qn(lup  ,ldn+1,tC) && mul!(dn(tC,lup,ldn), adjoint(block(B,lup,ldn)), dn(tA,lup,ldn), α, β)
    in_col_qn(lup+1,ldn+1,tC) && mul!(●●(tC,lup,ldn), adjoint(block(B,lup,ldn)), ●●(tA,lup,ldn), α, β)
  end
  return C
end

"""
LinearAlgebra.mul!(C::AdjointSparseCore, A::Frame, B::AdjointSparseCore, α=1, β=0)
   --B-- -
     |   |
C =      A
         |
        --
"""
@propagate_inbounds function LinearAlgebra.mul!(C::AdjointCore{T,Nup,Ndn,d}, A::Frame{T,Nup,Ndn,d,M}, B::AdjointCore{T,Nup,Ndn,d}, α::Number=1, β::Number=0) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  tB = parent(B)
  tC = parent(C)
  @boundscheck begin
    @assert site(A)-1 == site(B) == site(C)
    @assert row_ranks(tC) == row_ranks(tB)
    @assert col_ranks(tB) == col_ranks(A)
    @assert col_ranks(tC) == row_ranks(A)
  end
  for (lup,ldn) in row_qn(tC)
    in_col_qn(lup  ,ldn  ,tC) && mul!(○○(tC,lup,ldn), ○○(tB,lup,ldn), adjoint(block(A,lup  ,ldn  )), α, β)
    in_col_qn(lup+1,ldn  ,tC) && mul!(up(tC,lup,ldn), up(tB,lup,ldn), adjoint(block(A,lup+1,ldn  )), α, β)
    in_col_qn(lup  ,ldn+1,tC) && mul!(dn(tC,lup,ldn), dn(tB,lup,ldn), adjoint(block(A,lup  ,ldn+1)), α, β)
    in_col_qn(lup+1,ldn+1,tC) && mul!(●●(tC,lup,ldn), ●●(tB,lup,ldn), adjoint(block(A,lup+1,ldn+1)), α, β)
  end

  return C
end


"""
LinearAlgebra.mul!(C::Frame, A::AdjointSparseCore, B::SparseCore, α, β)
     - --A--
     |   |
C =  |   
     |   |
     ----B--
"""
@propagate_inbounds function LinearAlgebra.mul!(C::Frame{T,Nup,Ndn,d,Mat}, A::AdjointCore{T,Nup,Ndn,d}, B::SparseCore{T,Nup,Ndn,d}, α::Number, β::Number) where {T<:Number,Nup,Ndn,d,Mat<:AbstractMatrix{T}}
  tA = parent(A)
  @boundscheck begin
    @assert site(A) == site(B) == site(C)-1
    @assert row_ranks(C) == col_ranks(tA)
    @assert row_ranks(B) == row_ranks(tA)
    @assert col_ranks(C) == col_ranks(B)
  end
  check_β = zeros(Bool,Nup+1,Ndn+1)
  for (rup,rdn) in qn(C)
    in_row_qn(rup  ,rdn  ,B) && (mul!(block(C,rup,rdn), adjoint(○○(tA,rup  ,rdn  )), ○○(B,rup  ,rdn  ), α, (check_β[rup,rdn] ? 1 : β)); check_β[rup,rdn] = true)
    in_row_qn(rup-1,rdn  ,B) && (mul!(block(C,rup,rdn), adjoint(up(tA,rup-1,rdn  )), up(B,rup-1,rdn  ), α, (check_β[rup,rdn] ? 1 : β)); check_β[rup,rdn] = true)
    in_row_qn(rup  ,rdn-1,B) && (mul!(block(C,rup,rdn), adjoint(dn(tA,rup  ,rdn-1)), dn(B,rup  ,rdn-1), α, (check_β[rup,rdn] ? 1 : β)); check_β[rup,rdn] = true)
    in_row_qn(rup-1,rdn-1,B) && (mul!(block(C,rup,rdn), adjoint(●●(tA,rup-1,rdn-1)), ●●(B,rup-1,rdn-1), α, (check_β[rup,rdn] ? 1 : β)); check_β[rup,rdn] = true)
  end

  return C
end

"""
LinearAlgebra.mul!(C::Frame, A::SparseCore, B::AdjointSparseCore, α, β)
     --B-- -
       |   |
C =        |
       |   |
     --A----
"""
@propagate_inbounds function LinearAlgebra.mul!(C::Frame{T,Nup,Ndn,d,Mat}, A::SparseCore{T,Nup,Ndn,d}, B::AdjointCore{T,Nup,Ndn,d}, α::Number, β::Number) where {T<:Number,Nup,Ndn,d,Mat<:AbstractMatrix{T}}
  tB = parent(B)
  @boundscheck begin
    @assert site(C) == site(B) == site(A)
    @assert row_ranks(C) == row_ranks(A)
    @assert col_ranks(A) == col_ranks(tB)
    @assert col_ranks(C) == row_ranks(tB)
  end
  check_β = zeros(Bool,Nup+1,Ndn+1)
  for (lup,ldn) in qn(C)
    in_col_qn(lup  ,ldn  ,A) && (mul!(block(C,lup,ldn), ○○(A,lup,ldn), adjoint(○○(tB,lup,ldn)), α, (check_β[lup,ldn] ? 1 : β)); check_β[lup,ldn] = true)
    in_col_qn(lup+1,ldn  ,A) && (mul!(block(C,lup,ldn), up(A,lup,ldn), adjoint(up(tB,lup,ldn)), α, (check_β[lup,ldn] ? 1 : β)); check_β[lup,ldn] = true)
    in_col_qn(lup  ,ldn+1,A) && (mul!(block(C,lup,ldn), dn(A,lup,ldn), adjoint(dn(tB,lup,ldn)), α, (check_β[lup,ldn] ? 1 : β)); check_β[lup,ldn] = true)
    in_col_qn(lup+1,ldn+1,A) && (mul!(block(C,lup,ldn), ●●(A,lup,ldn), adjoint(●●(tB,lup,ldn)), α, (check_β[lup,ldn] ? 1 : β)); check_β[lup,ldn] = true)
  end

  return C
end

@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,Nup,Ndn,d}, A::M, B::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  @boundscheck begin
    @assert site(C) == site(B) == 1
    @assert row_ranks(C) == row_ranks(B) && col_ranks(C) == col_ranks(B)
    @assert size(A,1) == size(A,2) == row_rank(B,0)
  end

  for (lup,ldn) in row_qn(C)
    in_col_qn(lup  ,ldn  ,C) && mul!(○○(C,lup,ldn), A, ○○(B,lup,ldn))
    in_col_qn(lup+1,ldn  ,C) && mul!(up(C,lup,ldn), A, up(B,lup,ldn))
    in_col_qn(lup  ,ldn+1,C) && mul!(dn(C,lup,ldn), A, dn(B,lup,ldn))
    in_col_qn(lup+1,ldn+1,C) && mul!(●●(C,lup,ldn), A, ●●(B,lup,ldn))
  end

  return C
end

@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,Nup,Ndn,d}, A::SparseCore{T,Nup,Ndn,d}, B::M, α::Number=1, β::Number=0) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  @boundscheck begin
    @assert site(A) == site(C) == d
    @assert col_ranks(A)[N] == size(B,1)
    @assert col_ranks(C)[N] == size(B,2)
    @assert row_ranks(C) == row_ranks(A)
  end

  for (lup,ldn) in row_qn(C)
    in_col_qn(lup  ,ldn  ,C) && mul!(○○(C,lup,ldn), ○○(A,lup,ldn), B)
    in_col_qn(lup+1,ldn  ,C) && mul!(up(C,lup,ldn), up(A,lup,ldn), B)
    in_col_qn(lup  ,ldn+1,C) && mul!(dn(C,lup,ldn), dn(A,lup,ldn), B)
    in_col_qn(lup+1,ldn+1,C) && mul!(●●(C,lup,ldn), ●●(A,lup,ldn), B)
  end

  return C
end

function Base.:*(A::Frame{T,Nup,Ndn,d,M}, B::AbstractCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  return mul!(SparseCore{T,Nup,Ndn,d}(site(B), row_ranks(A), col_ranks(B)), A, B)
end

function Base.:*(A::AbstractCore{T,Nup,Ndn,d}, B::Frame{T,Nup,Ndn,d,M}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  return mul!(SparseCore{T,Nup,Ndn,d}(site(A), row_ranks(A), col_ranks(B)), A, B)
end

function Base.:*(A::AdjointCore{T,Nup,Ndn,d}, B::Frame{T,Nup,Ndn,d,M}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  return mul!(adjoint(SparseCore{T,Nup,Ndn,d}(site(A), col_ranks(B), row_ranks(A))), A, B)
end

function Base.:*(A::Frame{T,Nup,Ndn,d,M}, B::AdjointCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  return mul!(adjoint(SparseCore{T,Nup,Ndn,d}(site(B), col_ranks(B), row_ranks(A))), A, B)
end

function Base.:*(A::AdjointCore{T,Nup,Ndn,d}, B::AbstractCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return mul!(Frame{T,Nup,Ndn,d}(site(A)+1,row_ranks(A),col_ranks(B)), A, B)
end

function Base.:*(A::AbstractCore{T,Nup,Ndn,d}, B::AdjointCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return mul!(Frame{T,Nup,Ndn,d}(site(A),  row_ranks(A),col_ranks(B)), A, B)
end

function Base.:*(A::M, B::AbstractCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  @boundscheck @assert site(B) == 1 && size(A,1) == size(A,2) == row_rank(B,1)
  return mul!(similar(B), A, B)
end

function Base.:*(A::AbstractCore{T,Nup,Ndn,d}, B::M) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  @boundscheck @assert site(A) == d
  return mul!(SparseCore{T,Nup,Ndn,d}(d,row_ranks(A), [size(B,2) for n in state_qn(Nup,Ndn,d,d)]), A, B)
end


function ⊕(A::SparseCore{T,Nup,Ndn,d}, B::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert site(A) == site(B)
  k = site(A)
  if d>1
    if k==1
      @boundscheck @assert row_ranks(A) == row_ranks(B)
      rowranks = row_ranks(A)
      colranks = col_ranks(A) + col_ranks(B)
    elseif 1<k<d
      rowranks = row_ranks(A) + row_ranks(B)
      colranks = col_ranks(A) + col_ranks(B)
    else # k == d
      @boundscheck @assert col_ranks(A) == col_ranks(B)
      rowranks = row_ranks(A) + row_ranks(B)
      colranks = col_ranks(A)
    end
  else #d==1
    @boundscheck @assert row_ranks(A) == row_ranks(B)
    @boundscheck @assert col_ranks(A) == col_ranks(B)
    rowranks = row_ranks(A)
    colranks = col_ranks(A)
  end
  C = SparseCore{T,Nup,Ndn,d}(k,rowranks,colranks)
  if d>1
    if k==1
      for qn in it_○○(A)
        a,b,c = ○○(A,qn),○○(B,qn),○○(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2))     , a)
        copyto!(view(c,axes(a,1), size(a,2).+axes(b,2)), b)
      end
      for qn in it_up(A)
        a,b,c = up(A,qn),up(B,qn),up(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2))     , a)
        copyto!(view(c,axes(a,1), size(a,2).+axes(b,2)), b)
      end
      for qn in it_dn(A)
        a,b,c = dn(A,qn),dn(B,qn),dn(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2))     , a)
        copyto!(view(c,axes(a,1), size(a,2).+axes(b,2)), b)
      end
      for qn in it_●●(A)
        a,b,c = ●●(A,qn),●●(B,qn),●●(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2))     , a)
        copyto!(view(c,axes(a,1), size(a,2).+axes(b,2)), b)
      end

    elseif 1<k<d
      for qn in it_○○(A)
        a,b,c = ○○(A,qn),○○(B,qn),○○(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2))     , a)
        copyto!(view(c,size(a,1).+axes(b,1), size(a,2).+axes(b,2)), b)
      end
      for qn in it_up(A)
        a,b,c = up(A,qn),up(B,qn),up(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2))     , a)
        copyto!(view(c,size(a,1).+axes(b,1), size(a,2).+axes(b,2)), b)
      end
      for qn in it_dn(A)
        a,b,c = dn(A,qn),dn(B,qn),dn(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2))     , a)
        copyto!(view(c,size(a,1).+axes(b,1), size(a,2).+axes(b,2)), b)
      end
      for qn in it_●●(A)
        a,b,c = ●●(A,qn),●●(B,qn),●●(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2))     , a)
        copyto!(view(c,size(a,1).+axes(b,1), size(a,2).+axes(b,2)), b)
      end
    else # k == d
      for qn in it_○○(A)
        a,b,c = ○○(A,qn),○○(B,qn),○○(C,qn)
        copyto!(view(c,axes(a,1)           , axes(a,2)), a)
        copyto!(view(c,size(a,1).+axes(b,1), axes(a,2)), b)
      end
      for qn in it_up(A)
        a,b,c = up(A,qn),up(B,qn),up(C,qn)
        copyto!(view(c,axes(a,1)           , axes(a,2)), a)
        copyto!(view(c,size(a,1).+axes(b,1), axes(a,2)), b)
      end
      for qn in it_dn(A)
        a,b,c = dn(A,qn),dn(B,qn),dn(C,qn)
        copyto!(view(c,axes(a,1)           , axes(a,2)), a)
        copyto!(view(c,size(a,1).+axes(b,1), axes(a,2)), b)
      end
      for qn in it_●●(A)
        a,b,c = ●●(A,qn),●●(B,qn),●●(C,qn)
        copyto!(view(c,axes(a,1)           , axes(a,2)), a)
        copyto!(view(c,size(a,1).+axes(b,1), axes(a,2)), b)
      end
    end
  else #d==1
    for qn in it_○○(A)
      a,b,c = ○○(A,qn),○○(B,qn),○○(C,qn)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
    for qn in it_up(A)
      a,b,c = up(A,qn),up(B,qn),up(C,qn)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
    for qn in it_dn(A)
      a,b,c = dn(A,qn),dn(B,qn),dn(C,qn)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
    for qn in it_●●(A)
      a,b,c = ●●(A,qn),●●(B,qn),●●(C,qn)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
  end

  return C
end

@propagate_inbounds function LinearAlgebra.axpy!(α, V::SparseCore{T,Nup,Ndn,d}, W::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert row_ranks(V) == row_ranks(W)
  @boundscheck @assert col_ranks(V) == col_ranks(W)

  for (lup,ldn) in row_qn(V)
    in_col_qn(lup  ,ldn  ,V) && axpy!(α,○○(V,lup,ldn),○○(W,lup,ldn))
    in_col_qn(lup+1,ldn  ,V) && axpy!(α,up(V,lup,ldn),up(W,lup,ldn))
    in_col_qn(lup  ,ldn+1,V) && axpy!(α,dn(V,lup,ldn),dn(W,lup,ldn))
    in_col_qn(lup+1,ldn+1,V) && axpy!(α,●●(V,lup,ldn),●●(W,lup,ldn))
  end
  return W
end

@propagate_inbounds function LinearAlgebra.axpby!(α, V::SparseCore{T,Nup,Ndn,d}, β, W::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert row_ranks(V) == row_ranks(W)
  @boundscheck @assert col_ranks(V) == col_ranks(W)

  for (lup,ldn) in row_qn(V)
    in_col_qn(lup  ,ldn  ,V) && axpby!(α,○○(V,lup,ldn),β, ○○(W,lup,ldn))
    in_col_qn(lup+1,ldn  ,V) && axpby!(α,up(V,lup,ldn),β, up(W,lup,ldn))
    in_col_qn(lup  ,ldn+1,V) && axpby!(α,dn(V,lup,ldn),β, dn(W,lup,ldn))
    in_col_qn(lup+1,ldn+1,V) && axpby!(α,●●(V,lup,ldn),β, ●●(W,lup,ldn))
  end
  return W
end

@propagate_inbounds function LinearAlgebra.dot(V::SparseCore{T,Nup,Ndn,d}, W::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert row_ranks(V) == row_ranks(W)
  @boundscheck @assert col_ranks(V) == col_ranks(W)

  s = T(0)
  for (lup,ldn) in row_qn(V)
    in_col_qn(lup  ,ldn  ,V) && (s += dot(○○(V,lup,ldn),○○(W,lup,ldn)) )
    in_col_qn(lup+1,ldn  ,V) && (s += dot(up(V,lup,ldn),up(W,lup,ldn)) )
    in_col_qn(lup  ,ldn+1,V) && (s += dot(dn(V,lup,ldn),dn(W,lup,ldn)) )
    in_col_qn(lup+1,ldn+1,V) && (s += dot(●●(V,lup,ldn),●●(W,lup,ldn)) )
  end
  return s
end

function Base.abs2(V::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}

  s = T(0)
  for (lup,ldn) in row_qn(V)
    in_col_qn(lup  ,ldn  ,V) && (s += sum(abs2, ○○(V,lup,ldn)) )
    in_col_qn(lup+1,ldn  ,V) && (s += sum(abs2, up(V,lup,ldn)) )
    in_col_qn(lup  ,ldn+1,V) && (s += sum(abs2, dn(V,lup,ldn)) )
    in_col_qn(lup+1,ldn+1,V) && (s += sum(abs2, ●●(V,lup,ldn)) )
  end
  return s
end

function LinearAlgebra.norm(V::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return sqrt(abs2(V))
end

function ⊕(A::SparseCore{T,Nup,Ndn,d}, b::Number) where {T<:Number,Nup,Ndn,d}
  rowranks = deepcopy(row_ranks(A))
  colranks = deepcopy(col_ranks(A))

  if k>1
    for (nup,ndn) in row_qn(A)
      row_ranks[nup,ndn] += 1
    end
  end
  if k<d
    for (nup,ndn) in col_qn(A)
      col_ranks[nup,ndn] += 1
    end
  end

  C = SparseCore{T,Nup,Ndn,d}(site(A),rowranks,colranks)
  if d>1
    if k==1
      for qn in it_○○(A)
        a,b,c = ○○(A,qn),○○(B,qn),○○(C,qn)
        copyto!(view(c,:, axes(a,2)  ), a   )
        c[1,size(a,2)+1] = b
      end
      for qn in it_up(A)
        a,b,c = up(A,qn),up(B,qn),up(C,qn)
        copyto!(view(c,:, axes(a,2)  ), a   )
        c[1,size(a,2)+1] = b
      end
      for qn in it_dn(A)
        a,b,c = dn(A,qn),dn(B,qn),dn(C,qn)
        copyto!(view(c,:, axes(a,2)  ), a   )
        c[1,size(a,2)+1] = b
      end
      for qn in it_●●(A)
        a,b,c = ●●(A,qn),●●(B,qn),●●(C,qn)
        copyto!(view(c,:, axes(a,2)  ), a   )
        c[1,size(a,2)+1] = b
      end

    elseif 1<k<d
      for qn in it_○○(A)
        a,b,c = ○○(A,qn),○○(B,qn),○○(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2)), a)
        c[size(a,1)+1, size(a,2)+1] = b
      end
      for qn in it_up(A)
        a,b,c = up(A,qn),up(B,qn),up(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2)), a)
        c[size(a,1)+1, size(a,2)+1] = b
      end
      for qn in it_dn(A)
        a,b,c = dn(A,qn),dn(B,qn),dn(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2)), a)
        c[size(a,1)+1, size(a,2)+1] = b
      end
      for qn in it_●●(A)
        a,b,c = ●●(A,qn),●●(B,qn),●●(C,qn)
        copyto!(view(c,axes(a,1), axes(a,2)), a)
        c[size(a,1)+1, size(a,2)+1] = b
      end
    else # k == d
      for qn in it_○○(A)
        a,b,c = ○○(A,qn),○○(B,qn),○○(C,qn)
        copyto!(view(c,axes(a,1),axes(a,2)), a)
        c[size(a,2)+1,1] = b
      end
      for qn in it_up(A)
        a,b,c = up(A,qn),up(B,qn),up(C,qn)
        copyto!(view(c,axes(a,1),axes(a,2)), a)
        c[size(a,2)+1,1] = b
      end
      for qn in it_dn(A)
        a,b,c = dn(A,qn),dn(B,qn),dn(C,qn)
        copyto!(view(c,axes(a,1),axes(a,2)), a)
        c[size(a,2)+1,1] = b
      end
      for qn in it_●●(A)
        a,b,c = ●●(A,qn),●●(B,qn),●●(C,qn)
        copyto!(view(c,axes(a,1),axes(a,2)), a)
        c[size(a,2)+1,1] = b
      end
    end
  else #d==1
    for qn in it_○○(A)
      a,b,c = ○○(A,qn),○○(B,qn),○○(C,qn)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
    for qn in it_up(A)
      a,b,c = up(A,qn),up(B,qn),up(C,qn)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
    for qn in it_dn(A)
      a,b,c = dn(A,qn),dn(B,qn),dn(C,qn)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
    for qn in it_●●(A)
      a,b,c = ●●(A,qn),●●(B,qn),●●(C,qn)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
  end

  return C
end

⊕(b::Number, A::SparseCore) = A⊕b

function ⊗(A::SparseCore{T,Nup,Ndn,d},B::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @assert site(A) == site(B)
  k = site(A)

  C = SparseCore{T,Nup,Ndn,d}(k, 
                        row_ranks(A) .* row_ranks(B), 
                        col_ranks(A) .* col_ranks(B))
  for qn in it_○○(A)
    a,b,c = ○○(A,qn),○○(B,qn),○○(C,qn)
    c .= reshape( 
            reshape(a, (size(a,1),1,size(a,2),1)) .* 
            reshape(b, (1,size(b,1),1,size(b,2))),
            (size(a,1)*size(b,1),size(a,2)*size(b,2))
                )
  end
  for qn in it_up(A)
    a,b,c = up(A,qn),up(B,qn),up(C,qn)
    c .= reshape( 
                  reshape(a, (size(a,1),1,size(a,2),1)) .* 
                  reshape(b, (1,size(b,1),1,size(b,2))),
                  (size(a,1)*size(b,1),size(a,2)*size(b,2))
                      )
  end
  for qn in it_dn(A)
    a,b,c = dn(A,qn),dn(B,qn),dn(C,qn)
    c .= reshape( 
                  reshape(a, (size(a,1),1,size(a,2),1)) .* 
                  reshape(b, (1,size(b,1),1,size(b,2))),
                  (size(a,1)*size(b,1),size(a,2)*size(b,2))
                      )
  end
  for qn in it_●●(A)
    a,b,c = ●●(A,qn),●●(B,qn),●●(C,qn)
    c .= reshape( 
                  reshape(a, (size(a,1),1,size(a,2),1)) .* 
                  reshape(b, (1,size(b,1),1,size(b,2))),
                  (size(a,1)*size(b,1),size(a,2)*size(b,2))
                      )
  end

  return C
end

function LinearAlgebra.qr!(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  R = Matrix{UpperTriangular{T,Matrix{T}}}(undef, Nup+1,Ndn+1)
  for (rup,rdn) in col_qn(A)
    An = A[(rup,rdn),:vertical]
    R[rup,rdn] = UpperTriangular(zeros(T,col_rank(A,rup,rdn),col_rank(A,rup,rdn)))
    if size(An,1)>0 && size(An,2)>0
      F = qr!(An)

      L = collect((lup,ldn) for (lup,ldn) in ( (rup,rdn), (rup-1,rdn), (rup,rdn-1), (rup-1,rdn-1) ) if in_row_qn(lup,ldn,A))
      rank = min(sum(row_rank(A,l) for l in L), col_rank(A,rup,rdn))
      Q = Matrix(F.Q)
      copyto!(R[rup,rdn], 1:rank, 1:col_rank(A,rup,rdn), 
              F.R,        1:rank, 1:col_rank(A,rup,rdn))
      
      # We make sure all ranks are the same, even if we have to introduce redundant zeros.
      Ls = cumsum(row_rank(A,l) for l in L)
      for (i,l) in enumerate(L)
        copyto!(A[l,(rup,rdn)],           1 :row_rank(A,l),      1:rank, 
                Q,    (i==1 ? 1 : Ls[i-1]+1):Ls[i],              1:rank)
        fill!(view(A[l,(rup,rdn)],        1 :row_rank(A,l), rank+1:col_rank(A,rup,rdn)), T(0))
      end
    end
  end

  return Frame{T,Nup,Ndn,d}(site(A)+1, R)
end

function LinearAlgebra.lq!(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  L = Matrix{LowerTriangular{T,Matrix{T}}}(undef, Nup+1,Ndn+1)
  for (lup,ldn) in row_qn(A)
    An = A[(lup,ldn),:horizontal]
    L[lup,ldn] = LowerTriangular(zeros(T,row_rank(A,lup,ldn),row_rank(A,lup,ldn)))
    if size(An,1) > 0 && size(An,2) > 0
      F = lq!(An)

      R = collect((rup,rdn) for (rup,rdn) in ( (lup,ldn), (lup+1,ldn), (lup,ldn+1), (lup+1,ldn+1) ) if in_col_qn(rup,rdn,A))
      rank = min(row_rank(A,lup,ldn), sum(col_rank(A,r) for r in R))
      Q = Matrix(F.Q)
      copyto!(L[lup,ldn], 1:row_rank(A,lup,ldn), 1:rank,
              F.L,        1:row_rank(A,lup,ldn), 1:rank)

      # We make sure all ranks are the same, even if we have to introduce redundant zeros.    
      Rs = cumsum(col_rank(A,r) for r in R)
      for (i,r) in enumerate(R)
        copyto!(A[(lup,ldn),r],         1:rank,                     1 :col_rank(A,r),
                Q,                      1:rank, (i==1 ? 1 : Rs[i-1]+1):Rs[i]        )
        fill!(view(A[(lup,ldn),r], rank+1:row_rank(A,lup,ldn),      1 :col_rank(A,r)), T(0))
      end
    end
  end

  return Frame{T,Nup,Ndn,d}(site(A), L)
end


function my_qc!(A::Matrix{T}) where {T<:Number}
  m = size(A,1)
  n = size(A,2)
  if m>0 && n>0
    # Lapack in-place pivoted QR factorization
    A, tau, jpvt = LinearAlgebra.LAPACK.geqp3!(A)
    # Search for effective rank
    ϵ = 16 * A[1,1] * eps()
    rank = min(m,n) - searchsortedlast(view(A, reverse(diagind(A))), ϵ, by=abs)
    # Extract C = R*P' factor
    C = zeros(T, rank, n)
    for j=1:n, i=1:min(j,rank)
      C[i, jpvt[j]] = A[i,j]
    end
    # Extract Q factor into A
    LinearAlgebra.LAPACK.orgqr!(A, tau)
    Q = view(A,:,1:rank)
  else  # n = 0
    rank = 0
    Q = view(Matrix{T}(undef,m,rank),:,1:rank)
    C = Matrix{T}(undef,rank,n)
  end

  return Q, C, rank
end

function qc!(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  Qblocks   = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  C         = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  col_ranks = zeros(Int, Nup+1, Ndn+1)
  for (nup,ndn) in col_qn(A)
    Qblocks[nup,ndn], C[nup,ndn], col_ranks[nup,ndn] = my_qc!(A[(nup,ndn),:vertical])
  end

  Q = SparseCore{T,Nup,Ndn,d}(site(A), row_ranks(A), col_ranks)
  for (nup,ndn) in col_qn(A)
      Q[(nup,ndn),:vertical] = Qblocks[nup,ndn]
  end
  return Q, Frame{T,Nup,Ndn,d}(site(A)+1, C), col_ranks
end

function my_cq(A::Matrix{T}) where {T<:Number}
  m = size(A,1)
  n = size(A,2)
  tA = copy(transpose(A))
  Q,C,rank = my_qc!(tA)
  return copy(transpose(Q)), copy(transpose(C)), rank
end

function cq!(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  Qblocks   = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  C         = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  row_ranks = zeros(Int, Nup+1, Ndn+1)
  for (nup,ndn) in row_qn(A)
    Qblocks[nup,ndn], C[nup,ndn], row_ranks[nup,ndn] = my_cq(A[(nup,ndn),:horizontal])
  end

  Q = SparseCore{T,Nup,Ndn,d}(site(A), row_ranks, col_ranks(A))
  for (nup,ndn) in row_qn(A)
      Q[(nup,ndn),:horizontal] = Qblocks[nup,ndn]
  end
  return Q, Frame{T,Nup,Ndn,d}(site(A), C), row_ranks
end

function svd_horizontal(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}  
  U  = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  S  = Matrix{Diagonal{T,Vector{T}}}(undef, Nup+1, Ndn+1)
  Vt = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  for (lup,ldn) in row_qn(A)
    if row_rank(A,(lup,ldn)) > 0
      F = svd!(A[(lup,ldn),:horizontal])
      U[lup,ldn] = F.U
      S[lup,ldn] = Diagonal(F.S)
      Vt[lup,ldn]= F.Vt
    else
      U[lup,ldn] = zeros(T,0,0)
      S[lup,ldn] = Diagonal(zeros(T,0))
      Vt[lup,ldn]= zeros(T,0,sum(col_rank(A,rup,rdn) for (rup,rdn) in ( (lup,ldn), (lup+1,ldn), (lup,ldn+1), (lup+1,ldn+1) ) if in_col_qn(rup,rdn,A)))
    end
  end
  return Frame{T,Nup,Ndn,d}(site(A), U), Frame{T,Nup,Ndn,d}(site(A), S), Vt
end

function svd_vertical(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}  
  U  = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  S  = Matrix{Diagonal{T,Vector{T}}}(undef, Nup+1, Ndn+1)
  Vt = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  for (rup,rdn) in col_qn(A)
    if col_rank(A,(rup,rdn)) > 0
      F = svd!(A[(rup,rdn),:vertical])
      U[nup,ndn] = F.U
      S[nup,ndn] = Diagonal(F.S)
      Vt[nup,ndn]= F.Vt
    else
      U[nup,ndn] = zeros(T,sum(row_rank(A,l) for l in ( (rup-1,rdn-1), (rup,rdn-1), (rup-1,rdn), (rup,rdn) ) if l ∈ row_qn(A)),0)
      S[nup,ndn] = Diagonal(zeros(T,0))
      Vt[nup,ndn]= zeros(T,0,0)
    end
  end
  return U, Frame{T,Nup,Ndn,d}(site(A)+1, S), Frame{T,Nup,Ndn,d}(site(A)+1, Vt)
end

function svd_horizontal_to_core(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}  
  U,S,Vt = svd_horizontal(A)
  C = SparseCore{T,Nup,Ndn,d}(site(A), col_ranks(S), col_ranks(A))
  for (nup,ndn) in row_qn(C)
    C[(nup,ndn),:horizontal] = Vt[nup,ndn]
  end
  return U, S, C
end

function svd_vertical_to_core(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  U,S,Vt = svd_vertical(A)
  C = SparseCore{T,Nup,Ndn,d}(site(A), row_ranks(A), row_ranks(S))
  for (nup,ndn) in col_qn(C)
    C[(nup,ndn),:vertical] = U[nup,ndn]
  end
  return C, S, Vt
end

function svdvals_horizontal(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return [svdvals!(A[n, :horizontal]) for n in axes(A,1)]
end

function svdvals_vertical(A::SparseCore{T,Nup,Ndn,d}, unfolding::Symbol) where {T<:Number,Nup,Ndn,d}
  return [svdvals!(A[n, :vertical]) for n in axes(A,3)]
end

function reduce_ranks(A::SparseCore{T,Nup,Ndn,d}, 
                       rowranks::Matrix{Int}, 
                       colranks::Matrix{Int}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert axes(rowranks) == axes(row_ranks(A))
    @assert axes(colranks) == axes(col_ranks(A))
    @assert all(0 .≤ rowranks .≤ row_ranks(A))
    @assert all(0 .≤ colranks .≤ col_ranks(A))
  end
  
  B = SparseCore{T,Nup,Ndn,d}(site(A), rowranks, colranks)
  
  for (nup,ndn) in row_qn(A)
    in_col_qn(nup  ,ndn  ,A) && copyto!(○○(B,nup,ndn),1:rowranks[nup,ndn],1:colranks[nup  ,ndn  ], 
                                        ○○(A,nup,ndn),1:rowranks[nup,ndn],1:colranks[nup  ,ndn  ])
    in_col_qn(nup+1,ndn  ,A) && copyto!(up(B,nup,ndn),1:rowranks[nup,ndn],1:colranks[nup+1,ndn  ], 
                                        up(A,nup,ndn),1:rowranks[nup,ndn],1:colranks[nup+1,ndn  ])
    in_col_qn(nup  ,ndn+1,A) && copyto!(dn(B,nup,ndn),1:rowranks[nup,ndn],1:colranks[nup  ,ndn+1], 
                                        dn(A,nup,ndn),1:rowranks[nup,ndn],1:colranks[nup  ,ndn+1])
    in_col_qn(nup+1,ndn+1,A) && copyto!(●●(B,nup,ndn),1:rowranks[nup,ndn],1:colranks[nup+1,ndn+1], 
                                        ●●(A,nup,ndn),1:rowranks[nup,ndn],1:colranks[nup+1,ndn+1])
  end

  return B
end

function Base.Array(A::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  array = zeros(T,sum(row_ranks(A)),4,sum(col_ranks(A)))
  rS = cumsum(row_rank(A,qn) for qn in row_qn(A))
  cS = cumsum(col_rank(A,qn) for qn in col_qn(A))

  for (i,qn) in enumerate(row_qn(A))
    nup,ndn = qn
    j = findfirst((nup  ,ndn  ), col_qn(A))
    if !isnothing(j)
      copyto!(array, rS[i-1]+1:rS[i], 1, cS[j-1]+1:cS[j], ○○(A,qn), 1:rowranks[nup,ndn], 1:col_ranks[nup,ndn])
    end
    j = findfirst((nup+1,ndn  ), col_qn(A))
    if !isnothing(j)
      copyto!(array, rS[i-1]+1:rS[i], 2, cS[j-1]+1:cS[j], up(A,qn), 1:rowranks[nup,ndn],1:colranks[nup+1,ndn  ])
    end
    j = findfirst((nup  ,ndn+1), col_qn(A)) 
    if !isnothing(j)
      copyto!(array, rS[i-1]+1:rS[i], 3, cS[j-1]+1:cS[j], dn(A,qn), 1:rowranks[nup,ndn],1:colranks[nup  ,ndn+1])
    end
    j = findfirst((nup+1,ndn+1), col_qn(A)) 
    if !isnothing(j)
      copyto!(array, rS[i-1]+1:rS[i], 4, cS[j-1]+1:cS[j], ●●(A,qn), 1:rowranks[nup,ndn],1:colranks[nup+1,ndn+1])
    end
  end

  return array
end

function VectorInterface.zerovector(x::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return SparseCore{T,Nup,Ndn,d}(x.k,row_ranks(x),col_ranks(x))
end

function VectorInterface.zerovector(x::SparseCore{S,Nup,Ndn,d}, T::Type{<:Number}) where {S<:Number,Nup,Ndn,d}
  return SparseCore{T,Nup,Ndn,d}(x.k,row_ranks(x),col_ranks(x))
end

function VectorInterface.add!!(y::SparseCore{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d}, α::Number, β::Number) where {T<:Number,Nup,Ndn,d}
  return axpby!(α,x,β,y)
end

function VectorInterface.scale(x::SparseCore{T,Nup,Ndn,d}, α::Number) where {T<:Number,Nup,Ndn,d}
    return VectorInterface.scale!!(deepcopy(x), α)
end

function VectorInterface.scale!!(x::SparseCore{T,Nup,Ndn,d}, α::Number) where {T<:Number,Nup,Ndn,d}
    α === VectorInterface.One() && return x
    return lmul!(α,x)
end
function VectorInterface.scale!!(y::SparseCore{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d}, α::Number) where {T<:Number,Nup,Ndn,d}
    return mul!(y,x,α)
end

function VectorInterface.inner(x::SparseCore{T,Nup,Ndn,d}, y::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return LinearAlgebra.dot(x,y)
end
