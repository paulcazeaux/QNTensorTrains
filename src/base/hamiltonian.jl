module Hamiltonian
import ..QNTensorTrains
using ..QNTensorTrains: Orbital, Spin, Up, Dn, Frame, IdFrame, SparseCore, AdjointCore, TTvector
using ..QNTensorTrains: site, core, ○○, up, dn, ●●, it_○○, it_up, it_dn, it_●●
using ..QNTensorTrains: row_ranks, col_ranks, row_rank, col_rank, state_qn, row_qn, col_qn, shift_qn, ε, cores2tensor
using LinearAlgebra, OffsetArrays

export SparseHamiltonian, H_matvec_core, RayleighQuotient, xᵀHy

const OV{T} = OffsetVector{T,Vector{T}}
const OM{T} = OffsetMatrix{T,Matrix{T}}
const sparseblock{M} = @NamedTuple{○○::M,up::M,dn::M,●●::M}
const Ts = Tuple{@NamedTuple{up::NTuple{3,Int},dn::NTuple{3,Int}}, NTuple{2,Int}}

using Graphs, MetaGraphsNext
struct SparseHamiltonian{T<:Number,Nup,Ndn,d}
  states::Vector{Matrix{Vector{Ts}}}
  state_qns::Vector{Matrix{Vector{NTuple{2,Int}}}}
  coo_blocks::Vector{sparseblock{Matrix{sparseblock{OM{Tuple{Vector{Int},Vector{Int},Vector{T}}}}}}}
  csr_blocks::Vector{sparseblock{Matrix{sparseblock{OM{Tuple{OV{Int},Vector{Int},Vector{T}}}}}}}
  csc_blocks::Vector{sparseblock{Matrix{sparseblock{OM{Tuple{OV{Int},Vector{Int},Vector{T}}}}}}}

  graph::MetaGraph

  function SparseHamiltonian(t::Matrix{T}, v::Array{T,4}, N::Int, Sz::Rational, d::Int; ϵ=eps()) where {T<:Number}
    Nup = Int(N+2Sz)÷2
    Ndn = Int(N-2Sz)÷2
    return SparseHamiltonian(t,v,Val(Nup),Val(Ndn),Val(d); ϵ=ϵ)
  end

  function SparseHamiltonian(t::Matrix{T}, v::Array{T,4}, ::Val{Nup}, ::Val{Ndn}, ::Val{d}; ϵ=eps()) where {T<:Number,Nup,Ndn,d}
    @boundscheck @assert size(t) == (d,d)
    @boundscheck @assert size(v) == (d,d,d,d)
    hd = (d+1)÷2
    
    # Flowchart for left-half cores: 
    # One-Body states
    #               /---A--> 3▲ -S-> 3▲ -Adag-\
    #              /                           \
    # 1  ---Id--> 1 ------------AdagA-----------> 4 ---Id--> 4
    #              \                           /
    #               \-Adag-> 2▲ -S-> 2▲ ---A--/
    #
    #
    # Two-body states
    #
    #                                   /---A--> 10⏫ --Id->10⏫ -▲Adag-▼-\
    #                                  /           kl         kl  ▲     ▼  \
    #                                 /                           ▲     ▼   \   j         j
    #              /---A-->7▲ --S-->7▲ ---------------------------▲AdagA▼---->12▼ --S-->12▼ -Adag-\
    #             /         k        k\                           ▲     ▼   /                      \
    #            /                     \-Adag-\               /---▲--A--▼--/                        \
    #           /                              \             /    ▲     ▼                            \
    # 5 --Id-> 5 --------------AdagA-----------> 8⏫-Id-> 8⏫-----▲AdagA▼----------------------------->13--Id->13
    #           \                              /  ik       ik\    ▲     ▼                            /
    #            \                     /--A---/               \---▲Adag-▼--\                        /
    #             \                   /                           ▲     ▼   \   l         l        /
    #              \-Adag->6▲ --S-->6▲ ---------------------------▲AdagA▼---->11▼ --S-->11▼ ---A--/
    #                       i        i\                           ▲     ▼   /
    #                                  \                          ▲     ▼  /
    #                                   \-Adag-> 9⏫  --Id-> 9⏫ -▲--A--▼-/
    #                                             ij          ij


    # states = (
    #     One body states
    #
    #     (up=( 0,0,1),dn=( 0,0,0)), #  1 #         n ≤ i↑,j↑
    #     (up=( 1,1,0),dn=( 0,0,0)), #  2 # i↑    < n ≤    j↑
    #     (up=(-1,0,1),dn=( 0,0,0)), #  3 #    j↑ < n ≤ i↑
    #     (up=( 0,1,0),dn=( 0,0,0)), #  4 # i↑,j↑ < n
    #
    #     (up=( 0,0,0),dn=( 0,0,1)), #  5 #         n ≤ i↓,j↓
    #     (up=( 0,0,0),dn=( 1,1,0)), #  6 # i↓    < n ≤    j↓
    #     (up=( 0,0,0),dn=(-1,0,1)), #  7 #    j↓ < n ≤ i↓
    #     (up=( 0,0,0),dn=( 0,1,0)), #  8 # i↓,j↓ < n
    #
    #     Two body states
    #
    #     (up=( 0,0,2),dn=( 0,0,0)), # 9  #               n ≤ i↑<k↑,l↑<j↑
    #     (up=( 1,1,1),dn=( 0,0,0)), # 10 # i↑          < n ≤    k↑,l↑<j↑
    #     (up=(-1,0,2),dn=( 0,0,0)), # 11 #       l↑    < n ≤ i↑<k↑,   j↑
    #     (up=( 0,1,1),dn=( 0,0,0)), # 12 # i↑,   l↑    < n ≤    k↑,   j↑
    #     (up=( 2,2,0),dn=( 0,0,0)), # 13 # i↑<k↑       < n ≤       l↑<j↑
    #     (up=(-2,0,2),dn=( 0,0,0)), # 14 #       l↑<j↑ < n ≤ i↑<k↑
    #     (up=( 1,2,0),dn=( 0,0,0)), # 15 # i↑<k↑,l↑    < n ≤          j↑
    #     (up=(-1,1,1),dn=( 0,0,0)), # 16 # i↑,   l↑<j↑ < n ≤    k↑
    #     (up=( 0,2,0),dn=( 0,0,0))  # 17 # i↑<k↑,l↑<j↑ < n
    #
    #     (up=( 0,0,0),dn=( 0,0,2)), # 18 #               n ≤ i↓<k↓,l↓<j↓
    #     (up=( 0,0,0),dn=( 1,1,1)), # 19 # i↓          < n ≤     k↓,l↓<j↓
    #     (up=( 0,0,0),dn=(-1,0,2)), # 20 #       k↓    < n ≤  i↓<k↓,   j↓
    #     (up=( 0,0,0),dn=( 0,1,1)), # 21 # i↓,   k↓    < n ≤     k↓,   j↓
    #     (up=( 0,0,0),dn=( 2,2,0)), # 22 # i↓<j↓       < n ≤        l↓<j↓
    #     (up=( 0,0,0),dn=(-2,0,2)), # 23 #       k↓<l↓ < n ≤  i↓<k↓
    #     (up=( 0,0,0),dn=( 1,2,0)), # 24 # i↓<j↓,k↓    < n ≤           j↓
    #     (up=( 0,0,0),dn=(-1,1,1)), # 25 # i↓,   k↓<l↓ < n ≤     k↓
    #     (up=( 0,0,0),dn=( 0,2,0))  # 26 # i↓<j↓,k↓<l↓ < n 
    #
    #     (up=( 0,0,1),dn=( 0,0,1)), # 27 #               n ≤ i↑,k↓,l↓,j↑
    #     (up=( 1,1,0),dn=( 0,0,1)), # 28 # i↑          < n ≤    k↓,l↓,j↑
    #     (up=( 0,0,1),dn=( 1,1,0)), # 29 #    k↓       < n ≤ i↑,   l↓,j↑
    #     (up=( 0,0,1),dn=(-1,0,1)), # 30 #       l↓    < n ≤ i↑,k↓,   j↑
    #     (up=(-1,0,1),dn=( 0,0,1)), # 31 #          j↑ < n ≤ i↑,k↓,l↓
    #     (up=( 1,1,0),dn=( 1,1,0)), # 32 # i↑,k↓       < n ≤       l↓,j↑
    #     (up=( 1,1,0),dn=(-1,0,1)), # 33 # i↑,   l↓    < n ≤    k↓,   j↑
    #     (up=( 0,1,0),dn=( 0,0,1)), # 34 # i↑,      j↑ < n ≤    k↓,l↓
    #     (up=( 0,0,1),dn=( 0,1,0)), # 35 #    k↓,l↓    < n ≤ i↑,      j↑
    #     (up=(-1,0,1),dn=( 1,1,0)), # 36 #    k↓,   j↑ < n ≤ i↑,   l↓
    #     (up=(-1,0,1),dn=(-1,0,1)), # 37 #       l↓,j↑ < n ≤ i↑,k↓
    #     (up=( 1,1,0),dn=( 0,1,0)), # 38 # i↑,k↓,l↓    < n ≤          j↑
    #     (up=( 0,1,0),dn=( 1,1,0)), # 39 # i↑,k↓,   j↑ < n ≤       l↓
    #     (up=( 0,1,0),dn=(-1,0,1)), # 40 # i↑,   l↓,j↑ < n ≤    k↓
    #     (up=(-1,0,1),dn=( 0,1,0)), # 41 #    k↓,l↓,j↑ < n ≤ i↑   
    #     (up=( 0,0,1),dn=( 0,1,0)), # 42 # i↑,k↓,l↓,j↑ < n

############################################################

    function occupation(l::Tuple{Int,Int},r::Tuple{Int,Int})
      lup,ldn = l
      if l==r
        return :○○
      elseif r == (lup+1,ldn  )
        return :up
      elseif r == (lup  ,ldn+1)
        return :dn
      elseif r == (lup+1,ldn+1)
        return :●●
      else
        throw(BoundsError())
      end
    end

    function jordanwigner(l::Tuple{Int,Int}, r::Tuple{Int,Int}, w::T, op::@NamedTuple{up::Symbol,dn::Symbol})
      lup,ldn = l
      jw_up = (op.up in (:A, :Adag) && isodd(lup+ldn) ? -1 : 1)

      if l==r
        jw = jw_up * (op.dn in (:A, :Adag) && isodd(lup+ldn  ) ? -w : w)
        return (l,:○○,jw)
      elseif r == (lup+1,ldn  )
        jw = jw_up * (op.dn in (:A, :Adag) && isodd(lup+ldn+1) ? -w : w)
        return (l,:up,jw)
      elseif r == (lup  ,ldn+1)
        jw = jw_up * (op.dn in (:A, :Adag) && isodd(lup+ldn  ) ? -w : w)
        return (l,:dn,jw)
      elseif r == (lup+1,ldn+1)
        jw = jw_up * (op.dn in (:A, :Adag) && isodd(lup+ldn+1) ? -w : w)
        return (l,:●●,jw)
      else
        throw(BoundsError())
      end
    end

    # One body states
    function vertex(κ, iσ::Orbital,jσ::Orbital)
      @boundscheck @assert iσ.spin == jσ.spin
      if iσ.spin == jσ.spin == Up
        i,j = iσ.site, jσ.site
        if κ ≤ min(i,j)
          state = (up=( 0,0,1),dn=( 0,0,0)); index =       (0,0)
        elseif i < κ ≤ j
          state = (up=( 1,1,0),dn=( 0,0,0)); index = (κ ≤ hd ? (i,0) : (j,0))
        elseif j < κ ≤ i
          state = (up=(-1,0,1),dn=( 0,0,0)); index = (κ ≤ hd ? (j,0) : (i,0))
        else # max(i,j) < κ
          state = (up=( 0,1,0),dn=( 0,0,0)); index =       (0,0)
        end
      elseif iσ.spin == jσ.spin == Dn
        i,j = iσ.site, jσ.site
        if κ ≤ min(i,j)
          state = (up=( 0,0,0),dn=( 0,0,1)); index =       (0,0)
        elseif i < κ ≤ j
          state = (up=( 0,0,0),dn=( 1,1,0)); index = (κ ≤ hd ? (i,0) : (j,0))
        elseif j < κ ≤ i
          state = (up=( 0,0,0),dn=(-1,0,1)); index = (κ ≤ hd ? (j,0) : (i,0))
        else # max(i,j) < κ
          state = (up=( 0,0,0),dn=( 0,1,0)); index =       (0,0)
        end
      end
      return state, index
    end

    function weighted_edge(i::Orbital,j::Orbital)
      return sort([hd, i.site,j.site])[2]
    end

    function set_edges(κ, i::Orbital, j::Orbital, w::T=T(1))
      @boundscheck @assert  1 ≤ κ ≤ d && i.spin == j.spin
      s₋,idx₋ = vertex(κ,   i,j)
      s₊,idx₊ = vertex(κ+1, i,j)

      op = ( κ == i.site == j.site ? (i.spin == Up ? (up=:N,   dn=:Id) : (up=:Id,dn=:N   ) ) :
             κ == j.site           ? (j.spin == Up ? (up=:A,   dn=:Id) : (up=:Id,dn=:A   ) ) :
             κ == i.site           ? (j.spin == Up ? (up=:Adag,dn=:Id) : (up=:Id,dn=:Adag) ) : (up=:Id,dn=:Id) )

      qn₋ = shift_qn(state_qn(Nup,Ndn,d,κ  ),Nup,Ndn,s₋.up...,s₋.dn...)
      qn₊ = shift_qn(state_qn(Nup,Ndn,d,κ+1),Nup,Ndn,s₊.up...,s₊.dn...)
      continue_path = false
      for n₋ in qn₋
        nup₋,ndn₋ = n₋
        for n₊ in ( op == (up=:A,dn=:Id)                      ? ((nup₋,ndn₋),  (nup₋,ndn₋+1))   : 
                    op == (up=:Id,dn=:A)                      ? ((nup₋,ndn₋),  (nup₋+1,ndn₋))   : 
                    op in ((up=:Adag,dn=:Id), (up=:N,dn=:Id)) ? ((nup₋+1,ndn₋),(nup₋+1,ndn₋+1)) : 
                    op in ((up=:Id,dn=:Adag), (up=:Id,dn=:N)) ? ((nup₋,ndn₋+1),(nup₋+1,ndn₋+1)) 
                    :  ((nup₋,ndn₋),(nup₋+1,ndn₋),(nup₋,ndn₋+1),(nup₋+1,ndn₋+1)) ) ∩ qn₊
          nup₊, ndn₊ = n₊
          l  = (nup₋-s₋.up[1], ndn₋-s₋.dn[1])
          r  = (nup₊-s₊.up[1], ndn₊-s₊.dn[1])
          continue_path = add_vertex!(graph, (κ,n₋,s₋,idx₋)) | add_vertex!(graph, (κ+1,n₊,s₊,idx₊)) || continue_path
          add_edge!(graph, (κ,n₋,s₋,idx₋), (κ+1,n₊,s₊,idx₊), jordanwigner(l,r,w,op))
        end
      end
      return continue_path
    end

    # Two body states
    function vertex(κ, i::Orbital, j::Orbital, k::Orbital, l::Orbital)
      @boundscheck @assert i.spin == j.spin && k.spin == l.spin
      if i.spin == k.spin == Up
        i,j,k,l = i.site, j.site, k.site, l.site
        @boundscheck @assert i<k && l<j
        if κ ≤ min(i,l)
          state = (up=( 0,0,2),dn=( 0,0,0));    index =        (0,0)
        elseif i < κ ≤ min(k,l)
          state = (up=( 1,1,1),dn=( 0,0,0));    index =        (i,0)
        elseif l < κ ≤ min(i,j)
          state = (up=(-1,0,2),dn=( 0,0,0));    index =        (l,0)
        elseif max(i,l) < κ ≤ min(k,j)
          state = (up=( 0,1,1),dn=( 0,0,0));    index = (κ ≤ hd ?  (i,l) : (k,j))
        elseif k < κ ≤ l
          state = (up=( 2,2,0),dn=( 0,0,0));    index = (κ ≤ hd ?  (i,k) : (l,j))
        elseif j < κ ≤ i
          state = (up=(-2,0,2),dn=( 0,0,0));    index = (κ ≤ hd ?  (l,j) : (i,k))
        elseif max(k,l) < κ ≤ j
          state = (up=( 1,2,0),dn=( 0,0,0));    index =        (j,0)
        elseif max(i,j) < κ ≤ k
          state = (up=(-1,1,1),dn=( 0,0,0));    index =        (k,0)
        else # max(k,j) < κ
          state = (up=( 0,2,0),dn=( 0,0,0));    index =        (0,0)
        end 
      elseif i.spin == k.spin == Dn
        i,j,k,l = i.site, j.site, k.site, l.site
        @boundscheck @assert i<k && l<j
        if κ ≤ min(i,l)
          state = (up=( 0,0,0),dn=( 0,0,2));    index =        (0,0)
        elseif i < κ ≤ min(k,l)
          state = (up=( 0,0,0),dn=( 1,1,1));    index =        (i,0)
        elseif l < κ ≤ min(i,j)
          state = (up=( 0,0,0),dn=(-1,0,2));    index =        (l,0)
        elseif max(i,l) < κ ≤ min(k,j)
          state = (up=( 0,0,0),dn=( 0,1,1));    index = (κ ≤ hd ?  (i,l) : (k,j))
        elseif k < κ ≤ l
          state = (up=( 0,0,0),dn=( 2,2,0));    index = (κ ≤ hd ?  (i,k) : (l,j))
        elseif j < κ ≤ i
          state = (up=( 0,0,0),dn=(-2,0,2));    index = (κ ≤ hd ?  (l,j) : (i,k))
        elseif max(k,l) < κ ≤ j
          state = (up=( 0,0,0),dn=( 1,2,0));    index =        (j,0)
        elseif max(i,j) < κ ≤ k
          state = (up=( 0,0,0),dn=(-1,1,1));    index =        (k,0)
        else # max(k,j) < κ
          state = (up=( 0,0,0),dn=( 0,2,0));    index =        (0,0)
        end 
      elseif i.spin == Up && k.spin == Dn
        i,j,k,l = i.site, j.site, k.site, l.site
        if κ ≤ min(i,j,k,l)
          state = (up=( 0,0,1),dn=( 0,0,1));    index =        (0,0)
        elseif i < κ ≤ min(j,k,l)
          state = (up=( 1,1,0),dn=( 0,0,1));    index =        (i,0)
        elseif k < κ ≤ min(i,j,l)
          state = (up=( 0,0,1),dn=( 1,1,0));    index =        (k,0)
        elseif l < κ ≤ min(i,j,k)
          state = (up=( 0,0,1),dn=(-1,0,1));    index =        (l,0)
        elseif j < κ ≤ min(i,k,l)
          state = (up=(-1,0,1),dn=( 0,0,1));    index =        (j,0)
        elseif max(i,k) < κ ≤ min(j,l)
          state = (up=( 1,1,0),dn=( 1,1,0));    index =  (k ≤ hd ? (i,k) : (l,j))
        elseif max(i,l) < κ ≤ min(j,k)
          state = (up=( 1,1,0),dn=(-1,0,1));    index =  (k ≤ hd ? (i,l) : (k,j))
        elseif max(i,j) < κ ≤ min(k,l)
          state = (up=( 0,1,0),dn=( 0,0,1));    index =  (k ≤ hd ? (i,j) : (k,l))
        elseif max(k,l) < κ ≤ min(i,j)
          state = (up=( 0,0,1),dn=( 0,1,0));    index =  (k ≤ hd ? (k,l) : (i,j))
        elseif max(k,j) < κ ≤ min(i,l)
          state = (up=(-1,0,1),dn=( 1,1,0));    index =  (k ≤ hd ? (k,j) : (i,l))
        elseif max(l,j) < κ ≤ min(i,k)
          state = (up=(-1,0,1),dn=(-1,0,1));    index =  (k ≤ hd ? (l,j) : (i,k))
        elseif max(i,k,l) < κ ≤ j
          state = (up=( 1,1,0),dn=( 0,1,0));    index =        (j,0)
        elseif max(i,j,k) < κ ≤ l
          state = (up=( 0,1,0),dn=( 1,1,0));    index =        (l,0)
        elseif max(i,j,l) < κ ≤ k
          state = (up=( 0,1,0),dn=(-1,0,1));    index =        (k,0)
        elseif max(j,k,l) < κ ≤ i
          state = (up=(-1,0,1),dn=( 0,1,0));    index =        (i,0)
        else # max(i,j,k,l) < κ
          state = (up=( 0,0,1),dn=( 0,1,0));    index =        (0,0)
        end
      else
        throw(BoundsError())
      end
      return state, index
    end

    function weighted_edge(i::Orbital,j::Orbital,k::Orbital,l::Orbital)
      return sort([hd, i.site,j.site,k.site,l.site])[3]
    end 

    function set_edges(κ::Int, i::Orbital,j::Orbital,k::Orbital,l::Orbital, w=T(1))
      @assert i.spin == j.spin && k.spin == l.spin && 1 ≤ κ ≤ d

      s₋,idx₋ = vertex(κ,   i,j,k,l)
      s₊,idx₊ = vertex(κ+1, i,j,k,l)

      spin_op(Oκ::Orbital) = (Oκ∈(i,k) && Oκ∈(j,l) ? :N : Oκ∈(i,k) ? :Adag : Oκ∈(j,l) ? :A : :Id)
      op = (up = spin_op((site=κ,spin=Up)), dn = spin_op((site=κ,spin=Dn)))

      qn₋ = shift_qn(state_qn(Nup,Ndn,d,κ  ), Nup,Ndn,s₋.up...,s₋.dn...)
      qn₊ = shift_qn(state_qn(Nup,Ndn,d,κ+1), Nup,Ndn,s₊.up...,s₊.dn...)
      continue_path = false
      for n₋ in qn₋
        nup₋,ndn₋ = n₋
        for n₊ in [(nup₊, ndn₊) 
                    for nup₊ in ( op.up == :Id ? (nup₋:nup₋+1  ) :
                                  op.up == :A  ? (nup₋:nup₋    ) :
                                                 (nup₋+1:nup₋+1)), 
                        ndn₊ in ( op.dn == :Id ? (ndn₋:ndn₋+1  ) :
                                  op.dn == :A  ? (ndn₋:ndn₋    ) :
                                                 (ndn₋+1:ndn₋+1))] ∩ qn₊
          nup₊, ndn₊ = n₊
          l  = (nup₋-s₋.up[1], ndn₋-s₋.dn[1])
          r  = (nup₊-s₊.up[1], ndn₊-s₊.dn[1])
          continue_path = add_vertex!(graph, (κ,n₋,s₋,idx₋)) | add_vertex!(graph, (κ+1,n₊,s₊,idx₊)) || continue_path
          add_edge!(graph, (κ,n₋,s₋,idx₋), (κ+1,n₊,s₊,idx₊), jordanwigner(l,r,w,op))
        end
      end

      return continue_path
    end
    graph = MetaGraph(
      DiGraph();                       # Initialize empty graph
      label_type=Tuple{Int, NTuple{2,Int}, @NamedTuple{up::NTuple{3,Int},dn::NTuple{3,Int}}, NTuple{2,Int} },          
                                       # site, quantum number, state and index
      vertex_data_type=Nothing,        # State details 
      edge_data_type=Tuple{NTuple{2,Int},Symbol,T}, # relevant block indices, coefficient and single-site operator
      weight_function=ed -> ed[3],
      default_weight=0.,
      graph_data="Hamiltonian action graph",                  # tag for the whole graph
    )

    for i=1:d, j=1:d
      if abs(t[i,j]) > ϵ
        for σ in (Up,Dn)
          iσ = (site=i,spin=σ)
          jσ = (site=j,spin=σ)
          κₘ = weighted_edge(iσ,jσ)
          set_edges(κₘ, iσ,jσ, ε(iσ,jσ)*t[i,j])
          for κ=κₘ+1:d
            continue_path = set_edges(κ, iσ, jσ) 
            continue_path || break
          end
          for κ = κₘ-1:-1:1
            continue_path = set_edges(κ, iσ, jσ) 
            continue_path || break
          end
        end
      end
    end

    # Two-body terms a†(i↑)a†(k↑)a(l↑)a(j↑) and a†(i↓)a†(k↓)a(l↓)a(j↓)
    for i=1:d-1, j=i+1:d, k=1:d-1, l=k+1:d
      w = 1/2*(v[i,j,k,l]+v[k,l,i,j]-v[i,l,k,j]-v[k,j,i,l])
      if abs(w) > ϵ
        for σ in (Up,Dn)
          iσ, jσ, kσ, lσ = (site=i,spin=σ), (site=j,spin=σ), (site=k,spin=σ), (site=l,spin=σ)
          κₘ = weighted_edge(iσ,jσ,kσ,lσ)
          set_edges(κₘ, iσ,jσ,kσ,lσ, ε(iσ,jσ,kσ,lσ)*w)
          for κ=κₘ+1:d
            continue_path = set_edges(κ, iσ,jσ,kσ,lσ) 
            continue_path || break
          end
          for κ = κₘ-1:-1:1
            continue_path = set_edges(κ, iσ,jσ,kσ,lσ) 
            continue_path || break
          end
        end
      end
    end

    # Two-body terms a†(i↑)a†(k↓)a(l↓)a(j↑) and a†(i↑)a†(k↓)a(l↓)a(j↑)
    for i=1:d, j=1:d, k=1:d, l=1:d
      w = 1/2*(v[i,j,k,l]+v[k,l,i,j])
      if abs(w) > ϵ
        iσ, jσ, kσ, lσ = (site=i,spin=Up), (site=j,spin=Dn), (site=k,spin=Dn), (site=l,spin=Up)
        κₘ = weighted_edge(iσ,jσ,kσ,lσ)
        set_edges(κₘ, iσ,jσ,kσ,lσ, ε(iσ,jσ,kσ,lσ)*w)
        for κ=κₘ+1:d
          continue_path = set_edges(κ, iσ,jσ,kσ,lσ) 
          continue_path || break
        end
        for κ = κₘ-1:-1:1
          continue_path = set_edges(κ, iσ,jσ,kσ,lσ) 
          continue_path || break
        end
      end
    end

    nstates = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
    for (k,n,s,idx) in labels(graph)
      nup,ndn = n
      nstates[k][nup,ndn] += 1
    end

    states = [ Matrix{Vector{Ts}}(undef,Nup+1,Ndn+1) for k=1:d+1]
    i      = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
    for k=1:d+1, nup=1:Nup+1, ndn=1:Ndn+1
      states[k][nup,ndn] = Vector{Ts}(undef, nstates[k][nup,ndn])
    end
    for (k,(nup,ndn),s,idx) in labels(graph)
      states[k][nup,ndn][
        i[k][nup,ndn] += 1
                        ] = (s, idx)
    end

    for k=1:d+1, nup=1:Nup+1, ndn=1:Ndn+1
      sort!(states[k][nup,ndn]) # Let's try mostly the default lexicographic order.
    end
    state_to_index = [ [ Dict(s=>i for (i,s) in enumerate(states[k][nup,ndn])) 
                         for nup=1:Nup+1, ndn=1:Ndn+1 ] for k=1:d+1 ]
    state_qns = [ [ [(nup-fluxup,ndn-fluxdn) for (((fluxup,),(fluxdn,)),) in states[k][nup,ndn]] 
                         for nup=1:Nup+1, ndn=1:Ndn+1 ] for k=1:d+1 ]

    qn = [ (
       ○○ = collect(it_○○(state_qn(Nup,Ndn,d,k),state_qn(Nup,Ndn,d,k+1))),
       up = collect(it_up(state_qn(Nup,Ndn,d,k),state_qn(Nup,Ndn,d,k+1))),
       dn = collect(it_dn(state_qn(Nup,Ndn,d,k),state_qn(Nup,Ndn,d,k+1))),
       ●● = collect(it_●●(state_qn(Nup,Ndn,d,k),state_qn(Nup,Ndn,d,k+1)))
           ) for k=1:d ]

    wrap(f) = [(
      ○○ = [ 
        ( ○○ = OffsetMatrix([ f(k,nup,ndn,:○○,mup,mdn,:○○) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          up = OffsetMatrix([ f(k,nup,ndn,:○○,mup,mdn,:up) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          dn = OffsetMatrix([ f(k,nup,ndn,:○○,mup,mdn,:dn) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          ●● = OffsetMatrix([ f(k,nup,ndn,:○○,mup,mdn,:●●) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2))
        ) for nup=1:Nup+1, ndn=1:Ndn+1 ],
      up = [ 
        ( ○○ = OffsetMatrix([ f(k,nup,ndn,:up,mup,mdn,:○○) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          up = OffsetMatrix([ f(k,nup,ndn,:up,mup,mdn,:up) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          dn = OffsetMatrix([ f(k,nup,ndn,:up,mup,mdn,:dn) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          ●● = OffsetMatrix([ f(k,nup,ndn,:up,mup,mdn,:●●) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2))
        ) for nup=1:Nup+1, ndn=1:Ndn+1 ],
      dn = [ 
        ( ○○ = OffsetMatrix([ f(k,nup,ndn,:dn,mup,mdn,:○○) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          up = OffsetMatrix([ f(k,nup,ndn,:dn,mup,mdn,:up) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          dn = OffsetMatrix([ f(k,nup,ndn,:dn,mup,mdn,:dn) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          ●● = OffsetMatrix([ f(k,nup,ndn,:dn,mup,mdn,:●●) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2))
        ) for nup=1:Nup+1, ndn=1:Ndn+1 ],
      ●● = [ 
        ( ○○ = OffsetMatrix([ f(k,nup,ndn,:●●,mup,mdn,:○○) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          up = OffsetMatrix([ f(k,nup,ndn,:●●,mup,mdn,:up) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          dn = OffsetMatrix([ f(k,nup,ndn,:●●,mup,mdn,:dn) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2)),
          ●● = OffsetMatrix([ f(k,nup,ndn,:●●,mup,mdn,:●●) for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)], max(1,nup-2):min(Nup+1,nup+2), max(1,ndn-2):min(Ndn+1,ndn+2))
        ) for nup=1:Nup+1, ndn=1:Ndn+1 ]
              ) for k=1:d ]

    nblocks = wrap( (idx...) -> 0)
    for ( (k₁,n₁,s₁,idx₁), (k₂,n₂,s₂,idx₂) ) in edge_labels(graph)
      m,occ_m,α = graph[(k₁,n₁,s₁,idx₁), (k₂,n₂,s₂,idx₂)]
      nup,ndn = n₁
      mup,mdn = m
      nblocks[k₁][occupation(n₁,n₂)][nup,ndn][occ_m][mup,mdn] += 1
    end

    # Here we obtain the blocks of the TT-cores indexed by n and occupation, 
    # for any given block indexed by l and occupation from the original TT-core,
    # stored in sparse COO format: row/column state indices and corresponding multiplier
    coo_blocks = wrap( (k,nup,ndn,occ_n,mup,mdn,occ_m) -> (Vector{Int}(undef, nblocks[k][occ_n][nup,ndn][occ_m][mup,mdn]), 
                                                           Vector{Int}(undef, nblocks[k][occ_n][nup,ndn][occ_m][mup,mdn]), 
                                                           Vector{T}(  undef, nblocks[k][occ_n][nup,ndn][occ_m][mup,mdn])) )
    j = wrap( (idx...) -> 0)
    for ((k₁,n₁,s₁,idx₁), (k₂,n₂,s₂,idx₂)) in edge_labels(graph)
      nup,ndn = n₁
      i₁ = state_to_index[k₁][nup,ndn][(s₁,idx₁)]
      i₂ = state_to_index[k₂][n₂...][(s₂,idx₂)]
      m,occ_m,α = graph[(k₁,n₁,s₁,idx₁), (k₂,n₂,s₂,idx₂)]

      mup,mdn = m
      j[k₁][occupation(n₁,n₂)][nup,ndn][occ_m][mup,mdn] += 1
      J = j[k₁][occupation(n₁,n₂)][nup,ndn][occ_m][mup,mdn]
      block = coo_blocks[k₁][occupation(n₁,n₂)][nup,ndn][occ_m][mup,mdn]

      block[1][J], block[2][J], block[3][J] = i₁, i₂, α
    end

    # Post-processing to compute alternative CSR and CSC formats of the same sparse objects
    csc_blocks = wrap( (idx...) -> (OffsetVector(Int[], 1:0),Int[],T[]) )
    csr_blocks = wrap( (idx...) -> (OffsetVector(Int[], 1:0),Int[],T[]) )

    for k=1:d, occ_n in (:○○, :up, :dn, :●●), (nup,ndn) in qn[k][occ_n], occ_m in (:○○, :up, :dn, :●●), mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)
      nnz   = nblocks[k][occ_n][nup,ndn][occ_m][mup,mdn]
      if nnz>0
        block = coo_blocks[k][occ_n][nup,ndn][occ_m][mup,mdn]
        p_row = sortperm(1:nnz, lt= (i,j)->( (block[1][i],block[2][i]) < (block[1][j],block[2][j]) ) )

        permute!(block[1], p_row)
        permute!(block[2], p_row)
        permute!(block[3], p_row)
        row_range = block[1][1]:block[1][end]+1

        row_starts = OffsetVector(zeros(Int,length(row_range)), row_range)
        row = block[1][1]
        row_starts[row] = 1
        for i=1:nnz
          if block[1][i] > row
            row_starts[row+1:block[1][i]] .= i
            row = block[1][i]
          end
        end
        row_starts[end] = nnz+1

        csr_blocks[k][occ_n][nup,ndn][occ_m][mup,mdn] = (row_starts, block[2], block[3])

        p_col = sortperm(1:nnz, lt= (i,j)->( (block[2][i],block[1][i]) < (block[2][j],block[1][j]) ) )
        col_range = block[2][p_col[1]]:block[2][p_col[end]]+1

        col_starts = OffsetVector(zeros(Int,length(col_range)), col_range)
        col = block[2][p_col[1]]
        col_starts[col] = 1
        for i=1:nnz
          if block[2][p_col[i]] > col
            col_starts[col+1:block[2][p_col[i]]] .= i
            col = block[2][p_col[i]]
          end
        end
        col_starts[end] = nnz+1

        csc_blocks[k][occ_n][nup,ndn][occ_m][mup,mdn] = (col_starts, block[1][p_col], block[3][p_col] )
      end
    end

    return new{T,Nup,Ndn,d}(states, state_qns, coo_blocks, csr_blocks, csc_blocks, graph)
  end
end

function SparseHamiltonian(t::Matrix{T}, v::Array{T,4}, ψ::TTvector{T,Nup,Ndn,d}; ϵ=eps()) where {T<:Number,Nup,Ndn,d}
  return SparseHamiltonian(t,v,Val(Nup),Val(Ndn),Val(d); ϵ=ϵ)
end

function QNTensorTrains.row_ranks(H::SparseHamiltonian{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d,M}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = x.k
  if k==1
    return row_ranks(x)
  else
    state_qns = H.state_qns[k]
    ranks = zeros(Int,Nup+1,Ndn+1)
    for (nup,ndn) in row_qn(x)
      ranks[nup,ndn] = isempty(state_qns[nup,ndn]) ? 0 : sum(row_rank(x,mup,mdn) for (mup,mdn) in state_qns[nup,ndn])
    end
    return ranks
  end
end

function row_ranges(H::SparseHamiltonian{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d,M}) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = x.k
  state_qns = H.state_qns[k]
  row_ranges = Matrix{Vector{UnitRange{Int}}}(undef,Nup+1,Ndn+1)

  if k==1
    row_ranges[1,1] = [(1:row_ranks(x)[1,1])]
  else
    for (nup,ndn) in row_qn(x)
      ends = cumsum(row_rank(x,mup,mdn) for (mup,mdn) in state_qns[nup,ndn])
      row_ranges[nup,ndn] = [(i==1 ? 1 : ends[i-1]+1):ends[i] for i=1:length(state_qns[nup,ndn])]
    end
  end
  return row_ranges
end

function QNTensorTrains.col_ranks(H::SparseHamiltonian{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  k = x.k
  if k==d
    return col_ranks(x)
  else
    state_qns = H.state_qns[k+1]
    ranks = zeros(Int,Nup+1,Ndn+1)
    for (nup,ndn) in col_qn(x)
      ranks[nup,ndn] = isempty(state_qns[nup,ndn]) ? 0 : sum(col_rank(x,mup,mdn) for (mup,mdn) in state_qns[nup,ndn])
    end
    return ranks
  end
end

function col_ranges(H::SparseHamiltonian{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  k = x.k
  state_qns = H.state_qns[k+1]
  col_ranges = Matrix{Vector{UnitRange{Int}}}(undef,Nup+1,Ndn+1)

  if k==d
    col_ranges[Nup+1,Ndn+1] = [(1:col_ranks(x)[Nup+1,Ndn+1])]
  else
    for (nup,ndn) in col_qn(x)
      ends = cumsum(col_rank(x,mup,mdn) for (mup,mdn) in state_qns[nup,ndn])
      col_ranges[nup,ndn] = [(i==1 ? 1 : ends[i-1]+1):ends[i] for i=1:length(state_qns[nup,ndn])]
    end
  end
  return col_ranges
end

function Base.:*(Fᴸ::Frame{T,Nup,Ndn,d}, H::SparseHamiltonian{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert site(Fᴸ) == site(x)
    @assert Fᴸ.col_ranks == row_ranks(H,x)
  end
  I,J = row_ranges(H,x), col_ranges(H,x)

  y = SparseCore{T,Nup,Ndn,d}(x.k, Fᴸ.row_ranks, col_ranks(H,x))
  COO = H.coo_blocks[x.k]
  CSR = H.csr_blocks[x.k]
  CSC = H.csc_blocks[x.k]

  function innerkernel!(Y,F,X,I,J,COO,CSC,CSR)
    nnz = length(COO[1])
    nrow = length(CSR[1])
    ncol = length(CSC[1])
    if ncol < min(nnz,nrow)
      col_index, row_index, v = CSC
      for j in axes(col_index,1)[begin:end-1]
        if col_index[j+1] > col_index[j]+1
          Fj = zeros(T, size(F,1), size(X,1))
          for index in col_index[j]:col_index[j+1]-1
            i, α = row_index[index], v[index]
            axpy!(α, view(F,:,I[i]), Fj)
          end
          mul!( view(Y,:,J[j]), Fj, X, 1, 1)
        elseif col_index[j+1] == col_index[j]+1
          i, α = row_index[col_index[j]], v[col_index[j]]
          mul!(view(Y,:,J[j]), view(F,:,I[i]), X, α, 1)
        end
      end
    elseif nrow < min(nnz,ncol)
      row_index, col_index, v = CSR
      for i in axes(row_index,1)[begin:end-1]
        if row_index[i+1] > row_index[i]+1
          FiX = view(F,:,I[i])*X
          for index in row_index[i]:row_index[i+1]-1
            j, α = col_index[index], v[index]
            axpy!(α, FiX, view(Y,:,J[j]))
          end
        elseif row_index[i+1] == row_index[i]+1
          j, α = col_index[row_index[i]], v[row_index[i]]
          mul!(view(Y,:,J[j]), view(F,:,I[i]), X, α, 1)
        end
      end
    else # nnz < min(nrow,ncol)
      for (i,j,α) in zip(COO...)
        mul!(view(Y,:,J[j]), view(F,:,I[i]), X, α, 1)
      end
    end
  end

  function outerkernel!(Y,F,x,nup,ndn,In,Jn,COO,CSR,CSC)
    for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)
      innerkernel!(Y, F, ○○(x,mup,mdn), In, Jn, COO.○○[mup,mdn], CSR.○○[mup,mdn], CSC.○○[mup,mdn])
      innerkernel!(Y, F, up(x,mup,mdn), In, Jn, COO.up[mup,mdn], CSR.up[mup,mdn], CSC.up[mup,mdn])
      innerkernel!(Y, F, dn(x,mup,mdn), In, Jn, COO.dn[mup,mdn], CSR.dn[mup,mdn], CSC.dn[mup,mdn])
      innerkernel!(Y, F, ●●(x,mup,mdn), In, Jn, COO.●●[mup,mdn], CSR.●●[mup,mdn], CSC.●●[mup,mdn])
    end
  end

  @sync begin
    for (nup,ndn) in it_○○(y)
      @Threads.spawn outerkernel!(○○(y,nup,ndn),block(Fᴸ,nup,ndn),x,nup,ndn,I[nup,ndn],J[nup  ,ndn  ],COO.○○[nup,ndn],CSR.○○[nup,ndn],CSC.○○[nup,ndn])
    end
    for (nup,ndn) in it_up(y)
      @Threads.spawn outerkernel!(up(y,nup,ndn),block(Fᴸ,nup,ndn),x,nup,ndn,I[nup,ndn],J[nup+1,ndn  ],COO.up[nup,ndn],CSR.up[nup,ndn],CSC.up[nup,ndn])
    end
    for (nup,ndn) in it_dn(y)
      @Threads.spawn outerkernel!(dn(y,nup,ndn),block(Fᴸ,nup,ndn),x,nup,ndn,I[nup,ndn],J[nup  ,ndn+1],COO.dn[nup,ndn],CSR.dn[nup,ndn],CSC.dn[nup,ndn])
    end
    for (nup,ndn) in it_●●(y)
      @Threads.spawn outerkernel!(●●(y,nup,ndn),block(Fᴸ,nup,ndn),x,nup,ndn,I[nup,ndn],J[nup+1,ndn+1],COO.●●[nup,ndn],CSR.●●[nup,ndn],CSC.●●[nup,ndn])
    end
  end

  return y
end

function Base.:*(H::SparseHamiltonian{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d}, Fᴿ::Frame{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert site(Fᴿ) == site(x)+1
    @assert Fᴿ.row_ranks == col_ranks(H,x)
  end
  I,J = row_ranges(H,x), col_ranges(H,x)

  y = SparseCore{T,Nup,Ndn,d}(x.k, row_ranks(H,x), Fᴿ.col_ranks)
  COO = H.coo_blocks[x.k]
  CSR = H.csr_blocks[x.k]
  CSC = H.csc_blocks[x.k]

  function innerkernel!(Y,X,F,I,J,COO,CSC,CSR)
    nnz = length(COO[1])
    nrow = length(CSR[1])
    ncol = length(CSC[1])
    if ncol < min(nnz,nrow)
      col_index, row_index, v = CSC
      for j in axes(col_index,1)[begin:end-1]
        if col_index[j+1] > col_index[j]+1
          XFj = X*view(F,J[j],:)
          for index in col_index[j]:col_index[j+1]-1
            i, α = row_index[index], v[index]
            axpy!(α, XFj, view(Y,I[i],:))
          end
        elseif col_index[j+1] == col_index[j]+1
          i, α = row_index[col_index[j]], v[col_index[j]]
          mul!(view(Y,I[i],:), X, view(F,J[j],:), α, 1)
        end
      end
    elseif nrow < min(nnz,ncol)
      row_index, col_index, v = CSR
      for i in axes(row_index,1)[begin:end-1]
        if row_index[i+1] > row_index[i]+1
          Fi = zeros(T, size(X,2), size(F,2))
          for index in row_index[i]:row_index[i+1]-1
            j, α = col_index[index], v[index]
            axpy!(α, view(F,J[j],:), Fi)
          end
          mul!( view(Y,I[i],:), X, Fi, 1, 1)
        elseif row_index[i+1] == row_index[i]+1
          j, α = col_index[row_index[i]], v[row_index[i]]
          mul!(view(Y,I[i],:), X, view(F,J[j],:), α, 1)
        end
      end
    else # nnz < min(nrow,ncol)
      for (i,j,α) in zip(COO...)
        mul!(view(Y,I[i],:), X, view(F,J[j],:), α, 1)
      end
    end
  end

  function outerkernel!(Y,x,F,nup,ndn,I,J,COO,CSR,CSC)
    for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)
      innerkernel!(Y, ○○(x,mup,mdn), F, I, J, COO.○○[mup,mdn], CSR.○○[mup,mdn], CSC.○○[mup,mdn])
      innerkernel!(Y, up(x,mup,mdn), F, I, J, COO.up[mup,mdn], CSR.up[mup,mdn], CSC.up[mup,mdn])
      innerkernel!(Y, dn(x,mup,mdn), F, I, J, COO.dn[mup,mdn], CSR.dn[mup,mdn], CSC.dn[mup,mdn])
      innerkernel!(Y, ●●(x,mup,mdn), F, I, J, COO.●●[mup,mdn], CSR.●●[mup,mdn], CSC.●●[mup,mdn])
    end
  end

  @sync begin
    for (nup,ndn) in it_○○(y)
      @Threads.spawn outerkernel!(○○(y,nup,ndn), x, block(Fᴿ,nup  ,ndn  ), nup, ndn, I[nup,ndn], J[nup  ,ndn  ], COO.○○[nup,ndn], CSR.○○[nup,ndn], CSC.○○[nup,ndn])
    end
    for (nup,ndn) in it_up(y)
      @Threads.spawn outerkernel!(up(y,nup,ndn), x, block(Fᴿ,nup+1,ndn  ), nup, ndn, I[nup,ndn], J[nup+1,ndn  ], COO.up[nup,ndn], CSR.up[nup,ndn], CSC.up[nup,ndn])
    end
    for (nup,ndn) in it_dn(y)
      @Threads.spawn outerkernel!(dn(y,nup,ndn), x, block(Fᴿ,nup  ,ndn+1), nup, ndn, I[nup,ndn], J[nup  ,ndn+1], COO.dn[nup,ndn], CSR.dn[nup,ndn], CSC.dn[nup,ndn])
    end
    for (nup,ndn) in it_●●(y)
      @Threads.spawn outerkernel!(●●(y,nup,ndn), x, block(Fᴿ,nup+1,ndn+1), nup, ndn, I[nup,ndn], J[nup+1,ndn+1], COO.●●[nup,ndn], CSR.●●[nup,ndn], CSC.●●[nup,ndn])
    end
  end

  return y
end

function Base.:*(l::AdjointCore{T,Nup,Ndn,d}, H::SparseHamiltonian{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert site(l) == site(x)
    @assert col_ranks(l) == row_ranks(H,x)
  end
  I,J = row_ranges(H,x), col_ranges(H,x)

  y = Frame{T,Nup,Ndn,d}(x.k+1, row_ranks(l), col_ranks(H,x))
  COO = H.coo_blocks[x.k]
  CSR = H.csr_blocks[x.k]
  CSC = H.csc_blocks[x.k]

  function innerkernel!(Y,L,X,I,J,COO,CSC,CSR)
    nnz = length(COO[1])
    nrow = length(CSR[1])
    ncol = length(CSC[1])
    if ncol < min(nnz,nrow)
      col_index, row_index, v = CSC
      for j in axes(col_index,1)[begin:end-1]
        if col_index[j+1] > col_index[j]+1
          Lj = zeros(T, size(L,1), size(X,1))
          for index in col_index[j]:col_index[j+1]-1
            i, α = row_index[index], v[index]
            axpy!(α, view(L,:,I[i]), Lj)
          end
          mul!( view(Y,:,J[j]), Lj, X, 1, 1)
        elseif col_index[j+1] == col_index[j]+1
          i, α = row_index[col_index[j]], v[col_index[j]]
          mul!(view(Y,:,J[j]), view(L,:,I[i]), X, α, 1)
        end
      end
    elseif nrow < min(nnz,ncol)
      row_index, col_index, v = CSR
      for i in axes(row_index,1)[begin:end-1]
        if row_index[i+1] > row_index[i]+1
          LiX = view(L,:,I[i])*X
          for index in row_index[i]:row_index[i+1]-1
            j, α = col_index[index], v[index]
            axpy!(α, LiX, view(Y,:,J[j]))
          end
        elseif row_index[i+1] == row_index[i]+1
          j, α = col_index[row_index[i]], v[row_index[i]]
          mul!(view(Y,:,J[j]), view(L,:,I[i]), X, α, 1)
        end
      end
    else # nnz < min(nrow,ncol)
      for (i,j,α) in zip(COO...)
        mul!(view(Y,:,J[j]), view(L,:,I[i]), X, α, 1)
      end
    end
  end

  function outerkernel!(Y,L,x,nup,ndn,I,J,COO,CSR,CSC)
    for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)
      innerkernel!(Y, L, ○○(x,mup,mdn), I, J, COO.○○[mup,mdn], CSR.○○[mup,mdn], CSC.○○[mup,mdn])
      innerkernel!(Y, L, up(x,mup,mdn), I, J, COO.up[mup,mdn], CSR.up[mup,mdn], CSC.up[mup,mdn])
      innerkernel!(Y, L, dn(x,mup,mdn), I, J, COO.dn[mup,mdn], CSR.dn[mup,mdn], CSC.dn[mup,mdn])
      innerkernel!(Y, L, ●●(x,mup,mdn), I, J, COO.●●[mup,mdn], CSR.●●[mup,mdn], CSC.●●[mup,mdn])
    end
  end

  @sync begin
    for (nup,ndn) in it_○○(parent(l))
      @Threads.spawn outerkernel!(block(y,nup  ,ndn  ), ○○(l,nup  ,ndn  ), x, nup, ndn, I[nup,ndn], J[nup  ,ndn  ], COO.○○[nup,ndn], CSR.○○[nup,ndn], CSC.○○[nup,ndn])
    end
    for (nup,ndn) in it_up(parent(l))
      @Threads.spawn outerkernel!(block(y,nup+1,ndn  ), up(l,nup+1,ndn  ), x, nup, ndn, I[nup,ndn], J[nup+1,ndn  ], COO.up[nup,ndn], CSR.up[nup,ndn], CSC.up[nup,ndn])
    end
    for (nup,ndn) in it_dn(parent(l))
      @Threads.spawn outerkernel!(block(y,nup  ,ndn+1), dn(l,nup  ,ndn+1), x, nup, ndn, I[nup,ndn], J[nup  ,ndn+1], COO.dn[nup,ndn], CSR.dn[nup,ndn], CSC.dn[nup,ndn])
    end
    for (nup,ndn) in it_●●(parent(l))
      @Threads.spawn outerkernel!(block(y,nup+1,ndn+1), ●●(l,nup+1,ndn+1), x, nup, ndn, I[nup,ndn], J[nup+1,ndn+1], COO.●●[nup,ndn], CSR.●●[nup,ndn], CSC.●●[nup,ndn])
    end
  end

  return y
end

function Base.:*(H::SparseHamiltonian{T,Nup,Ndn,d}, x::SparseCore{T,Nup,Ndn,d}, r::AdjointCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert site(r) == site(x)
    @assert row_ranks(r) == col_ranks(H,x)
  end
  I,J = row_ranges(H,x), col_ranges(H,x)

  y = Frame{T,Nup,Ndn,d}(x.k, row_ranks(H,x), col_ranks(r))
  COO = H.coo_blocks[x.k]
  CSR = H.csr_blocks[x.k]
  CSC = H.csc_blocks[x.k]


  function innerkernel!(Y,X,R,I,J,COO,CSC,CSR)
    nnz = length(COO[1])
    nrow = length(CSR[1])
    ncol = length(CSC[1])
    if ncol < min(nnz,nrow)
      col_index, row_index, v = CSC
      for j in axes(col_index,1)[begin:end-1]
        if col_index[j+1] > col_index[j]+1
          XRj = X*view(R,J[j],:)
          for index in col_index[j]:col_index[j+1]-1
            i, α = row_index[index], v[index]
            axpy!(α, XRj, view(Y,I[i],:))
          end
        elseif col_index[j+1] == col_index[j]+1
          i, α = row_index[col_index[j]], v[col_index[j]]
          mul!(view(Y,I[i],:), X, view(R,J[j],:), α, 1)
        end
      end
    elseif nrow < min(nnz,ncol)
      row_index, col_index, v = CSR
      for i in axes(row_index,1)[begin:end-1]
        if row_index[i+1] > row_index[i]+1
          Ri = zeros(T, size(X,2), size(R,2))
          for index in row_index[i]:row_index[i+1]-1
            j, α = col_index[index], v[index]
            axpy!(α, view(R,J[j],:), Ri)
          end
          mul!( view(Y,I[i],:), X, Ri, 1, 1)
        elseif row_index[i+1] == row_index[i]+1
          j, α = col_index[row_index[i]], v[row_index[i]]
          mul!(view(Y,I[i],:), X, view(R,J[j],:), α, 1)
        end
      end
    else # nnz < min(nrow,ncol)
      for (i,j,α) in zip(COO...)
        mul!(view(Y,I[i],:), X, view(R,J[j],:), α, 1)
      end
    end
  end

  function outerkernel!(Y,x,R,nup,ndn,I,J,COO,CSR,CSC)
    for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)
      innerkernel!(Y, ○○(x,mup,mdn), R, I, J, COO.○○[mup,mdn], CSR.○○[mup,mdn], CSC.○○[mup,mdn])
      innerkernel!(Y, up(x,mup,mdn), R, I, J, COO.up[mup,mdn], CSR.up[mup,mdn], CSC.up[mup,mdn])
      innerkernel!(Y, dn(x,mup,mdn), R, I, J, COO.dn[mup,mdn], CSR.dn[mup,mdn], CSC.dn[mup,mdn])
      innerkernel!(Y, ●●(x,mup,mdn), R, I, J, COO.●●[mup,mdn], CSR.●●[mup,mdn], CSC.●●[mup,mdn])
    end
  end

  @sync begin
    for (nup,ndn) in it_○○(parent(r))
      @Threads.spawn outerkernel!(block(y,nup,ndn), x, ○○(r,nup  ,ndn  ), nup, ndn, I[nup,ndn], J[nup  ,ndn  ], COO.○○[nup,ndn], CSR.○○[nup,ndn], CSC.○○[nup,ndn])
    end
    for (nup,ndn) in it_up(parent(r))
      @Threads.spawn outerkernel!(block(y,nup,ndn), x, up(r,nup+1,ndn  ), nup, ndn, I[nup,ndn], J[nup+1,ndn  ], COO.up[nup,ndn], CSR.up[nup,ndn], CSC.up[nup,ndn])
    end
    for (nup,ndn) in it_dn(parent(r))
      @Threads.spawn outerkernel!(block(y,nup,ndn), x, dn(r,nup  ,ndn+1), nup, ndn, I[nup,ndn], J[nup  ,ndn+1], COO.dn[nup,ndn], CSR.dn[nup,ndn], CSC.dn[nup,ndn])
    end
    for (nup,ndn) in it_●●(parent(r))
      @Threads.spawn outerkernel!(block(y,nup,ndn), x, ●●(r,nup+1,ndn+1), nup, ndn, I[nup,ndn], J[nup+1,ndn+1], COO.●●[nup,ndn], CSR.●●[nup,ndn], CSC.●●[nup,ndn])
    end
  end
  
  return y
end

function Base.:*(H::SparseHamiltonian{T,Nup,Ndn,d}, X::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef, d)
  for k=1:d
    x = core(X,k)
    cores[k] = SparseCore{T,Nup,Ndn,d}(k, row_ranks(H,x), col_ranks(H,x))
    I,J = row_ranges(H,x), col_ranges(H,x)

    COO = H.coo_blocks[k]
    @sync begin
      for (nup,ndn) in it_○○(cores[k])
        @Threads.spawn let Y = ○○(cores[k],nup,ndn)
          for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)
            for (i,j,α) in zip(COO.○○[nup,ndn].○○[mup,mdn]...)
              axpy!(α, ○○(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup,ndn][j]))
            end
            for (i,j,α) in zip(COO.○○[nup,ndn].up[mup,mdn]...)
              axpy!(α, up(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup,ndn][j]))
            end
            for (i,j,α) in zip(COO.○○[nup,ndn].dn[mup,mdn]...)
              axpy!(α, dn(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup,ndn][j]))
            end
            for (i,j,α) in zip(COO.○○[nup,ndn].●●[mup,mdn]...)
              axpy!(α, ●●(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup,ndn][j]))
            end
          end
        end
      end
      for (nup,ndn) in it_up(cores[k])
        @Threads.spawn let Y = up(cores[k],nup,ndn)
          for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)
            for (i,j,α) in zip(COO.up[nup,ndn].○○[mup,mdn]...)
              axpy!(α, ○○(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup+1,ndn][j]))
            end
            for (i,j,α) in zip(COO.up[nup,ndn].up[mup,mdn]...)
              axpy!(α, up(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup+1,ndn][j]))
            end
            for (i,j,α) in zip(COO.up[nup,ndn].dn[mup,mdn]...)
              axpy!(α, dn(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup+1,ndn][j]))
            end
            for (i,j,α) in zip(COO.up[nup,ndn].●●[mup,mdn]...)
              axpy!(α, ●●(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup+1,ndn][j]))
            end
          end
        end
      end
      for (nup,ndn) in it_dn(cores[k])
        @Threads.spawn let Y = dn(cores[k],nup,ndn)
          for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)
            for (i,j,α) in zip(COO.dn[nup,ndn].○○[mup,mdn]...)
              axpy!(α, ○○(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup,ndn+1][j]))
            end
            for (i,j,α) in zip(COO.dn[nup,ndn].up[mup,mdn]...)
              axpy!(α, up(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup,ndn+1][j]))
            end
            for (i,j,α) in zip(COO.dn[nup,ndn].dn[mup,mdn]...)
              axpy!(α, dn(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup,ndn+1][j]))
            end
            for (i,j,α) in zip(COO.dn[nup,ndn].●●[mup,mdn]...)
              axpy!(α, ●●(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup,ndn+1][j]))
            end
          end
        end
      end
      for (nup,ndn) in it_●●(cores[k])
        @Threads.spawn let Y = ●●(cores[k],nup,ndn)
          for mup=max(1,nup-2):min(Nup+1,nup+2), mdn=max(1,ndn-2):min(Ndn+1,ndn+2)
            for (i,j,α) in zip(COO.●●[nup,ndn].○○[mup,mdn]...)
              axpy!(α, ○○(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup+1,ndn+1][j]))
            end
            for (i,j,α) in zip(COO.●●[nup,ndn].up[mup,mdn]...)
              axpy!(α, up(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup+1,ndn+1][j]))
            end
            for (i,j,α) in zip(COO.●●[nup,ndn].dn[mup,mdn]...)
              axpy!(α, dn(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup+1,ndn+1][j]))
            end
            for (i,j,α) in zip(COO.●●[nup,ndn].●●[mup,mdn]...)
              axpy!(α, ●●(x,mup,mdn), view(Y, I[nup,ndn][i], J[nup+1,ndn+1][j]))
            end
          end
        end
      end
    end
  end
  return cores2tensor(cores)
end

function RayleighQuotient(H::SparseHamiltonian{T,Nup,Ndn,d}, x::TTvector{T,Nup,Ndn,d}; orthogonalize::Bool=false) where {T<:Number,Nup,Ndn,d}

# to = TimerOutput()
# @timeit to "Orthogonalize" begin
  if (orthogonalize)
    QNTensorTrains.leftOrthogonalize!(x)
  end
# end

# @timeit to "Frame 1" begin
  p = IdFrame(Val(d), Val(Nup), Val(Ndn), 1)
# end
# @timeit to "Contractions" begin
  for k=1:d
    p = adjoint(core(x,k)) * (p*H*core(x,k))
  end
# end
# display(to)

  return block(p,Nup,Ndn)[1]
end

# using TimerOutputs
function xᵀHy(x::TTvector{T,Nup,Ndn,d}, H::SparseHamiltonian{T,Nup,Ndn,d}, y::TTvector{T,Nup,Ndn,d}; orthogonalize::Bool=false) where {T<:Number,Nup,Ndn,d}

# to = TimerOutput()
# @timeit to "Orthogonalize" begin
  if (orthogonalize)
    QNTensorTrains.leftOrthogonalize!(x)
    QNTensorTrains.leftOrthogonalize!(y)
  end
# end

# @timeit to "Frame 1" begin
  p = IdFrame(Val(d), Val(Nup), Val(Ndn), 1)
# end
# @timeit to "Contractions" begin
  for k=1:d
    p = adjoint(core(x,k)) * (p*H*core(y,k))
  end
# end
# display(to)

  return block(p,Nup,Ndn)[1]
end


end