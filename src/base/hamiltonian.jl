module Hamiltonian
using ..QNTensorTrains: Frame, IdFrame, SparseCore, AdjointSparseCore, UnsafeSparseCore, TTvector
using ..QNTensorTrains: site, core, cores2tensor, Id_view, S_view, Adag_view, A_view, AdagA_view
using LinearAlgebra, OffsetArrays

export SparseHamiltonian, H_matvec_core, RayleighQuotient, xᵀHy

const Ts = Tuple{NTuple{3,Int64},Union{Int64, NTuple{2,Int64}}}


function reduce(v::Array{T,4}; tol=16eps(real(T))) where {T<:Number}
  d = size(v,1)
  @boundscheck @assert size(v) == (d,d,d,d)

  v_reduced = zeros(T,size(v))
  for i=1:d, k=1:d, j=i+1:d, l=k+1:d
    v_reduced[i,j,k,l] = v[i,j,k,l]+v[j,i,l,k]-v[j,i,k,l]-v[i,j,l,k]
  end
  v_reduced[abs.(v_reduced).<tol] .= 0

  return v_reduced
end
 
function shift_ranks(ranks::AbstractVector{Int}, 
                      flux::Int, nl::Int, nr::Int, N::Int)
  @boundscheck length(ranks) ≥ flux 
  @boundscheck @assert nl ≥ 0 && nr ≥ 0 && nl + nr ≤ N

  new_ranks = similar(ranks)

  start = min(max(nl,   axes(ranks,1)[begin]+(flux>0 ? flux : 0)), lastindex(ranks )+1)
  stop  = max(min(N-nr, axes(ranks,1)[end]  +(flux<0 ? flux : 0)), firstindex(ranks)-1)
  qn = start:stop

  new_ranks[qn] = ranks[qn.-flux]
  new_ranks[begin:start-1]  .= 0
  new_ranks[stop+1:end] .= 0

  return new_ranks
end

using Graphs, MetaGraphsNext

struct SparseHamiltonian{T<:Number,N,d}
  states::Vector{Vector{Ts}}
  blocks::Vector{Dict{NTuple{2,Ts},Function}}
  graph::MetaGraph

  function SparseHamiltonian(t::Matrix{T}, v::Array{T,4}, ::Val{N}, ::Val{d}; ϵ=eps(), reduced=false) where {T<:Number,N,d}
    @boundscheck @assert size(t) == (d,d)
    @boundscheck @assert size(v) == (d,d,d,d)
    if !reduced
      v = reduce(v)
    end
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
    #     # One body states
    #     ( 0,0,1), #  1 #       n ≤ i,j
    #     ( 1,1,0), #  2 # i   < n ≤   j
    #     (-1,0,1), #  3 #   j < n ≤ i
    #     ( 0,1,0), #  4 # i,j < n
    #     # Two body states
    #     ( 0,0,2), #  5 #           n ≤ i<j,k<l
    #     ( 1,1,1), #  6 # i       < n ≤   j,k<l
    #     (-1,0,2), #  7 #     k   < n ≤ i<j,  l
    #     ( 0,1,1), #  8 # i,  k   < n ≤   j,  l
    #     ( 2,2,0), #  9 # i<j     < n ≤     k<l
    #     (-2,0,2), # 10 #     k<l < n ≤ i<j
    #     ( 1,2,0), # 11 # i<j,k   < n ≤       l
    #     (-1,1,1), # 12 # i,  k<l < n ≤   j
    #     ( 0,2,0)  # 13 # i<j,k<l < n
    #   )

############################################################
  function ×(α::Number, op::Function)
    αop = (A, flux, nl, nr) -> begin
      B, s... = op(A,flux,nl,nr)
      return lmul!(α,B), s...
    end

    return αop
  end

    # One body states
    function vertex(n, i,j)
      if n ≤ min(i,j)
        state = ( 0,0,1); index =         1
      elseif i < n ≤ j
        state = ( 1,1,0); index = (n ≤ hd ? i : j)
      elseif j < n ≤ i
        state = (-1,0,1); index = (n ≤ hd ? j : i)
      else # max(i,j) < n
        state = ( 0,1,0); index =         1
      end
      return (n, state, index)
    end

    function weighted_edge(i,j)
      return sort([hd, i,j])[2]
    end

    function set_edge(n, i,j)
      @assert 1 ≤ n ≤ d
      v₋ = vertex(n,   i,j)
      v₊ = vertex(n+1, i,j)
      stop_path = !add_vertex!(graph, v₋) & !add_vertex!(graph, v₊)

      edge_data = ( n == i == j ? AdagA_view :
                    n == j      ? A_view     :
                    n == i      ? Adag_view  :
                    i < n < j || j < n < i ? 
                                  S_view : Id_view )
      add_edge!(graph, v₋, v₊, edge_data)

      return stop_path
    end

    function set_edge(n, i,j, w)
      @assert 1 ≤ n ≤ d
      v₋ = vertex(n,   i,j)
      v₊ = vertex(n+1, i,j)
      stop_path = !add_vertex!(graph, v₋) & !add_vertex!(graph, v₊)

      edge_data = ( n == i == j ? AdagA_view :
                    n == j      ? A_view     :
                    n == i      ? Adag_view  :
                    i < n < j || j < n < i ? 
                                  S_view : Id_view )
      add_edge!(graph, v₋, v₊, w × edge_data)

      return stop_path
    end

    # Two body states
    function vertex(n, i,j,k,l)
      @assert i<j && k<l
      if n ≤ min(i,k)
        state = ( 0,0,2);    index =         1
      elseif i < n ≤ min(j,k)
        state = ( 1,1,1);    index =         i
      elseif k < n ≤ min(i,l)
        state = (-1,0,2);    index =         k
      elseif max(i,k) < n ≤ min(j,l)
        state = ( 0,1,1);    index = (n ≤ hd ?  (i,k) : (j,l))
      elseif j < n ≤ k
        state = ( 2,2,0);    index = (n ≤ hd ?  (i,j) : (k,l))
      elseif l < n ≤ i
        state = (-2,0,2);    index = (n ≤ hd ?  (k,l) : (i,j))
      elseif max(j,k) < n ≤ l
        state = ( 1,2,0);    index =         l
      elseif max(i,l) < n ≤ j
        state = (-1,1,1);    index =         j
      else # max(j,l) < n
        state = ( 0,2,0);    index =         1
      end 
      return n, state, index
    end

    function weighted_edge(i,j,k,l)
      @assert i<j && k<l
      return sort([hd, i,j,k,l])[3]
    end 

    function set_edge(n, i,j,k,l)
      @assert i<j && k<l && 1 ≤ n ≤ d
      v₋ = vertex(n,   i,j,k,l)
      v₊ = vertex(n+1, i,j,k,l)
      stop_path = !add_vertex!(graph, v₋) & !add_vertex!(graph, v₊)

      edge_data = ( n ∈ (i,j) && n ∈ (k,l) ? AdagA_view :
                    n ∈ (k,l)              ? A_view     :
                    n ∈ (i,j)              ? Adag_view  :
                    i < n < min(j,k) || k < n < min(i,l) || max(j,k) < n < l || max(i,l) < n < j ?
                                             S_view : Id_view )
      add_edge!(graph, v₋, v₊, edge_data)

      return stop_path
    end

    function set_edge(n, i,j,k,l, w)
      @assert i<j && k<l && 1 ≤ n ≤ d
      v₋ = vertex(n,   i,j,k,l)
      v₊ = vertex(n+1, i,j,k,l)
      stop_path = !add_vertex!(graph, v₋) & !add_vertex!(graph, v₊)

      edge_data = ( n ∈ (i,j) && n ∈ (k,l) ? AdagA_view :
                    n ∈ (k,l)              ? A_view     :
                    n ∈ (i,j)              ? Adag_view  :
                    i < n < min(j,k) || k < n < min(i,l) || max(j,k) < n < l || max(i,l) < n < j ?
                                             S_view : Id_view )
      add_edge!(graph, v₋, v₊, w × edge_data)

      return stop_path
    end

    graph = MetaGraph(
      DiGraph();                       # Initialize empty graph
      label_type=Tuple{Int,NTuple{3, Int},Union{Int,NTuple{2,Int}}},          
                                       # site, state and index
      vertex_data_type=Nothing,        # State details 
      edge_data_type=Function,         # Coefficient and single-site matrix-free core operation
      weight_function=ed -> ed[1],
      default_weight=0.,
      graph_data="Hamiltonian action graph",                  # tag for the whole graph
    )

    for i=1:d, j=1:d
      if abs(t[i,j]) > ϵ
        nₘ = weighted_edge(i,j)
        set_edge(nₘ, i,j, t[i,j])
        for n=nₘ+1:d
          stop_path = set_edge(n, i,j) 
          stop_path && break
        end
        for n = nₘ-1:-1:1
          stop_path = set_edge(n, i,j) 
          stop_path && break
        end
      end
    end

    for i=1:d-1, j=i+1:d, k=1:d-1, l=k+1:d
      if abs(v[i,j,k,l]) > ϵ
        nₘ = weighted_edge(i,j,k,l)
        set_edge(nₘ, i,j,k,l, v[i,j,k,l])
        for n=nₘ+1:d
          stop_path = set_edge(n, i,j,k,l) 
          stop_path && break
        end
        for n = nₘ-1:-1:1
          stop_path = set_edge(n, i,j,k,l) 
          stop_path && break
        end
      end
    end

    nstates = [0 for n=1:d+1]
    for (n,s,idx) in labels(graph)
      nstates[n] += 1
    end

    states = [ Vector{Ts}(undef, nstates[n]) for n=1:d+1]
    i = [0 for n=1:d+1]
    for (n, s, idx) in labels(graph)
      i[n]+=1
      states[n][i[n]] = (s, idx)
    end
    blocks = [ Dict{Tuple{Ts, Ts}, Function}()     for n=1:d]
    for ((n₁,s₁,idx₁), (n₂,s₂,idx₂)) in edge_labels(graph)
      blocks[n₁][((s₁,idx₁),(s₂,idx₂))] = graph[(n₁,s₁,idx₁), (n₂,s₂,idx₂)]
    end

    return new{T,N,d}(states, blocks, graph)
  end
end

function SparseHamiltonian(t::Matrix{T}, v::Array{T,4}, ψ::TTvector{T,N,d}; ϵ=eps(), reduced=false) where {T<:Number,N,d}
  return SparseHamiltonian(t,v,Val(N),Val(d); ϵ=ϵ, reduced=reduced)
end

"""
  `H_matvec_core(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d})`

Core function for the matrix-free application of two-body Hamiltonian operators,
implementing the action on a given TT `x`.
The Hamiltonian is given in terms of one-electron integrals `t_{ij}` and two-electron integrals `v_{ijkl}`.

In particular, the two-electron integrals are assumed to be given in physicists' notation and *reduced format* that is,
`v_{ijkl} = 1/2 ( <ij|kl>  + <ji|kl> - <ij|lk> - <ji|kl> )` for `i < j` and `k < l`,
where `<ij|kl> = ∫∫ ψ†_i(x₁) ψ†_j(x₂) 1/|x₁-x₂| ψ_k(x₁) ψ_l(x₂) dx₁dx₂`.

Returns a tuple containing a sparse representation of the resulting core:
block ranks, block row and column index ranges and views into block data.
"""
function H_matvec_core(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}) where {T<:Number,N,d}
  n = x.k
  row_states = H.states[n]
  col_states = H.states[n+1]
  blocks     = H.blocks[n]

  ##################################################################################################
  ###                       Precompute ranks and block structure ranges                          ###
  ##################################################################################################

  HC = ( similar(x.row_ranks), 
         similar(x.col_ranks), 
         Vector{OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}}(undef, length(blocks)), 
         Vector{OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}}(undef, length(blocks)), 
         Vector{UnsafeSparseCore{T,N,d}}(undef, length(blocks))
        )

  rowsize = length(row_states)
  colsize = length(col_states)
  blockrow_starts = [1 for i=1:rowsize, ql in axes(x, 1)]
  blockrow_ends   = [0 for i=1:rowsize, ql in axes(x, 1)]
  blockcol_starts = [1 for i=1:colsize, ql in axes(x, 3)]
  blockcol_ends   = [0 for i=1:colsize, ql in axes(x, 3)]

  row_ranks = HC[1]
  col_ranks = HC[2]

  if n == 1 # First core is special; stacking horizontally (row rank should be same as `x`)
    for (i, state) in enumerate(row_states)
      blockrow_ends[i,:] = shift_ranks(x.row_ranks, first(state)..., N) 
        # should be filled with zeros and x.row_ranks[0] (usually 1)'s
    end
    row_ranks .= x.row_ranks
  else # n > 1
    starts = [1 for ql in axes(x, 1)]
    for (i, (s,idx)) in enumerate(row_states)
      R = shift_ranks(x.row_ranks, s..., N)
      blockrow_starts[i,:] = starts
      blockrow_ends[  i,:] = starts .+ R .- 1
      starts .+= R
    end
    row_ranks .= blockrow_ends[end,:]
  end

  if n<d
    starts = [1 for ql in axes(x, 3)]
    for (i, (s,idx)) in enumerate(col_states)
      R = shift_ranks(x.col_ranks, s..., N)
      blockcol_starts[i,:] .= starts
      blockcol_ends[  i,:] .= starts .+ R .- 1
      starts .+= R
    end
    col_ranks .= blockcol_ends[end,:]
  else # n == d # Last core: stacking vertically (column rank should be same as `x`)
    for  (i,(s,idx)) in enumerate(col_states)
      blockcol_ends[i,:] .= shift_ranks(x.col_ranks, s..., N)
        # should be filled with zeros and x.col_ranks[N] (usually 1)'s
    end
    col_ranks .= x.col_ranks
  end

  blockrow_ranges = Dict{ Ts, 
                          OffsetArrays.OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}
                        }(
    v => [ blockrow_starts[i,l]:blockrow_ends[i,l] for l in axes(x,1)] 
    for (i, v) in enumerate(row_states)
                        )
  blockcol_ranges = Dict{ Ts, 
                          OffsetArrays.OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}
                        }(
    v => [ blockcol_starts[i,l]:blockcol_ends[i,l] for l in axes(x,3)] 
    for (i, v) in enumerate(col_states) 
                        )

  ##########################################################################################################################################

  for (j, ( ((s₁,idx₁),(s₂,idx₂)), op ) ) in enumerate(blocks)
    OpC, colstate... = op(x, s₁...)
    @boundscheck @assert colstate == s₂
    HC[3][j] = blockrow_ranges[(s₁,idx₁)]
    HC[4][j] = blockcol_ranges[(s₂,idx₂)]
    HC[5][j] = OpC
  end

  return HC
end

function Base.:*(Fᴸ::Frame{T,N,d,M}, H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  Hx = H_matvec_core(H, x)
  @boundscheck begin
    @assert site(Fᴸ) == site(x)
    @assert Fᴸ.col_ranks == Hx[1]
  end
  y = SparseCore{T,N,d}(x.k, Fᴸ.row_ranks, Hx[2])
  for (I,J,v) in zip(Hx[3],Hx[4],Hx[5])
    mul!(y[:,J], Fᴸ[:,I], v, 1, 1)
  end
  return y
end

function Base.:*(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}, Fᴿ::Frame{T,N,d,M}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  Hx = H_matvec_core(H, x)
  @boundscheck begin
    @assert site(Fᴿ) == site(x)+1
    @assert Fᴿ.row_ranks == Hx[2]
  end
  y = SparseCore{T,N,d}(x.k, Hx[1], Fᴿ.col_ranks)
  for (I,J,v) in zip(Hx[3],Hx[4],Hx[5])
    mul!(y[I,:], v, Fᴿ[J,:], 1, 1)
  end
  return y
end


function Base.:*(l::AdjointSparseCore{T,N,d}, H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}) where {T<:Number,N,d}
  Hx = H_matvec_core(H, x)
  @boundscheck begin
    @assert site(l) == site(x)
    @assert parent(l).row_ranks == Hx[1]
  end
  y = Frame{T,N,d}(x.k+1, parent(l).col_ranks, Hx[2])
  for (I,J,v) in zip(Hx[3],Hx[4],Hx[5])
    mul!(y[:,J], l[:,I], v, 1, 1)
  end
  return y
end

function Base.:*(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}, r::AdjointSparseCore{T,N,d}) where {T<:Number,N,d}
  Hx = H_matvec_core(H, x)
  @boundscheck begin
    @assert site(r) == site(x)
    @assert parent(r).col_ranks == Hx[2]
  end
  y = Frame{T,N,d}(x.k, Hx[1], parent(r).row_ranks)
  for (I,J,v) in zip(Hx[3],Hx[4],Hx[5])
    mul!(y[I,:], v, r[J,:], 1, 1)
  end
  return y
end

function Base.:*(H::SparseHamiltonian{T,N,d}, X::TTvector{T,N,d}) where {T<:Number,N,d}
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef, d)

  for k=1:d
    Hxₖ = H_matvec_core(H, core(X,k))
    cores[k] = SparseCore{T,N,d}(k, Hxₖ[1], Hxₖ[2])
    for (I,J,v) in zip(Hxₖ[3],Hxₖ[4],Hxₖ[5])
      copyto!(cores[k][I,J], v)
    end
  end

  return cores2tensor(cores)
end

function RayleighQuotient(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}; orthogonalize::Bool=false) where {T<:Number,N,d}

# to = TimerOutput()
# @timeit to "Orthogonalize" begin
  if (orthogonalize)
    leftOrthogonalize!(x)
  end
# end

# @timeit to "Frame 1" begin
  p = IdFrame(Val(N), Val(d), 1)
# end
# @timeit to "Contractions" begin
  for k=1:d
    p = adjoint(core(x,k)) * (p*H*core(x,k))
  end
# end
# display(to)

  return p[N][1]
end

# using TimerOutputs
function xᵀHy(x::TTvector{T,N,d}, H::SparseHamiltonian{T,N,d}, y::TTvector{T,N,d}; orthogonalize::Bool=false) where {T<:Number,N,d}

# to = TimerOutput()
# @timeit to "Orthogonalize" begin
  if (orthogonalize)
    leftOrthogonalize!(x)
    leftOrthogonalize!(y)
  end
# end

# @timeit to "Frame 1" begin
  p = IdFrame(Val(N), Val(d), 1)
# end
# @timeit to "Contractions" begin
  for k=1:d
    p = adjoint(core(x,k)) * (p*H*core(y,k))
  end
# end
# display(to)

  return p[N][1]
end


end