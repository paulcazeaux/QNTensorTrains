module Hamiltonian
import ..QNTensorTrains
using ..QNTensorTrains: Frame, IdFrame, SparseCore, AdjointCore, TTvector
using ..QNTensorTrains: site, core, unoccupied, occupied, row_ranks, col_ranks, row_rank, col_rank
using ..QNTensorTrains: occupation_qn, shift_qn, cores2tensor
using LinearAlgebra, OffsetArrays

export SparseHamiltonian, H_matvec_core, RayleighQuotient, xᵀHy

const Ts = Tuple{NTuple{3,Int64}, NTuple{2,Int64}}
const OV{T} = OffsetVector{T,Vector{T}} where T


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

using Graphs, MetaGraphsNext

struct SparseHamiltonian{T<:Number,N,d}
  states::Vector{OV{Vector{Ts}}}
  state_qns::Vector{OV{Vector{Int}}}
  coo_blocks::Vector{
    @NamedTuple{unoccupied::OV{
             @NamedTuple{unoccupied::OV{Tuple{Vector{Int},Vector{Int},Vector{T}}},
                           occupied::OV{Tuple{Vector{Int},Vector{Int},Vector{T}}}}
                              }, 
                  occupied::OV{
             @NamedTuple{unoccupied::OV{Tuple{Vector{Int},Vector{Int},Vector{T}}},
                           occupied::OV{Tuple{Vector{Int},Vector{Int},Vector{T}}}}
                              } 
               }}
  csr_blocks::Vector{
    @NamedTuple{unoccupied::OV{
             @NamedTuple{unoccupied::OV{Tuple{OV{Int},Vector{Int},Vector{T}}},
                           occupied::OV{Tuple{OV{Int},Vector{Int},Vector{T}}}}
                              }, 
                  occupied::OV{
             @NamedTuple{unoccupied::OV{Tuple{OV{Int},Vector{Int},Vector{T}}},
                           occupied::OV{Tuple{OV{Int},Vector{Int},Vector{T}}}}
                              }
               }}
  csc_blocks::Vector{
    @NamedTuple{unoccupied::OV{
             @NamedTuple{unoccupied::OV{Tuple{OV{Int},Vector{Int},Vector{T}}},
                           occupied::OV{Tuple{OV{Int},Vector{Int},Vector{T}}}}
                              }, 
                  occupied::OV{
             @NamedTuple{unoccupied::OV{Tuple{OV{Int},Vector{Int},Vector{T}}},
                           occupied::OV{Tuple{OV{Int},Vector{Int},Vector{T}}}}
                              }
               }}
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

    function occupation(l,r)
      if l==r
        return :unoccupied
      elseif l+1==r
        return :occupied
      else
        throw(BoundsError())
      end
    end
    # One body states
    function vertex(κ, i,j)
      if κ ≤ min(i,j)
        state = ( 0,0,1); index =       (0,0)
      elseif i < κ ≤ j
        state = ( 1,1,0); index = (κ ≤ hd ? (i,0) : (j,0))
      elseif j < κ ≤ i
        state = (-1,0,1); index = (κ ≤ hd ? (j,0) : (i,0))
      else # max(i,j) < κ
        state = ( 0,1,0); index =       (0,0)
      end
      return state, index
    end

    function weighted_edge(i,j)
      return sort([hd, i,j])[2]
    end

    function set_edges(κ, i,j, w=T(1))
      @assert 1 ≤ κ ≤ d
      s₋,idx₋ = vertex(κ,   i,j)
      s₊,idx₊ = vertex(κ+1, i,j)

      op = ( κ == i == j ? :AdagA :
             κ == j      ? :A     :
             κ == i      ? :Adag  :
             i < κ < j || j < κ < i ? 
                           :S : :Id )

      qn₋ = shift_qn(occupation_qn(N,d,κ  ), s₋..., N)
      qn₊ = shift_qn(occupation_qn(N,d,κ+1), s₊..., N)
      continue_path = false
      for n₋ in qn₋, n₊ in ( op == :A ? (n₋:n₋) : op ∈ (:Adag, :AdagA) ? (n₋+1:n₋+1) : (n₋:n₋+1) ) ∩ qn₊
        l  = n₋-s₋[1]
        r  = n₊-s₊[1]
        continue_path = add_vertex!(graph, (κ,n₋,s₋,idx₋)) | add_vertex!(graph, (κ+1,n₊,s₊,idx₊)) || continue_path
        if op==:S && occupation(n₋,n₊)==:occupied # Jordan-Wigner factor
          add_edge!(graph, (κ,n₋,s₋,idx₋), (κ+1,n₊,s₊,idx₊), (l,occupation(l,r),-w))
        else
          add_edge!(graph, (κ,n₋,s₋,idx₋), (κ+1,n₊,s₊,idx₊), (l,occupation(l,r),w))
        end
      end
      return continue_path
    end

    # Two body states
    function vertex(κ, i,j,k,l)
      @assert i<j && k<l
      if κ ≤ min(i,k)
        state = ( 0,0,2);    index =        (0,0)
      elseif i < κ ≤ min(j,k)
        state = ( 1,1,1);    index =        (i,0)
      elseif k < κ ≤ min(i,l)
        state = (-1,0,2);    index =        (k,0)
      elseif max(i,k) < κ ≤ min(j,l)
        state = ( 0,1,1);    index = (κ ≤ hd ?  (i,k) : (j,l))
      elseif j < κ ≤ k
        state = ( 2,2,0);    index = (κ ≤ hd ?  (i,j) : (k,l))
      elseif l < κ ≤ i
        state = (-2,0,2);    index = (κ ≤ hd ?  (k,l) : (i,j))
      elseif max(j,k) < κ ≤ l
        state = ( 1,2,0);    index =        (l,0)
      elseif max(i,l) < κ ≤ j
        state = (-1,1,1);    index =        (j,0)
      else # max(j,l) < κ
        state = ( 0,2,0);    index =        (0,0)
      end 
      return state, index
    end

    function weighted_edge(i,j,k,l)
      @assert i<j && k<l
      return sort([hd, i,j,k,l])[3]
    end 

    function set_edges(κ, i,j,k,l,w=T(1))
      @assert i<j && k<l && 1 ≤ κ ≤ d

      s₋,idx₋ = vertex(κ,   i,j,k,l)
      s₊,idx₊ = vertex(κ+1, i,j,k,l)

      op = ( κ ∈ (i,j) && κ ∈ (k,l) ? :AdagA :
             κ ∈ (k,l)              ? :A     :
             κ ∈ (i,j)              ? :Adag  :
             i < κ < min(j,k) || k < κ < min(i,l) || max(j,k) < κ < l || max(i,l) < κ < j ?
                                      :S : :Id )

      qn₋ = shift_qn(occupation_qn(N,d,κ  ), s₋..., N)
      qn₊ = shift_qn(occupation_qn(N,d,κ+1), s₊..., N)
      continue_path = false
      for n₋ in qn₋, n₊ in ( op == :A ? (n₋:n₋) : op ∈ (:Adag, :AdagA) ? (n₋+1:n₋+1) : (n₋:n₋+1) ) ∩ qn₊
        l  = n₋-s₋[1]
        r  = n₊-s₊[1]
        continue_path = add_vertex!(graph, (κ,n₋,s₋,idx₋)) | add_vertex!(graph, (κ+1,n₊,s₊,idx₊)) || continue_path
        if op==:S && occupation(n₋,n₊)==:occupied # Jordan-Wigner factor
          add_edge!(graph, (κ,n₋,s₋,idx₋), (κ+1,n₊,s₊,idx₊), (l,occupation(l,r),-w))
        else
          add_edge!(graph, (κ,n₋,s₋,idx₋), (κ+1,n₊,s₊,idx₊), (l,occupation(l,r), w))
        end
      end

      return continue_path
    end

    graph = MetaGraph(
      DiGraph();                       # Initialize empty graph
      label_type=Tuple{Int, Int, NTuple{3, Int}, NTuple{2,Int} },          
                                       # site, quantum number, state and index
      vertex_data_type=Nothing,        # State details 
      edge_data_type=Tuple{Int,Symbol,T}, # relevant block indices, coefficient and single-site operator
      weight_function=ed -> ed[3],
      default_weight=0.,
      graph_data="Hamiltonian action graph",                  # tag for the whole graph
    )

    for i=1:d, j=1:d
      if abs(t[i,j]) > ϵ
        κₘ = weighted_edge(i,j)
        set_edges(κₘ, i,j, t[i,j])
        for κ=κₘ+1:d
          continue_path = set_edges(κ, i,j) 
          continue_path || break
        end
        for κ = κₘ-1:-1:1
          continue_path = set_edges(κ, i,j) 
          continue_path || break
        end
      end
    end

    for i=1:d-1, j=i+1:d, k=1:d-1, l=k+1:d
      if abs(v[i,j,k,l]) > ϵ
        κₘ = weighted_edge(i,j,k,l)
        set_edges(κₘ, i,j,k,l, v[i,j,k,l])
        for κ=κₘ+1:d
          continue_path = set_edges(κ, i,j,k,l) 
          continue_path || break
        end
        for κ = κₘ-1:-1:1
          continue_path = set_edges(κ, i,j,k,l) 
          continue_path || break
        end
      end
    end

    nstates = [ [ 0 for n in occupation_qn(N,d,k)]                               for k=1:d+1]
    for (k,n,s,idx) in labels(graph)
      nstates[k][n] += 1
    end

    states = [ [ Vector{Ts}(undef, nstates[k][n]) for n in occupation_qn(N,d,k)] for k=1:d+1]
    i      = [ [ 0 for n in occupation_qn(N,d,k)                               ] for k=1:d+1]
    for (k,n,s,idx) in labels(graph)
      states[k][n][
        i[k][n] += 1
                  ] = (s, idx)
    end

    for k=1:d+1, n in occupation_qn(N,d,k)
      sort!(states[k][n]) # Let's try mostly the default lexicographic order.
    end
    state_to_index = [ [ Dict(s=>i for (i,s) in enumerate(states[k][n])) 
                         for n in occupation_qn(N,d,k) ] for k=1:d+1 ]
    state_qns = [ [ [n-flux for ((flux,),) in states[k][n]] 
                         for n in occupation_qn(N,d,k) ] for k=1:d+1 ]

    qn = [ (
      unoccupied = occupation_qn(N,d,k)∩ occupation_qn(N,d,k+1),
      occupied   = occupation_qn(N,d,k)∩(occupation_qn(N,d,k+1).-1)
           ) for k=1:d ]

    wrap(f) = [(
      unoccupied = OffsetVector( 
        [( unoccupied = OffsetVector([f(k,n,:unoccupied,m,:unoccupied) for m in qn[k].unoccupied∩(n-2:n+2)], qn[k].unoccupied∩(n-2:n+2)),
             occupied = OffsetVector([f(k,n,:unoccupied,m,:occupied)   for m in qn[k].occupied  ∩(n-2:n+2)], qn[k].occupied  ∩(n-2:n+2))
         ) for n in qn[k].unoccupied ], qn[k].unoccupied ),
      occupied   = OffsetVector( 
        [( unoccupied = OffsetVector([f(k,n,:occupied,  m,:unoccupied) for m in qn[k].unoccupied∩(n-2:n+2)], qn[k].unoccupied∩(n-2:n+2)),
             occupied = OffsetVector([f(k,n,:occupied,  m,:occupied)   for m in qn[k].occupied  ∩(n-2:n+2)], qn[k].occupied  ∩(n-2:n+2))
         ) for n in qn[k].occupied ],   qn[k].occupied   )
                         ) for k=1:d ]

    nblocks = wrap( (idx...) -> 0)
    for ( (k₁,n₁,s₁,idx₁), (k₂,n₂,s₂,idx₂) ) in edge_labels(graph)
      m,occ_m,α = graph[(k₁,n₁,s₁,idx₁), (k₂,n₂,s₂,idx₂)]
      nblocks[k₁][occupation(n₁,n₂)][n₁][occ_m][m] += 1
    end

    # Here we obtain the blocks of the TT-cores indexed by n and occupation, 
    # for any given block indexed by l and occupation from the original TT-core,
    # stored in sparse COO format: row/column state indices and corresponding multiplier
    coo_blocks = wrap( (k,n,occ_n,m,occ_m) -> (Vector{Int}(undef, nblocks[k][occ_n][n][occ_m][m]), 
                                               Vector{Int}(undef, nblocks[k][occ_n][n][occ_m][m]), 
                                               Vector{T}(  undef, nblocks[k][occ_n][n][occ_m][m])) )
    j = wrap( (idx...) -> 0)
    for ((k₁,n₁,s₁,idx₁), (k₂,n₂,s₂,idx₂)) in edge_labels(graph)
      i₁ = state_to_index[k₁][n₁][(s₁,idx₁)]
      i₂ = state_to_index[k₂][n₂][(s₂,idx₂)]
      m,occ_m,α = graph[(k₁,n₁,s₁,idx₁), (k₂,n₂,s₂,idx₂)]

      j[k₁][occupation(n₁,n₂)][n₁][occ_m][m] += 1
      J = j[k₁][occupation(n₁,n₂)][n₁][occ_m][m]
      block = coo_blocks[k₁][occupation(n₁,n₂)][n₁][occ_m][m]

      block[1][J], block[2][J], block[3][J] = i₁, i₂, α
    end

    # Post-processing to compute alternative CSR and CSC formats of the same sparse objects
    csc_blocks = wrap( (idx...) -> (OffsetVector(Int[], 1:0),Int[],T[]) )
    csr_blocks = wrap( (idx...) -> (OffsetVector(Int[], 1:0),Int[],T[]) )

    for k=1:d, occ_n in (:unoccupied, :occupied), n in axes(coo_blocks[k][occ_n],1), occ_m in (:unoccupied, :occupied), m in axes(coo_blocks[k][occ_n][n][occ_m],1)
      nnz   = nblocks[k][occ_n][n][occ_m][m]
      if nnz>0
        block = coo_blocks[k][occ_n][n][occ_m][m]
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

        csr_blocks[k][occ_n][n][occ_m][m] = (row_starts, block[2], block[3])

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

        csc_blocks[k][occ_n][n][occ_m][m] = ( col_starts, block[2][p_col], block[3][p_col] )
      end
    end

    return new{T,N,d}(states, state_qns, coo_blocks, csr_blocks, csc_blocks, graph)
  end
end

function SparseHamiltonian(t::Matrix{T}, v::Array{T,4}, ψ::TTvector{T,N,d}; ϵ=eps(), reduced=false) where {T<:Number,N,d}
  return SparseHamiltonian(t,v,Val(N),Val(d); ϵ=ϵ, reduced=reduced)
end

function QNTensorTrains.row_ranks(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d,M}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = x.k
  if k==1
    return row_ranks(x)
  else
    state_qns = H.state_qns[k]
    ranks = similar(row_ranks(x))
    for n in axes(ranks,1)
      ranks[n] = isempty(state_qns[n]) ? 0 : sum(row_rank(x,l) for l in state_qns[n])
    end
    return ranks
  end
end

function row_ranges(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d,M}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = x.k
  state_qns = H.state_qns[k]
  row_ranges = similar(row_ranks(x),Vector{UnitRange{Int}})

  if k==1
    row_ranges[0] = [(1:row_ranks(x)[0]) for i=1:length(state_qns[0])]
  else
    for n in axes(row_ranges,1)
      ends = cumsum(row_ranks(x)[l] for l in state_qns[n])
      row_ranges[n] = [(i==1 ? 1 : ends[i-1]+1):ends[i] for i=1:length(state_qns[n])]
    end
  end
  return row_ranges
end

function QNTensorTrains.col_ranks(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = x.k
  if k==d
    return col_ranks(x)
  else
    state_qns = H.state_qns[k+1]
    ranks = similar(col_ranks(x))
    for n in axes(ranks,1)
      ranks[n] = isempty(state_qns[n]) ? 0 : sum(col_rank(x,r) for r in state_qns[n])
    end
    return ranks
  end
end

function col_ranges(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = x.k
  state_qns = H.state_qns[k+1]
  col_ranges = similar(col_ranks(x),Vector{UnitRange{Int}})

  if k==d
    col_ranges[N] = [(1:col_ranks(x)[N]) for j=1:length(state_qns[N])]
  else
    for n in axes(col_ranges,1)
      ends = cumsum(col_ranks(x)[r] for r in state_qns[n])
      col_ranges[n] = [(i==1 ? 1 : ends[i-1]+1):ends[i] for i=1:length(state_qns[n])]
    end
  end
  return col_ranges
end

function Base.:*(Fᴸ::Frame{T,N,d}, H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(Fᴸ) == site(x)
    @assert Fᴸ.col_ranks == row_ranks(H,x)
  end
  I,J = row_ranges(H,x), col_ranges(H,x)

  y = SparseCore{T,N,d}(x.k, Fᴸ.row_ranks, col_ranks(H,x))
  COO = H.coo_blocks[x.k]
  @sync begin
    for n in axes(unoccupied(y),1)
      @Threads.spawn let Y = unoccupied(y,n), F = Fᴸ[n]
        for m in axes(unoccupied(x),1)∩(n-2:n+2)
          X = unoccupied(x,m)
          # if !isempty(COO.unoccupied[n].unoccupied[m][1])
          #   nnz = length(COO.unoccupied[n].unoccupied[m][1])
          #   row_ratio = -nnz/-(extrema(COO.unoccupied[n].unoccupied[m][1])...)
          #   col_ratio = -nnz/-(extrema(COO.unoccupied[n].unoccupied[m][2])...)
          #   @show site(x),n,m,nnz,row_ratio,col_ratio
          # end
          for (i,j,α) in zip(COO.unoccupied[n].unoccupied[m]...)
            mul!(view(Y,:,J[n][j]), view(F,:,I[n][i]), X, α, 1)
          end
        end
        for m in axes(occupied(x),1)∩(n-2:n+2)
          # if !isempty(COO.unoccupied[n].occupied[m][1])
          #   nnz = length(COO.unoccupied[n].occupied[m][1])
          #   row_ratio = -nnz/-(extrema(COO.unoccupied[n].occupied[m][1])...)
          #   col_ratio = -nnz/-(extrema(COO.unoccupied[n].occupied[m][2])...)
          #   @show site(x),n,m,nnz,row_ratio,col_ratio
          # end
          X = occupied(x,m)
          for (i,j,α) in zip(COO.unoccupied[n].occupied[m]...)
            mul!(view(Y,:,J[n][j]), view(F,:,I[n][i]), X, α, 1)
          end
        end
      end
    end
    for n in axes(occupied(y),1)
      @Threads.spawn let Y = occupied(y,n), F = Fᴸ[n]
        for m in axes(unoccupied(x),1)∩(n-2:n+2)
          X = unoccupied(x,m)
          # if !isempty(COO.occupied[n].unoccupied[m][1])
          #   nnz = length(COO.occupied[n].unoccupied[m][1])
          #   row_ratio = -nnz/-(extrema(COO.occupied[n].unoccupied[m][1])...)
          #   col_ratio = -nnz/-(extrema(COO.occupied[n].unoccupied[m][2])...)
          #   @show site(x),n,m,nnz,row_ratio,col_ratio
          # end
          for (i,j,α) in zip(COO.occupied[n].unoccupied[m]...)
            mul!(view(Y,:,J[n+1][j]), view(F,:,I[n][i]), X, α, 1)
          end
        end
        for m in axes(occupied(x),1)∩(n-2:n+2)
          X = occupied(x,m)
          # if !isempty(COO.occupied[n].occupied[m][1])
          #   nnz = length(COO.occupied[n].occupied[m][1])
          #   row_ratio = -nnz/-(extrema(COO.occupied[n].occupied[m][1])...)
          #   col_ratio = -nnz/-(extrema(COO.occupied[n].occupied[m][2])...)
          #   @show site(x),n,m,nnz,row_ratio,col_ratio
          # end
          for (i,j,α) in zip(COO.occupied[n].occupied[m]...)
            mul!(view(Y,:,J[n+1][j]), view(F,:,I[n][i]), X, α, 1)
          end
        end
      end
    end
  end

  return y
end

function Base.:*(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}, Fᴿ::Frame{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(Fᴿ) == site(x)+1
    @assert Fᴿ.row_ranks == col_ranks(H,x)
  end
  I,J = row_ranges(H,x), col_ranges(H,x)

  y = SparseCore{T,N,d}(x.k, row_ranks(H,x), Fᴿ.col_ranks)
  COO = H.coo_blocks[x.k]
  @sync begin
    for n in axes(unoccupied(y),1)
      @Threads.spawn let Y = unoccupied(y,n), F = Fᴿ[n]
        for m in axes(unoccupied(x),1)∩(n-2:n+2)
          X = unoccupied(x,m)
          for (i,j,α) in zip(COO.unoccupied[n].unoccupied[m]...)
            mul!(view(Y,I[n][i],:), X, view(F,J[n][j],:), α, 1)
          end
        end
        for m in axes(occupied(x),1)∩(n-2:n+2)
          X = occupied(x,m)
          for (i,j,α) in zip(COO.unoccupied[n].occupied[m]...)
            mul!(view(Y,I[n][i],:), X, view(F,J[n][j],:), α, 1)
          end
        end
      end
    end
    for n in axes(occupied(y),1)
      @Threads.spawn let Y = occupied(y,n), F = Fᴿ[n+1]
        for m in axes(unoccupied(x),1)∩(n-2:n+2)
          X = unoccupied(x,m)
          for (i,j,α) in zip(COO.occupied[n].unoccupied[m]...)
            mul!(view(Y,I[n][i],:), X, view(F,J[n+1][j],:), α, 1)
          end
        end
        for m in axes(occupied(x),1)∩(n-2:n+2)
          X = occupied(x,m)
          for (i,j,α) in zip(COO.occupied[n].occupied[m]...)
            mul!(view(Y,I[n][i],:), X, view(F,J[n+1][j],:), α, 1)
          end
        end
      end
    end
  end
  return y
end

function Base.:*(l::AdjointCore{T,N,d}, H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(l) == site(x)
    @assert col_ranks(l) == row_ranks(H,x)
  end
  I,J = row_ranges(H,x), col_ranges(H,x)

  y = Frame{T,N,d}(x.k+1, row_ranks(l), col_ranks(H,x))
  COO = H.coo_blocks[x.k]
  @sync for n in axes(unoccupied(parent(l)),1)
    @Threads.spawn let Y = y[n], L = unoccupied(l,n)
      for m in axes(unoccupied(x),1)∩(n-2:n+2)
        X = unoccupied(x,m)
        for (i,j,α) in zip(COO.unoccupied[n].unoccupied[m]...)
          mul!(view(Y,:,J[n][j]), view(L,:,I[n][i]), X, α, 1)
        end
      end
      for m in axes(occupied(x),1)∩(n-2:n+2)
        X = occupied(x,m)
        for (i,j,α) in zip(COO.unoccupied[n].occupied[m]...)
          mul!(view(Y,:,J[n][j]), view(L,:,I[n][i]), X, α, 1)
        end
      end
    end
  end
  @sync for n in axes(occupied(parent(l)),1)
    @Threads.spawn let Y = y[n+1], L = occupied(l,n+1)
      for m in axes(unoccupied(x),1)∩(n-2:n+2)
        X = unoccupied(x,m)
        for (i,j,α) in zip(COO.occupied[n].unoccupied[m]...)
          mul!(view(Y,:,J[n+1][j]), view(L,:,I[n][i]), X, α, 1)
        end
      end
      for m in axes(occupied(x),1)∩(n-2:n+2)
        X = occupied(x,m)
        for (i,j,α) in zip(COO.occupied[n].occupied[m]...)
          mul!(view(Y,:,J[n+1][j]), view(L,:,I[n][i]), X, α, 1)
        end
      end
    end
  end
  return y
end

function Base.:*(H::SparseHamiltonian{T,N,d}, x::SparseCore{T,N,d}, r::AdjointCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(r) == site(x)
    @assert row_ranks(r) == col_ranks(H,x)
  end
  I,J = row_ranges(H,x), col_ranges(H,x)

  y = Frame{T,N,d}(x.k, row_ranks(H,x), col_ranks(r))
  COO = H.coo_blocks[x.k]
  @sync for n in axes(unoccupied(parent(r)),1)
    @Threads.spawn let Y = y[n], R = unoccupied(r,n)
      for m in axes(unoccupied(x),1)∩(n-2:n+2)
        X = unoccupied(x,m)
        for (i,j,α) in zip(COO.unoccupied[n].unoccupied[m]...)
          mul!(view(Y,I[n][i],:), X, view(R,J[n][j],:), α, 1)
        end
      end
      for m in axes(occupied(x),1)∩(n-2:n+2)
        X = occupied(x,m)
        for (i,j,α) in zip(COO.unoccupied[n].occupied[m]...)
          mul!(view(Y,I[n][i],:), X, view(R,J[n][j],:), α, 1)
        end
      end
    end
  end
  @sync for n in axes(occupied(parent(r)),1)
    @Threads.spawn let Y = y[n], R = occupied(r,n+1)
      for m in axes(unoccupied(x),1)∩(n-2:n+2)
        X = unoccupied(x,m)
        for (i,j,α) in zip(COO.occupied[n].unoccupied[m]...)
          mul!(view(Y,I[n][i],:), X, view(R,J[n+1][j],:), α, 1)
        end
      end
      for m in axes(occupied(x),1)∩(n-2:n+2)
        X = occupied(x,m)
        for (i,j,α) in zip(COO.occupied[n].occupied[m]...)
          mul!(view(Y,I[n][i],:), X, view(R,J[n+1][j],:), α, 1)
        end
      end
    end
  end
  return y
end

function Base.:*(H::SparseHamiltonian{T,N,d}, X::TTvector{T,N,d}) where {T<:Number,N,d}
  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef, d)
  for k=1:d
    x = core(X,k)
    cores[k] = SparseCore{T,N,d}(k, row_ranks(H,x), col_ranks(H,x))
    I,J = row_ranges(H,x), col_ranges(H,x)

    COO = H.coo_blocks[k]
    @sync begin
      for n in axes(unoccupied(cores[k]),1)
        @Threads.spawn let Y = unoccupied(cores[k],n)
          for m in axes(unoccupied(x),1)∩(n-2:n+2)
            for (i,j,α) in zip(COO.unoccupied[n].unoccupied[m]...)
              axpy!(α, unoccupied(x,m), view(Y, I[n][i], J[n][j]))
            end
          end
          for m in axes(occupied(x),1)∩(n-2:n+2)
            for (i,j,α) in zip(COO.unoccupied[n].occupied[m]...)
              axpy!(α, occupied(x,m), view(Y, I[n][i], J[n][j]))
            end
          end
        end
      end
      for n in axes(occupied(cores[k]),1)
        @Threads.spawn let Y = occupied(cores[k],n)
          for m in axes(unoccupied(x),1)∩(n-2:n+2)
            for (i,j,α) in zip(COO.occupied[n].unoccupied[m]...)
              axpy!(α, unoccupied(x,m), view(Y, I[n][i], J[n+1][j]))
            end
          end
          for m in axes(occupied(x),1)∩(n-2:n+2)
            for (i,j,α) in zip(COO.occupied[n].occupied[m]...)
              axpy!(α, occupied(x,m), view(Y, I[n][i], J[n+1][j]))
            end
          end
        end
      end
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
  p = IdFrame(Val(d), Val(N), 1)
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
  p = IdFrame(Val(d), Val(N), 1)
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