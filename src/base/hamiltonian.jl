module Hamiltonian
using ..QNTensorTrains: SparseCore, UnsafeSparseCore, TTvector, core, cores2tensor, Id_view, S_view, Adag_view, A_view, AdagA_view, factor, data, isnonzero
using LinearAlgebra, OffsetArrays

export sparse_H_matvec, H_matvec, RayleighQuotient

function reduce(v::Array{T,4}) where {T<:Number}
  d = size(v,1)
  @boundscheck @assert size(v) == (d,d,d,d)

  v_reduced = zeros(T,size(v))
  for i=1:d, k=1:d, j=i+1:d, l=k+1:d
    v_reduced[i,j,k,l] = v[i,j,k,l]+v[j,i,l,k]-v[j,i,k,l]-v[i,j,l,k]
  end

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

"""
  `sparse_H_matvec(core::SparseCore{T,N,d}, t::Matrix{T}, v::Array{T,4})`

Core function for the matrix-free application of two-body Hamiltonian operators,
implementing the action on a given TT `core`.
The Hamiltonian is given in terms of one-electron integrals `t_{ij}` and two-electron integrals `v_{ijkl}`.

In particular, the two-electron integrals are assumed to be given in physicists' notation and *reduced format* that is,
`v_{ijkl} = 1/2 ( <ij|kl>  + <ji|kl> - <ij|lk> - <ji|kl> )` for `i < j` and `k < l`,
where `<ij|kl> = ∫∫ ψ†_i(x₁) ψ†_j(x₂) 1/|x₁-x₂| ψ_k(x₁) ψ_l(x₂) dx₁dx₂`.

Returns a tuple containing a sparse representation of the resulting core:
block ranks, block row and column index ranges and views into block data.
"""
function sparse_H_matvec(core::SparseCore{T,N,d}, t::Matrix{T}, v::Array{T,4}) where {T<:Number,N,d}
  n = core.k
  # @boundscheck @assert 1 ≤ n ≤ d

  hd = d÷2
  
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


  left_states = (
      # One body states
      ( 0,0,1), #  1 #       n < i<j
      ( 1,1,0), #  2 # i   ≤ n <   j
      (-1,0,1), #  3 #   j ≤ n < i
      ( 0,1,0), #  4 # i<j ≤ n
      # Two body states
      ( 0,0,2), #  5 #           n < i<j,k<l
      ( 1,1,1), #  6 # i       ≤ n <   j,k<l
      (-1,0,2), #  7 #     k   ≤ n < i<j,  l
      ( 0,1,1), #  8 # i,  k   ≤ n <   j,  l
      ( 2,2,0), #  9 # i<j     ≤ n <     k<l
      (-2,0,2), # 10 #     k<l ≤ n < i<j
      ( 1,2,0), # 11 # i<j,k   ≤ n <       l
      (-1,1,1), # 12 # i,  k<l ≤ n <   j
      ( 0,2,0)  # 13 # i<j,k<l ≤ n
    )

  left_iterators(n) = ( #One-body
      (   1         for _=1:1     ),
      (    i   for i=1:n          ),
      (     j            for j=1:n),
      (      1      for _=1:1     ),
      # Two body
      ((   1   )                    for _=1:1                   ),
      ((i      ) for i=1:n                                      ),
      ((    k  )                         for k=1:n              ),
      ((i,  k  ) for i=1:n               for k=1:n              ),
      ((i,j    ) for i=1:n-1 for j=i+1:n                        ),
      ((    k,l)                         for k=1:n-1 for l=k+1:n),
      ((      l)                                     for l=n+1:d),
      ((  j    )             for j=n+1:d                        ),
      ((   1   )                    for _=1:1                   )
  )

  left_sizes = OffsetVector([ 
      ( #One-body
        1, n, n, 1,
        # Two-body
        1, n, n, n^2, (n*(n-1))÷2, (n*(n-1))÷2, d-n, d-n, 1
      ) for n=0:hd ], 0:hd)

  left_size(n) = 2d+4+n+2n^2

  left_indices(n) = ( #One-body
      (~ )::Int->1                                                                          , 
      (i )::Int->i                                                        +    1            , 
      ( j)::Int-> j                                                       +    1+ n         , 
      (~ )::Int->1                                                        +    1+2n         ,
      # Two body
      (   ~   )::Int          ->1                                         +    2+ 2n        , 
      (i      )::Int          ->i                                         +    3+ 2n        , 
      (    k  )::Int          ->                       k                  +    3+ 3n        , 
      (i,  k  )::NTuple{2,Int}->i +                 n*(k-1)               +    3+ 4n        , 
      (i,j    )::NTuple{2,Int}->i + (j-1)*(j-2)÷2                         +    3+ 4n+ n^2   ,
      (    k,l)::NTuple{2,Int}->                       k  + (l-1)*(l-2)÷2 +    3+(7n+3n^2)÷2,
      (      l)::Int          ->                              l-n         +    3+ 3n+2n^2   ,
      (  j    )::Int          ->     j-n                                  +  d+3+ 2n+2n^2   ,
      (   ~   )::Int          ->1                                         + 2d+3+  n+2n^2 
  )

  # Flowchart for right-half cores: 
  # One-Body states
  #                 /-Adag- 2▼ <-S- 2▼ <--A---\
  #                /                         \
  # 1 <---Id--- 1 <----------AdagA----------- 4 <--Id--- 4
  #                \                         /
  #                 \--A--- 3▼ <-S- 3▼ <-Adag-/
  #
  #
  # Two-body states
  #
  #                                    /-Adag-10⏬ <--Id--10⏬ <-▼--A--▲-\
  #                                   /                          ▼     ▲  \
  #                                  /                           ▼     ▲   \  
  #               /-Adag- 7▼ <--S- 7▼ <--------------------------▼AdagA▲ - 12▲ <-S-- 12▲ <---A--\
  #              /                   \                           ▼     ▲   /                     \
  #             /                     \--A---\              /----▼Adag-▲--/                       \
  #            /                              \            /     ▼     ▲                           \
  # 5 <-Id--5 <-------------AdagA------------- 8⏬ <--Id-- 8⏬ <-▼AdagA▲--------------------------- 13 <-Id-- 13
  #            \                              /            \     ▼     ▲                           /
  #             \                     /-Adag-/              \----▼--A--▲--\                       /
  #              \                   /                           ▼         \                     /
  #               \--A--- 6▼ <--S- 6▼ <--------------------------▼AdagA▲ - 11▲ <-S-- 11▲ <-Adag-/
  #                                  \                           ▼     ▲   /
  #                                   \                          ▼     ▲  /
  #                                    \--A--- 9⏬ <--Id-- 9⏬ <-▼Adag-▲-/
  #


  right_states = (
    # One body states
    ( 0,1,0), #  1 # i,j < n
    (-1,0,1), #  2 #   j < n ≤ i
    ( 1,1,0), #  3 # i   < n ≤   j
    ( 0,0,1), #  4 #       n ≤ i,j
    # Two body states
    ( 0,2,0), #  5 # i<j,k<l < n
    (-1,1,1), #  6 # i,  k<l < n ≤   j
    ( 1,2,0), #  7 # i<j,k   < n ≤       l
    ( 0,1,1), #  8 # i,  k   < n ≤   j,  l
    (-2,0,2), #  9 #     k<l < n ≤ i<j
    ( 2,2,0), # 10 # i<j     < n ≤     k<l
    (-1,0,2), # 11 #     k   < n ≤ i<j,  l
    ( 1,1,1), # 12 # i       < n ≤   j,k<l
    ( 0,0,2)  # 13 #           n ≤ i<j,k<l
  )

  right_iterators(n) = ( #One-body
      (   1           for _=1:1     ),
      (    i   for i=n+1:d          ),
      (     j            for j=n+1:d),
      (      1        for _=1:1     ),
      # Two body
      ((   1   )                        for _=1:1                   ),
      ((  j    )               for j=n+1:d                          ),
      ((      l)                                         for l=n+1:d),
      ((  j,  l)               for j=n+1:d               for l=n+1:d),
      ((i,j    ) for i=n+1:d-1 for j=i+1:d                          ),
      ((    k,l)                           for k=n+1:d-1 for l=k+1:d),
      ((    k  )                           for k=1:n                ),
      ((i      ) for i=1:n                                          ),
      ((   1   )                      for _=1:1                     )
  )

  right_sizes = OffsetVector([
    ( #One-body
      1, nc, nc, 1,
      # Two-body
      1, nc, nc, nc^2, (nc*(nc-1))÷2, (nc*(nc-1))÷2, d-nc, d-nc, 1
    ) 
    for nc in (d-hd):-1:0], hd:d)

  function right_indices(n)
    nc = d-n
    return ( #One-body
       (~ )::Int->1                                                                                   , 
       (i )::Int->i-n                                                             +    1              , 
       ( j)::Int->   j-n                                                          +    1+  nc         , 
       (~ )::Int->1                                                               +    1+ 2nc         ,
       # Two body 
       (  ~    )::Int          ->1                                                +    2+ 2nc         , 
       ( j     )::Int          ->        j-n                                      +    3+ 2nc         , 
       (      l)::Int          ->                                l-n              +    3+ 3nc         , 
       (  j,  l)::NTuple{2,Int}->        j-n +                  nc*(l-n-1)        +    3+ 4nc         , 
       (i,j    )::NTuple{2,Int}->i-n + (j-n-1)*(j-n-2)÷2                          +    3+ 4nc+ nc^2   , 
       (    k,l)::NTuple{2,Int}->                        k-n +  (l-n-1)*(l-n-2)÷2 +    3+(7nc+3nc^2)÷2, 
       (    k  )::Int          ->                        k                        +    3+ 3nc+2nc^2   , 
       (i      )::Int          ->i                                                +  d+3+ 2nc+2nc^2   ,  
       (   ~   )::Int          ->1                                                + 2d+3+  nc+2nc^2
      )
  end

  function right_size(n) 
    nc = d-n
    return 2d+4+nc+2nc^2
  end

  ##################################################################################################
  ###                       Precompute ranks and block structure ranges                          ###
  ##################################################################################################

  HC = ( similar(core.row_ranks), 
         similar(core.col_ranks), 
         OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}[], 
         OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}[], 
         UnsafeSparseCore{T,N,d}[]
        )

  rowsize = (n ≤ hd ? 2d+4+(n-1)+2(n-1)^2 : 2d+4+(d+1-n)+2(d+1-n)^2)
  colsize = (n < hd ? 2d+4+(n  )+2(n  )^2 : 2d+4+(d  -n)+2(d  -n)^2)
  blockrow_starts = [1 for i=1:rowsize, ql in axes(core, 1)]
  blockrow_ends   = [0 for i=1:rowsize, ql in axes(core, 1)]
  blockcol_starts = [1 for i=1:colsize, ql in axes(core, 3)]
  blockcol_ends   = [0 for i=1:colsize, ql in axes(core, 3)]

  row_ranks = HC[1]
  col_ranks = HC[2]

  if n == 1 # First core is special; stacking horizontally (row rank should be same as `core`)
    index = 1
    for (state, r) in zip(left_states, left_sizes[0])
      R = shift_ranks(core.row_ranks, state..., N) 
        # should be filled with zeros and core.row_ranks[0] (usually 1)'s
      blockrow_ends[index:index+r-1,:] .= R'
      index += r
    end
    row_ranks .= core.row_ranks
  else # n > 1
    index = 1
    starts = [1 for ql in axes(core, 1)]
    for (state, r) in (n ≤ hd ? zip(left_states, left_sizes[n-1]) : zip(right_states, right_sizes[n-1]))
      R = shift_ranks(core.row_ranks, state..., N)
      blockrow_starts[index:index+r-1,:] .= starts' .+ (0:r-1) .* R'
      blockrow_ends[  index:index+r-1,:] .= starts' .+ (1:r)   .* R' .- 1
      starts .+= r .* R
      index += r
    end
    row_ranks .= blockrow_ends[end,:]
  end

  if n<d
    index = 1
    starts = [1 for ql in axes(core, 3)]
    for (state, r) in (n < hd ? zip(left_states, left_sizes[n]) : zip(right_states, right_sizes[n]))
      R = shift_ranks(core.col_ranks, state..., N)
      blockcol_starts[index:index+r-1,:] .= starts' .+ (0:r-1) .* R'
      blockcol_ends[  index:index+r-1,:] .= starts' .+ (1:r)   .* R' .- 1
      starts .+= r .* R
      index += r
    end
    col_ranks .= blockcol_ends[end,:]
  else # n == d # Last core: stacking vertically (column rank should be same as `core`)
    index = 1
    for (state, r) in zip(right_states, right_sizes[d])
      R = shift_ranks(core.col_ranks, state..., N) 
        # should be filled with zeros and core.col_ranks[N] (usually 1)'s
      blockcol_ends[index:index+r-1,:] .= R'
      index += r
    end
    col_ranks .= core.col_ranks
  end

  ##########################################################################################################################################

  function ×(α::Number, b::UnsafeSparseCore{T,N,d})
    if (α != 0)
      c = similar(b)
      for l in axes(b.unoccupied,1)
        c.unoccupied[l] = lmul!(α, copy(b.unoccupied[l]))
      end
      for l in axes(b.occupied, 1)
        c.occupied[l] = lmul!(α, copy(b.occupied[l]))
      end
    else
      c = UnsafeSparseCore{T,N,d}(1:0,1:0)
    end

    return c
  end

  ##########################################################################################################################################
  ###                                                    LEFT HALF OF CORES                                                              ###
  ##########################################################################################################################################

  if n < hd let
    blockrow_ranges = Dict{ Tuple{Int, Union{Int, NTuple{2,Int}, NTuple{3,Int}}}, 
                            OffsetArrays.OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}
                          }(
      (a,idx) => [ blockrow_starts[left_indices(n-1)[a](idx),l]:blockrow_ends[left_indices(n-1)[a](idx),l] for l in axes(core,1)] 
      for a in eachindex(left_states) for idx in left_iterators(n-1)[a] 
                          )
    blockcol_ranges = Dict{ Tuple{Int, Union{Int, NTuple{2,Int}, NTuple{3,Int}}}, 
                            OffsetArrays.OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}
                          }( 
      (b,idx) => [ blockcol_starts[left_indices(n  )[b](idx),r]:blockcol_ends[left_indices(n  )[b](idx),r] for r in axes(core,3)]
      for b in eachindex(left_states) for idx in left_iterators(n)[b] 
                          )

    #######################
    # One-Body Components #
    #######################
    let a=1, b=1
        OpC, colstate... =                           Id_view(   core, left_states[a]...)
        @boundscheck @assert colstate == left_states[b]
        append!(HC[3], blockrow_ranges[a, 1          ]                                     for _=1:1 )
        append!(HC[4], blockcol_ranges[b, 1          ]                                     for _=1:1 )
        append!(HC[5],                             OpC                                     for _=1:1 )
    end
    let a=1, b=2
      OpC, colstate... =                           Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                     for _=1:1 )
      append!(HC[4], blockcol_ranges[b,  n         ]                                     for _=1:1 )
      append!(HC[5],                             OpC                                     for _=1:1 )
    end
    let a=1, b=3
      OpC, colstate... =                           A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                     for _=1:1 )
      append!(HC[4], blockcol_ranges[b,    n       ]                                     for _=1:1 )
      append!(HC[5],                             OpC                                     for _=1:1 )
    end
    let a=1, b=4
      OpC, colstate... =                           AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                     for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                     for _=1:1 )
      append!(HC[5],                    t[n,n] × OpC                                     for _=1:1 )
    end
    let a=2, b=2
      OpC, colstate... =                           S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]           for i=1:n-1                         )
      append!(HC[4], blockcol_ranges[b,   i        ]           for i=1:n-1                         )
      append!(HC[5],                             OpC           for i=1:n-1                         )
    end
    let a=2, b=4
      OpC, colstate... =                           A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]           for i=1:n-1                         )
      append!(HC[4], blockcol_ranges[b, 1          ]           for i=1:n-1                         )
      append!(HC[5],                    t[i,n] × OpC           for i=1:n-1                         )
    end
    let a=3, b=3
      OpC, colstate... =                           S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,     j      ]                       for j=1:n-1             )
      append!(HC[4], blockcol_ranges[b,     j      ]                       for j=1:n-1             )
      append!(HC[5],                             OpC                       for j=1:n-1             )
    end
    let a=3, b=4
      OpC, colstate... =                           Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,     j      ]                       for j=1:n-1             )
      append!(HC[4], blockcol_ranges[b, 1          ]                       for j=1:n-1             )
      append!(HC[5],                    t[n,j] × OpC                       for j=1:n-1             )
    end
    let a=4, b=4
      OpC, colstate... =                           Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                     for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                     for _=1:1 )
      append!(HC[5],                             OpC                                     for _=1:1 )
    end
    #######################
    # Two-Body Components #
    #######################

    let a= 5, b= 5 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 5, b= 6 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b,   n        ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 5, b= 7 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b,   n        ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 5, b= 8 
      OpC, colstate... =                               AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b,  (n,  n  ) ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 6, b= 6 
      OpC, colstate... =                               S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1                                        )
      append!(HC[4], blockcol_ranges[b,   i        ]                               for i=1:n-1                                        )
      append!(HC[5],                                 OpC                           for i=1:n-1                                        )
    end
    let a= 6, b= 8 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1                                        )
      append!(HC[4], blockcol_ranges[b,  (i,  n  ) ]                               for i=1:n-1                                        )
      append!(HC[5],                                 OpC                           for i=1:n-1                                        )
    end
    let a= 6, b= 9 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1                                        )
      append!(HC[4], blockcol_ranges[b,  (i,n    ) ]                               for i=1:n-1                                        )
      append!(HC[5],                                 OpC                           for i=1:n-1                                        )
    end
    let a= 6, b=11 
      OpC, colstate... =                               AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1                           for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,         l  ]                               for i=1:n-1                           for l=n+1:d  )
      append!(HC[5],                    v[i,n,n,l] × OpC                           for i=1:n-1                           for l=n+1:d  )
    end
    let a= 7, b= 7 
      OpC, colstate... =                               S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                                         for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,       k    ]                                                         for k=1:n-1              )
      append!(HC[5],                                 OpC                                                     for k=1:n-1              )
    end
    let a= 7, b= 8 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                                         for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,  (n,  k  ) ]                                                         for k=1:n-1              )
      append!(HC[5],                                 OpC                                                     for k=1:n-1              )
    end
    let a= 7, b=10 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                                         for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,  (    k,n) ]                                                         for k=1:n-1              )
      append!(HC[5],                                 OpC                                                     for k=1:n-1              )
    end
    let a= 7, b=12 
      OpC, colstate... =                               AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                           for j=n+1:d   for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d   for k=1:n-1              )
      append!(HC[5],                    v[n,j,k,n] × OpC                                       for j=n+1:d   for k=1:n-1              )
    end
    let a= 8, b= 8 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,  k  ) ]                               for i=1:n-1               for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,  (i,  k  ) ]                               for i=1:n-1               for k=1:n-1              )
      append!(HC[5],                                 OpC                           for i=1:n-1               for k=1:n-1              )
    end
    let a= 8, b=11 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,  k  ) ]                               for i=1:n-1               for k=1:n-1 for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,         l  ]                               for i=1:n-1               for k=1:n-1 for l=n+1:d  )
      append!(HC[5],                    v[i,n,k,l] × OpC                           for i=1:n-1               for k=1:n-1 for l=n+1:d  )
    end
    let a= 8, b=12 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,  k  ) ]                               for i=1:n-1 for j=n+1:d   for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,     j      ]                               for i=1:n-1 for j=n+1:d   for k=1:n-1              )
      append!(HC[5],                    v[i,j,k,n] × OpC                           for i=1:n-1 for j=n+1:d   for k=1:n-1              )
    end
    let a= 8, b=13 
      OpC, colstate... =                               AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,  k  ) ]                               for i=1:n-1               for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b, 1          ]                               for i=1:n-1               for k=1:n-1              )
      append!(HC[5],                    v[i,n,k,n] × OpC                           for i=1:n-1               for k=1:n-1              )
    end
    let a= 9, b= 9 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,j    ) ]                               for i=1:n-1 for j=i+1:n-1                          )
      append!(HC[4], blockcol_ranges[b,  (i,j    ) ]                               for i=1:n-1 for j=i+1:n-1                          )
      append!(HC[5],                                 OpC                           for i=1:n-1 for j=i+1:n-1                          )
    end
    let a= 9, b=11 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,j    ) ]                               for i=1:n-1 for j=i+1:n-1             for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,         l  ]                               for i=1:n-1 for j=i+1:n-1             for l=n+1:d  )
      append!(HC[5],                    v[i,j,n,l] × OpC                           for i=1:n-1 for j=i+1:n-1             for l=n+1:d  )
    end
    let a=10, b=10 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,  (    k,l) ]                                                         for k=1:n-1 for l=k+1:n-1)
      append!(HC[4], blockcol_ranges[b,  (    k,l) ]                                                         for k=1:n-1 for l=k+1:n-1)
      append!(HC[5],                                 OpC                                                     for k=1:n-1 for l=k+1:n-1)
    end
    let a=10, b=12 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,  (    k,l) ]                                           for j=n+1:d   for k=1:n-1 for l=k+1:n-1)
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d   for k=1:n-1 for l=k+1:n-1)
      append!(HC[5],                    v[n,j,k,l] × OpC                                       for j=n+1:d   for k=1:n-1 for l=k+1:n-1)
    end
    let a=11, b=11 
      OpC, colstate... =                               S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,         l  ]                                                                     for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,         l  ]                                                                     for l=n+1:d  )
      append!(HC[5],                                 OpC                                                                 for l=n+1:d  )
    end
    let a=11, b=13 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,         n  ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a=12, b=12 
      OpC, colstate... =                               S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,     j      ]                                           for j=n+1:d                            )
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d                            )
      append!(HC[5],                                 OpC                                       for j=n+1:d                            )
    end
    let a=12, b=13 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a,      n     ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a=13, b=13 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == left_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
  end
  ##########################################################################################################################################
  ###                                                           MIDDLE CORE                                                              ###
  ##########################################################################################################################################

  elseif n == hd let
    blockrow_ranges = Dict{ Tuple{Int, Union{Int, NTuple{2,Int}, NTuple{3,Int}}}, 
                            OffsetArrays.OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}
                          }( 
      (a,idx) => [ blockrow_starts[left_indices(n-1)[a](idx),l]:blockrow_ends[left_indices(n-1)[a](idx),l] for l in axes(core,1)] 
      for a in eachindex(left_states ) for idx in left_iterators(n-1)[a] 
                          )
    blockcol_ranges = Dict{ Tuple{Int, Union{Int, NTuple{2,Int}, NTuple{3,Int}}}, 
                            OffsetArrays.OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}
                          }( 
      (b,idx) => [ blockcol_starts[right_indices(n )[b](idx),r]:blockcol_ends[right_indices(n )[b](idx),r] for r in axes(core,3)]
      for b in eachindex(right_states) for idx in right_iterators(n )[b] 
                          )
    #######################
    # One-Body Components #
    #######################

    let a=1, b=1
      OpC, colstate... =                           AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                       for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                       for _=1:1 )
      append!(HC[5],                    t[n,n] × OpC                                       for _=1:1 )
    end
    let a=1, b=2
      OpC, colstate... =                           A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                         for i=n+1:d             )
      append!(HC[4], blockcol_ranges[b,   i        ]                         for i=n+1:d             )
      append!(HC[5],                    t[i,n] × OpC                         for i=n+1:d             )
    end
    let a=1, b=3
      OpC, colstate... =                           Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                     for j=n+1:d )
      append!(HC[4], blockcol_ranges[b,     j      ]                                     for j=n+1:d )
      append!(HC[5],                    t[n,j] × OpC                                     for j=n+1:d )
    end
    let a=1, b=4
      OpC, colstate... =                           Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b] 
      append!(HC[3], blockrow_ranges[a, 1          ]                                       for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                       for _=1:1 )
      append!(HC[5],                             OpC                                       for _=1:1 )
    end
    let a=2, b=1
      OpC, colstate... =                           A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                         for i=1:n-1             )
      append!(HC[4], blockcol_ranges[b, 1          ]                         for i=1:n-1             )
      append!(HC[5],                    t[i,n] × OpC                         for i=1:n-1             )
    end
    let a=2, b=3
      OpC, colstate... =                           S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                         for i=1:n-1 for j=n+1:d )
      append!(HC[4], blockcol_ranges[b,     j      ]                         for i=1:n-1 for j=n+1:d )
      append!(HC[5],                    t[i,j] × OpC                         for i=1:n-1 for j=n+1:d )
    end
    let a=3, b=1
      OpC, colstate... =                           Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,     j      ]                                     for j=1:n-1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                     for j=1:n-1 )
      append!(HC[5],                    t[n,j] × OpC                                     for j=1:n-1 )
    end
    let a=3, b=2
      OpC, colstate... =                           S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,     j      ]                         for i=n+1:d for j=1:n-1 )
      append!(HC[4], blockcol_ranges[b,   i        ]                         for i=n+1:d for j=1:n-1 )
      append!(HC[5],                    t[i,j] × OpC                         for i=n+1:d for j=1:n-1 )
    end
    let a=4, b=1
      OpC, colstate... =                           Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                       for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                       for _=1:1 )
      append!(HC[5],                             OpC                                       for _=1:1 )
    end
    #######################
    # Two-Body Components #
    #######################

    let a= 5, b= 8 
      OpC, colstate... =                               AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                           for j=n+1:d               for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,  (  j,  l) ]                                           for j=n+1:d               for l=n+1:d  )
      append!(HC[5],                    v[n,j,n,l] × OpC                                       for j=n+1:d               for l=n+1:d  )
    end
    let a= 5, b=11 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b,       n    ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 5, b=12 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b,   n        ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 5, b=13 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 6, b= 7 
      OpC, colstate... =                               AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1                           for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,         l  ]                               for i=1:n-1                           for l=n+1:d  )
      append!(HC[5],                    v[i,n,n,l] × OpC                           for i=1:n-1                           for l=n+1:d  )
    end
    let a= 6, b= 8 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1 for j=n+1:d               for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,  (  j,  l) ]                               for i=1:n-1 for j=n+1:d               for l=n+1:d  )
      append!(HC[5],                    v[i,j,n,l] × OpC                           for i=1:n-1 for j=n+1:d               for l=n+1:d  )
    end
    let a= 6, b=10 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1               for k=n+1:d for l=k+1:d  )
      append!(HC[4], blockcol_ranges[b,  (    k,l) ]                               for i=1:n-1               for k=n+1:d for l=k+1:d  )
      append!(HC[5],                    v[i,n,k,l] × OpC                           for i=1:n-1               for k=n+1:d for l=k+1:d  )
    end
    let a= 6, b=12 
      OpC, colstate... =                               S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1                                        )
      append!(HC[4], blockcol_ranges[b,   i        ]                               for i=1:n-1                                        )
      append!(HC[5],                                 OpC                           for i=1:n-1                                        )
    end
    let a= 7, b= 6 
      OpC, colstate... =                               AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                           for j=n+1:d   for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d   for k=1:n-1              )
      append!(HC[5],                    v[n,j,k,n] × OpC                                       for j=n+1:d   for k=1:n-1              )
    end
    let a= 7, b= 8 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                           for j=n+1:d   for k=1:n-1 for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,  (  j,  l) ]                                           for j=n+1:d   for k=1:n-1 for l=n+1:d  )
      append!(HC[5],                    v[n,j,k,l] × OpC                                       for j=n+1:d   for k=1:n-1 for l=n+1:d  )
    end
    let a= 7, b= 9 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                               for i=n+1:d for j=i+1:d   for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,  (i,j    ) ]                               for i=n+1:d for j=i+1:d   for k=1:n-1              )
      append!(HC[5],                    v[i,j,k,n] × OpC                           for i=n+1:d for j=i+1:d   for k=1:n-1              )
    end
    let a= 7, b=11 
      OpC, colstate... =                               S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                                         for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,       k    ]                                                         for k=1:n-1              )
      append!(HC[5],                                 OpC                                                     for k=1:n-1              )
    end
    let a= 8, b= 5 
      OpC, colstate... =                               AdagA_view(core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,  k  ) ]                               for i=1:n-1               for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b, 1          ]                               for i=1:n-1               for k=1:n-1              )
      append!(HC[5],                    v[i,n,k,n] × OpC                           for i=1:n-1               for k=1:n-1              )
    end
    let a= 8, b= 6 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,  k  ) ]                               for i=1:n-1 for j=n+1:d   for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,     j      ]                               for i=1:n-1 for j=n+1:d   for k=1:n-1              )
      append!(HC[5],                    v[i,j,k,n] × OpC                           for i=1:n-1 for j=n+1:d   for k=1:n-1              )
    end
    let a= 8, b= 7 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,  k  ) ]                               for i=1:n-1               for k=1:n-1 for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,         l  ]                               for i=1:n-1               for k=1:n-1 for l=n+1:d  )
      append!(HC[5],                    v[i,n,k,l] × OpC                           for i=1:n-1               for k=1:n-1 for l=n+1:d  )
    end
    let a= 8, b= 8 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,  k  ) ]                               for i=1:n-1 for j=n+1:d   for k=1:n-1 for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,  (  j,  l) ]                               for i=1:n-1 for j=n+1:d   for k=1:n-1 for l=n+1:d  )
      append!(HC[5],                    v[i,j,k,l] × OpC                           for i=1:n-1 for j=n+1:d   for k=1:n-1 for l=n+1:d  )
    end
    let a= 9, b= 7 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,j    ) ]                               for i=1:n-1 for j=i+1:n-1             for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,         l  ]                               for i=1:n-1 for j=i+1:n-1             for l=n+1:d  )
      append!(HC[5],                    v[i,j,n,l] × OpC                           for i=1:n-1 for j=i+1:n-1             for l=n+1:d  )
    end
    let a= 9, b=10 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,j    ) ]                               for i=1:n-1 for j=i+1:n-1 for k=n+1:d for l=k+1:d  )
      append!(HC[4], blockcol_ranges[b,  (    k,l) ]                               for i=1:n-1 for j=i+1:n-1 for k=n+1:d for l=k+1:d  )
      append!(HC[5],                    v[i,j,k,l] × OpC                           for i=1:n-1 for j=i+1:n-1 for k=n+1:d for l=k+1:d  )
    end
    let a=10, b= 6 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (    k,l) ]                                           for j=n+1:d   for k=1:n-1 for l=k+1:n-1)
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d   for k=1:n-1 for l=k+1:n-1)
      append!(HC[5],                    v[n,j,k,l] × OpC                                       for j=n+1:d   for k=1:n-1 for l=k+1:n-1)
    end
    let a=10, b= 9 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (    k,l) ]                               for i=n+1:d for j=i+1:d   for k=1:n-1 for l=k+1:n-1)
      append!(HC[4], blockcol_ranges[b,  (i,j    ) ]                               for i=n+1:d for j=i+1:d   for k=1:n-1 for l=k+1:n-1)
      append!(HC[5],                    v[i,j,k,l] × OpC                           for i=n+1:d for j=i+1:d   for k=1:n-1 for l=k+1:n-1)
    end
    let a=11, b= 5 
      OpC, colstate... =                               A_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,         n  ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a=11, b= 7 
      OpC, colstate... =                               S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,         l  ]                                                                     for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,         l  ]                                                                     for l=n+1:d  )
      append!(HC[5],                                 OpC                                                                 for l=n+1:d  )
    end
    let a=12, b= 5 
      OpC, colstate... =                               Adag_view( core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,     n      ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a=12, b= 6 
      OpC, colstate... =                               S_view(    core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,     j      ]                                           for j=n+1:d                            )
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d                            )
      append!(HC[5],                                 OpC                                       for j=n+1:d                            )
    end
    let a=13, b= 5 
      OpC, colstate... =                               Id_view(   core, left_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
  end
  ##########################################################################################################################################
  ###                                                    RIGHT HALF OF CORES                                                             ###
  ##########################################################################################################################################

  else let # hd<n≤d
    blockrow_ranges = Dict{
                            Tuple{Int, Union{Int, NTuple{2,Int}, NTuple{3,Int}}}, 
                            OffsetArrays.OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}
                          }( 
      (a,idx) => [ blockrow_starts[right_indices(n-1)[a](idx),l]:blockrow_ends[right_indices(n-1)[a](idx),l] for l in axes(core,1)] 
      for a in eachindex(right_states)  for idx in right_iterators(n-1)[a] 
                          )
    blockcol_ranges = Dict{
                            Tuple{Int, Union{Int, NTuple{2,Int}, NTuple{3,Int}}}, 
                            OffsetArrays.OffsetVector{UnitRange{Int}, Vector{UnitRange{Int}}}
                          }( 
      (b,idx) => [ blockcol_starts[right_indices(n  )[b](idx),r]:blockcol_ends[right_indices(n  )[b](idx),r] for r in axes(core,3)]
      for b in eachindex(right_states) for idx in right_iterators(n  )[b] 
                          )
    #######################
    # One-Body Components #
    #######################

    let a=1, b=1
      OpC, colstate... =                           Id_view(   core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b] 
      append!(HC[3], blockrow_ranges[a, 1          ]                                     for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                     for _=1:1 )
      append!(HC[5],                             OpC                                     for _=1:1 )
    end
    let a=2, b=1
      OpC, colstate... =                           Adag_view( core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   n        ]                                     for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                     for _=1:1 )
      append!(HC[5],                             OpC                                     for _=1:1 )
    end
    let a=3, b=1
      OpC, colstate... =                           A_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,     n      ]                                     for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                     for _=1:1 )
      append!(HC[5],                             OpC                                     for _=1:1 )
    end
    let a=4, b=1
      OpC, colstate... =                           AdagA_view(core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                     for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                     for _=1:1 )
      append!(HC[5],                    t[n,n] × OpC                                     for _=1:1 )
    end
    let a=2, b=2
      OpC, colstate... =                           S_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                       for i=n+1:d             )
      append!(HC[4], blockcol_ranges[b,   i        ]                       for i=n+1:d             )
      append!(HC[5],                             OpC                       for i=n+1:d             )
    end
    let a=4, b=2
      OpC, colstate... =                           A_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                       for i=n+1:d             )
      append!(HC[4], blockcol_ranges[b,   i        ]                       for i=n+1:d             )
      append!(HC[5],                    t[i,n] × OpC                       for i=n+1:d             )
    end
    let a=3, b=3
      OpC, colstate... =                           S_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,     j      ]                                   for j=n+1:d )
      append!(HC[4], blockcol_ranges[b,     j      ]                                   for j=n+1:d )
      append!(HC[5],                             OpC                                   for j=n+1:d )
    end
    let a=4, b=3
      OpC, colstate... =                           Adag_view( core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                   for j=n+1:d )
      append!(HC[4], blockcol_ranges[b,     j      ]                                   for j=n+1:d )
      append!(HC[5],                    t[n,j] × OpC                                   for j=n+1:d )
    end
    let a=4, b=4
      OpC, colstate... =                           Id_view(   core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                     for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                     for _=1:1 )
      append!(HC[5],                             OpC                                     for _=1:1 )
    end
    #######################
    # Two-Body Components #
    #######################

    let a= 5, b= 5 
      OpC, colstate... =                               Id_view(   core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 6, b= 5 
      OpC, colstate... =                               Adag_view( core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   n        ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 7, b= 5 
      OpC, colstate... =                               A_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   n        ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 8, b= 5 
      OpC, colstate... =                               AdagA_view(core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, (n,  n  )  ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a= 6, b= 6 
      OpC, colstate... =                               S_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,     j      ]                                           for j=n+1:d                            )
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d                            )
      append!(HC[5],                                 OpC                                       for j=n+1:d                            )
    end
    let a= 8, b= 6 
      OpC, colstate... =                               A_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (  j,  n) ]                                           for j=n+1:d                            )
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d                            )
      append!(HC[5],                                 OpC                                       for j=n+1:d                            )
    end
    let a= 9, b= 6 
      OpC, colstate... =                               Adag_view( core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (n,j    ) ]                                           for j=n+1:d                            )
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d                            )
      append!(HC[5],                                 OpC                                       for j=n+1:d                            )
    end
    let a=11, b= 6 
      OpC, colstate... =                               AdagA_view(core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                           for j=n+1:d for k=1:n-1                )
      append!(HC[4], blockcol_ranges[b,     j      ]                                           for j=n+1:d for k=1:n-1                )
      append!(HC[5],                    v[n,j,k,n] × OpC                                       for j=n+1:d for k=1:n-1                )
    end
    let a= 7, b= 7 
      OpC, colstate... =                               S_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,         l  ]                                                                   for l=n+1:d    )
      append!(HC[4], blockcol_ranges[b,         l  ]                                                                   for l=n+1:d    )
      append!(HC[5],                                 OpC                                                               for l=n+1:d    )
    end
    let a= 8, b= 7 
      OpC, colstate... =                               Adag_view( core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (  n,  l) ]                                                                   for l=n+1:d    )
      append!(HC[4], blockcol_ranges[b,         l  ]                                                                   for l=n+1:d    )
      append!(HC[5],                                 OpC                                                               for l=n+1:d    )
    end
    let a=10, b= 7 
      OpC, colstate... =                               A_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (    n,l) ]                                                                   for l=n+1:d    )
      append!(HC[4], blockcol_ranges[b,         l  ]                                                                   for l=n+1:d    )
      append!(HC[5],                                 OpC                                                               for l=n+1:d    )
    end
    let a=12, b= 7 
      OpC, colstate... =                               AdagA_view(core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1                         for l=n+1:d    )
      append!(HC[4], blockcol_ranges[b,         l  ]                               for i=1:n-1                         for l=n+1:d    )
      append!(HC[5],                    v[i,n,n,l] × OpC                           for i=1:n-1                         for l=n+1:d    )
    end
    let a= 8, b= 8 
      OpC, colstate... =                               Id_view(   core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (  j,  l) ]                                           for j=n+1:d              for l=n+1:d   )
      append!(HC[4], blockcol_ranges[b,  (  j,  l) ]                                           for j=n+1:d              for l=n+1:d   )
      append!(HC[5],                                 OpC                                       for j=n+1:d              for l=n+1:d   )
    end
    let a=11, b= 8 
      OpC, colstate... =                               Adag_view( core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                           for j=n+1:d  for k=1:n-1 for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,  (  j,  l) ]                                           for j=n+1:d  for k=1:n-1 for l=n+1:d  )
      append!(HC[5],                    v[n,j,k,l] × OpC                                       for j=n+1:d  for k=1:n-1 for l=n+1:d  )
    end
    let a=12, b= 8 
      OpC, colstate... =                               A_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1 for j=n+1:d              for l=n+1:d  )
      append!(HC[4], blockcol_ranges[b,  (  j,  l) ]                               for i=1:n-1 for j=n+1:d              for l=n+1:d  )
      append!(HC[5],                    v[i,j,n,l] × OpC                           for i=1:n-1 for j=n+1:d              for l=n+1:d  )
    end
    let a=13, b= 8 
      OpC, colstate... =                               AdagA_view(core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                           for j=n+1:d              for l=n+1:d   )
      append!(HC[4], blockcol_ranges[b,  (  j,  l) ]                                           for j=n+1:d              for l=n+1:d   )
      append!(HC[5],                    v[n,j,n,l] × OpC                                       for j=n+1:d              for l=n+1:d   )
    end
    let a= 9, b= 9 
      OpC, colstate... =                               Id_view(   core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (i,j    ) ]                               for i=n+1:d for j=i+1:d                            )
      append!(HC[4], blockcol_ranges[b,  (i,j    ) ]                               for i=n+1:d for j=i+1:d                            )
      append!(HC[5],                                 OpC                           for i=n+1:d for j=i+1:d                            )
    end
    let a=11, b= 9 
      OpC, colstate... =                               A_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (    k  ) ]                               for i=n+1:d for j=i+1:d   for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,  (i,j    ) ]                               for i=n+1:d for j=i+1:d   for k=1:n-1              )
      append!(HC[5],                    v[i,j,k,n] × OpC                           for i=n+1:d for j=i+1:d   for k=1:n-1              )
    end
    let a=10, b=10 
      OpC, colstate... =                               Id_view(   core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,  (    k,l) ]                                                         for k=n+1:d for l=k+1:d  )
      append!(HC[4], blockcol_ranges[b,  (    k,l) ]                                                         for k=n+1:d for l=k+1:d  )
      append!(HC[5],                                 OpC                                                     for k=n+1:d for l=k+1:d  )
    end
    let a=12, b=10 
      OpC, colstate... =                               Adag_view( core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1               for k=n+1:d for l=k+1:d  )
      append!(HC[4], blockcol_ranges[b,  (    k,l) ]                               for i=1:n-1               for k=n+1:d for l=k+1:d  )
      append!(HC[5],                    v[i,n,k,l] × OpC                           for i=1:n-1               for k=n+1:d for l=k+1:d  )
    end
    let a=11, b=11 
      OpC, colstate... =                               S_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,       k    ]                                                         for k=1:n-1              )
      append!(HC[4], blockcol_ranges[b,       k    ]                                                         for k=1:n-1              )
      append!(HC[5],                                 OpC                                                     for k=1:n-1              )
    end
    let a=13, b=11 
      OpC, colstate... =                               A_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b,       n    ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a=12, b=12 
      OpC, colstate... =                               S_view(    core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a,   i        ]                               for i=1:n-1                                        )
      append!(HC[4], blockcol_ranges[b,   i        ]                               for i=1:n-1                                        )
      append!(HC[5],                                 OpC                           for i=1:n-1                                        )
    end
    let a=13, b=12 
      OpC, colstate... =                               Adag_view( core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b,   n        ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
    let a=13, b=13 
      OpC, colstate... =                               Id_view(   core, right_states[a]...)
      @boundscheck @assert colstate == right_states[b]
      append!(HC[3], blockrow_ranges[a, 1          ]                                                                        for _=1:1 )
      append!(HC[4], blockcol_ranges[b, 1          ]                                                                        for _=1:1 )
      append!(HC[5],                                 OpC                                                                    for _=1:1 )
    end
  end
  end

  return HC
end

function H_matvec(tt_in::TTvector{T,N,d}, t::Matrix{T}, v::Array{T,4}; reduced::Bool=false) where {T<:Number,N,d}
  @boundscheck begin
    @assert size(t) == (d,d)
    @assert size(v) == (d,d,d,d)
  end

  reduced || (v = reduce(v))

  cores = [SparseCore{T,N,d}(k) for k=1:d]

  for n=1:d
    sC = sparse_H_matvec(core(tt_in,n), t, v)

    C = cores[n]
    C.row_ranks .= sC[1]
    C.col_ranks .= sC[2]

    for l in axes(C,1), r in (l,l+1) ∩ axes(C,3)
      X = zeros(T, C.row_ranks[l], C.col_ranks[r])
      for (I,J,v) in zip(sC[3:5]...)
        if isnonzero(v,l,r)
          X[I[l], J[r]] = v[l,r]
        end
      end
      C[l,r] = X
    end
  end

  return cores2tensor(cores)
end

function RayleighQuotient(tt_in::TTvector{T,N,d}, t::Matrix{T}, v::Array{T,4}; reduced::Bool=false, orthogonalize::Bool=false) where {T<:Number,N,d}
  @boundscheck begin
    @assert size(t) == (d,d)
    @assert size(v) == (d,d,d,d)
  end

  if (orthogonalize)
    rightOrthogonalize!(tt_in)
  end

  reduced || (v = reduce(v))

  p = OffsetVector([n == N ? ones(T,1,1) : zeros(T,0,0) for n in 0:N], 0:N)

  for k=d:-1:1
    Xₖ = core(tt_in,k)
    HXₖ = sparse_H_matvec(Xₖ, t, v)
    for l in axes(Xₖ,1)
      Pl = zeros(T, HXₖ[1][l], Xₖ.row_ranks[l])
      for r in axes(Xₖ,3) ∩ (l:l+1)
        if isnonzero(Xₖ,l,r)
          for (I,J,V) in zip(HXₖ[3],HXₖ[4],HXₖ[5])
            if isnonzero(V,l,r)
              mul!( view(Pl, I[l], :), 
                    data(V[l,r]),
                    p[r][J[r],:] * adjoint(data(Xₖ[l,r])),
                    factor(V[l,r]) * conj(factor(Xₖ[l,r])),
                    T(1)
                  )
            end
          end
        end
      end
      p[l] = Pl
    end
  end

  return p[0][1]
end

end