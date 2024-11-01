using OffsetArrays

"""
    TTvector{T <:Number,N,d}

Implementation of Block Sparse TTvector class and associated core functions.
Mode sizes are assumed to be all 2 and total quantum number is N.
"""
mutable struct TTvector{T<:Number,N,d,S<:AbstractMatrix{T}} # <: AbstractArray{T,d}
  r::Vector{OffsetVector{Int,Vector{Int}}}
  cores::Vector{SparseCore{T,N,d,S}}
  orthogonal::Bool
  corePosition::Int

  function TTvector(r::Vector{OffsetVector{Int,Vector{Int}}}, 
                    cores::Vector{SparseCore{T,N,d,S}}) where {T<:Number,N,d,S<:AbstractMatrix{T}}
    @boundscheck begin
      length(cores) == d || throw(DimensionMismatch("Trying to form $d-dimensional tensor train with only $(length(cores)) cores"))
      N ≤ d || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
    end
    
    for k=1:d
      @assert cores[k].k == k
      @assert cores[k].row_ranks == r[k]
      @assert cores[k].col_ranks == r[k+1]
    end
    return new{T,N,d,S}(r,cores,false,0)
  end
end

function check(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  # Check rank consistency
  for k=1:d
    for m in core(tt,k).row_qn
      if core(tt,k).row_ranks[m] !== tt.r[k][m]
        @warn "Row ranks at $(m) for $(k)-th block are inconsistent between core and tensor train: $(core(tt,k).row_ranks[m]) instead of $(tt.r[k][m])"
      end
    end
    for n in core(tt,k).col_qn
      if core(tt,k).col_ranks[n] !== tt.r[k+1][n]
        @warn "Column ranks at $(n) for $(k)-th block are inconsistent between core and tensor train: $(core(tt,k).col_ranks[n]) instead of $(tt.r[k+1][n])"
      end
    end

    for l in core(tt,k).row_qn, r in core(tt,k).col_qn ∩ (l:l+1)
      if size(core(tt,k)[l,r]) !== (core(tt,k).row_ranks[l], core(tt,k).col_ranks[r])
        @warn "Wrong $(k)-th core ($l,$r) block size: $(size(core(tt,k)[l,r])) instead of $((core(tt,k).row_ranks[l], core(tt,k).col_ranks[r]))"
      end
    end
  end
end

function Base.show(io::IO, ::MIME"text/plain", tt::TTvector{T,N,d}) where {T<:Number,N,d}
  if get(io, :compact, true)
    str = "TTvector{$T,$N,$d}. Maximum rank $(maximum(sum.(tt.r)))"
  else
    # Manage some formatting and padding
    strr = ["r[i]=$(sum(r))" for r in tt.r]
    len = max(length.(strr)...)
    padr = len .- length.(strr)
    strr = ["$(s)"*" "^(pad+len+3)          for (s,pad) in zip(strr, padr)]

    str = string("TTvector{$T,$N,$d}. Ranks are:\n", strr...)
  end
    print(io, str)
end

function Base.show(io::IO, tt::TTvector{T,N,d}) where {T<:Number,N,d}
    str = "TTvector{$T,$N,$d}. Maximum rank $(maximum(sum.(tt.r)))"
    println(io, str)
    for k=1:d
      println(io)
      show(io, core(tt,k))
    end
end

function Base.summary(io::IO, tt::TTvector{T,N,d}) where {T<:Number,N,d}
  show(summary(tt))
end

function Base.summary(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  if d>0
    return string("$(d)-dimensional TTvector object with eltype $T, ranks $(tt.r)")
  else
    return string("0-dimensional tensor train object with eltype $T")
  end
end

"""
    tt = tt_zeros(d::Int, N::Int, [T=Float64])

Compute the d-dimensional sTT-tensor of all zeros.
"""
function tt_zeros(d::Int, N::Int, ::Type{T}=Float64) where T<:Number
  r = [[0 for n in occupation_qn(N,d,k)] for k=1:d+1]
  r[1][0] = 1
  r[d+1][N] = 1
  cores = [SparseCore{T,N,d}(k, r[k], r[k+1]) for k=1:d]

  tt = TTvector(r, cores)
  check(tt)

  return tt
end

function tt_zeros(::Val{d}, ::Val{N}, ::Type{T}=Float64) where {T<:Number,N,d}
  r = [[0 for n in occupation_qn(N,d,k)] for k=1:d+1]
  r[1][0] = 1
  r[d+1][N] = 1
  cores = [SparseCore{T,N,d}(k, r[k], r[k+1]) for k=1:d]

  tt = TTvector(r, cores)
  check(tt)

  return tt
end

function tt_zeros(::Val{d}, ::Val{N}, r::Vector{OffsetVector{Int,Vector{Int}}}, ::Type{T}=Float64) where {T<:Number,N,d}
  @boundscheck (length(r) == d+1) || (length(r) == d-1)

  if length(r) == d+1
    for k=1:d
      @boundscheck axes(r[k],1) == occupation_qn(N,d,k)
    end
    cores = [SparseCore{T,N,d}(k, r[k], r[k+1]) for k=1:d]

  else # length(r) == d-1
    for k=1:d-1
      @boundscheck axes(r[k],1) == occupation_qn(N,d,k-1)
    end
    cores = [SparseCore{T,N,d}(k, k>1 ? r[k-1] : 1, k<d ? r[k] : 1) for k=1:d]
  end
  tt = TTvector(r, cores)
  check(tt)

  return(tt)
end

"""
    tt = tt_ones(d::Int, N::Int, [T=Float64])

Compute the d-dimensional sTT-tensor of all zeros.
"""
function tt_ones(d::Int, N::Int, ::Type{T}=Float64) where T<:Number
  r = [[1 for n in occupation_qn(N,d,k)] for k=1:d+1]
  cores = [SparseCore{T,N,d}(k, r[k], r[k+1]) for k=1:d]

  for k=1:d
    fill!.(cores[k].unoccupied, 1)
    fill!.(cores[k].occupied, 1)
  end

  tt = TTvector(r, cores)
  check(tt)

  return tt
end

function tt_ones(::Val{d}, ::Val{N}, ::Type{T}=Float64) where {T<:Number,N,d}
  r = [[1 for n in occupation_qn(N,d,k)] for k=1:d+1]
  cores = [SparseCore{T,N,d}(k, r[k], r[k+1]) for k=1:d]

  for k=1:d
    fill!.(cores[k].unoccupied, 1)
    fill!.(cores[k].occupied, 1)
  end

  tt = TTvector(r, cores)
  check(tt)

  return tt
end



function tt_state(state::NTuple{N, Int}, ::Val{d}, ::Type{T}=Float64)::TTvector{T,N,d} where {T<:Number,N,d}
  @assert all(1 .≤ state .≤ d)
  @assert allunique(state)

  return tt_state([k∈state for k=1:d], T)
end

function tt_state(state::Vector{Bool}, ::Type{T}=Float64) where T<:Number
  d = length(state)
  N = sum(state)

  Nl = [0;cumsum(state)]
  cores = [SparseCore{T,N,d}(state[k], Nl[k], k) for k=1:d]
  r = [deepcopy(cores[k].row_ranks) for k=1:d]
  r = [r;[deepcopy(cores[d].col_ranks)]]

  for k=1:d-1
    @assert r[k+1] == cores[k].col_ranks
  end

  tt = TTvector(r, cores)
  check(tt)

  return tt
end

"""
    add_non_essential_dims(tt::TTvector{T,N,d}, new_n::NTuple{newd,Int}, old_vars_pos::NTuple{d,Int})

Create `enlarged_tt` object with new 'dummy' modes. Mode sizes of `enlarged_tt` are equal to `new_n`, and the old modal indices positions should be given in the sorted list `old_vars_pos`.
"""
function add_non_essential_dims(tt::TTvector{T,N,d,Matrix{T}}, newd::Int, old_vars_pos::NTuple{d,Int}) where {T<:Number,N,d}
  @assert issorted(old_vars_pos)
  @assert length(old_vars_pos) == d
  @assert newd ≥ d

  r = [[0 for n in occupation_qn(N,newd,k)] for k=1:newd+1]
  cores = Vector{SparseCore{T,N,newd,Matrix{T}}}(undef, newd)

  # New dimensions corresponding to real ones
  for (k, kk) = enumerate(old_vars_pos)
    Ck = core(tt, k)
    rqn = axes(Ck,1)
    cqn = axes(Ck,3)

    # Compute new tensor ranks
    r[kk  ][rqn] .= tt.r[k]
    r[kk+1][cqn] .= tt.r[k+1]

    # Reconcile tensor ranks with cores
    cores[kk] = SparseCore{T,N,newd}(kk, r[kk], r[kk+1])

    for l in axes(cores[kk],1) ∩ rqn, r ∈ (l:l+1) ∩ axes(cores[kk],3) ∩ cqn
      cores[kk][l,r] = core(tt,k)[l,r]
    end
  end

  # New dimensions before the first real dimension
  C1 = core(tt,1)
  qn = axes(C1,1)

  for kk in 1:old_vars_pos[1]-1
    # Compute new tensor ranks
    r[kk  ][qn] .= C1.row_ranks
    r[kk+1][qn] .= C1.row_ranks

    # Reconcile tensor ranks with cores
    cores[kk] = SparseCore{T,N,newd}(kk, r[kk], r[kk+1])

    # Add appropriate cores
    for l in axes(cores[kk],1) ∩ qn ∩ axes(cores[kk],3)
      cores[kk][l,l] = Array{T}(I(r[kk][l]))
    end
  end

  # New dimensions between real ones
  for k=2:d
    Ck = core(tt, k)
    qn = axes(Ck,1)

    for kk=old_vars_pos[k-1]+1:old_vars_pos[k]-1

      # Compute new tensor ranks
      r[kk  ][qn] .= Ck.row_ranks
      r[kk+1][qn] .= Ck.row_ranks

      # Reconcile tensor ranks with cores
      cores[kk] = SparseCore{T,N,newd}(kk, r[kk], r[kk+1])

      # Add appropriate cores
      for l in axes(cores[kk],1) ∩ qn ∩ axes(cores[kk],3)
        cores[kk][l,l] = Array{T}(I(r[kk][l]))
      end
    end
  end

  # New dimensions after the last real dimension
  Cd = core(tt,d)
  qn = axes(Cd,3)

  for kk in old_vars_pos[d]+1:newd

    # Compute new tensor ranks
    r[kk  ][qn] .= deepcopy(Ck.col_ranks)
    r[kk+1][qn] .= deepcopy(Ck.col_ranks)

    # Reconcile tensor ranks with cores
    cores[kk] = SparseCore{T,N,newd}(kk, r[kk], r[kk+1])

    # Add appropriate cores
    for l in axes(cores[kk],1) ∩ qn ∩ axes(cores[kk],3)
      cores[kk][l,l] = Array{T}(I(rank(enlarged_tt,kk,l)))
    end
  end
  enlarged_tt = TTvector(r, cores)
  check(enlarged_tt)

  # Assemble new enlarged TTvector and return it
  return enlarged_tt
end

"""
    cores2tensor(cores::Array{Array{T,3},1})

Create a TT-tensor view on a list of 3D SparseCores.
The TT-tensor shares the same underlying data structure with `cores` immediately after the call.
"""
function cores2tensor(cores::Vector{SparseCore{T,N,d,S}}) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  @assert d == length(cores)
  for k=1:d
    @assert cores[k].k == k
  end

  r = [[deepcopy(cores[k].row_ranks) for k=1:d]..., deepcopy(cores[d].col_ranks)]

  for k=2:d
    @assert cores[d-1].col_ranks == cores[d].row_ranks
  end

  tt = TTvector(r, cores)
  check(tt)

  return tt
end

"""
    tensor2cores(tt::TTvector{T,N,d})

Create a view on the list of SparseCores from `tt`.
"""
function tensor2cores(tt::TTvector)
  return tt.cores
end


function core(tt::TTvector, k::Int)
  return tt.cores[k]
end

function set_core!(tt::TTvector{T,N,d}, new_core::SparseCore{T,N,d}) where {T<:Number,N,d}
  tt.cores[new_core.k] = new_core
  return tt
end

function set_cores!(tt::TTvector{T,N,d,S}, first_core::SparseCore{T,N,d,S}, second_core::SparseCore{T,N,d,S}) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  @boundscheck begin
    @assert second_core.k == first_core.k+1
    @assert first_core.col_ranks == second_core.row_ranks
  end
  k = first_core.k
  tt.cores[k  ] = first_core
  tt.cores[k+1] = second_core
  tt.r[k+1]     = first_core.col_ranks
  return tt
end

"""
    ndims(tt::TTvector{T,N,d})

Compute the number of dimensions of `tt`.
"""
function Base.ndims(::TTvector{T,N,d}) where {T<:Number,N,d}
  return d
end

"""
    size(tt::TTvector{T,N,d}, [dim])

Return a tuple containing the dimensions of `tt`. Optionally you can specify a dimension `dim` to just get the length of that dimension.
"""
function Base.size(::TTvector{T,N,d}) where {T<:Number,N,d}
  return ntuple(k->2, d)
end

function Base.size(::TTvector{T,N,d}, dim::Int) where {T<:Number,N,d}
  return 2
end

"""
    Base.length(tt::TTvector{T,N,d})

Compute the number of elements in `tt`, viewed as a `d`-dimensional Array.
"""
function Base.length(::TTvector{T,N,d}) where {T<:Number,N,d}
  return 2^d
end


"""
    rank(tt::TTvector{T,N,d}, [ind::Int], [n::Int])

Return a tuple containing the TT-ranks of `tt`. Optionally you can specify an index `ind` to just get the rank of that edge, and additionally the block index `n`.
"""
function LinearAlgebra.rank(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  return tt.r
end

function LinearAlgebra.rank(tt::TTvector{T,N,d}, k::Int) where {T<:Number,N,d}
  @boundscheck 1 ≤ k ≤ d+1 || throw(BoundsError(tt, k))
  return tt.r[k]
end

function LinearAlgebra.rank(tt::TTvector{T,N,d}, k::Int, n::Int) where {T<:Number,N,d}
  @boundscheck begin
    1 ≤ k ≤ d+1 || throw(BoundsError(tt, k))
    n ∈ axes(tt.r[k],1) || throw(BoundsError(tt.r[k], n))
    if k ≤ d
      tt.r[k][n] == core(tt,k).row_ranks[n] || throw(DimensionMismatch("Tensor ranks $(tt.r[k]) do not match $k-th core row ranks $(core(tt,k).row_ranks)"))
    end
    if k ≥ 2
      tt.r[k][n] == core(tt,k-1).col_ranks[n] || throw(DimensionMismatch("Tensor ranks $(tt.r[k]) do not match $(k-1)-th core column ranks $(core(tt,k-1).col_ranks)"))
    end
  end
  return tt.r[k][n]
end

function IdFrame(k::Int, tt::TTvector{T,N,d}) where {T<:Number,N,d}
  return Frame{T,N,d}(k, [T(1)*I(rank(tt,k)) for k in occupation_qn(N,d,k)])
end

function IdFrame(::Val{N}, ::Val{d}, k::Int, r::Int=1, ::Type{T}=Float64) where {T<:Number,N,d}
  return Frame{T,N,d}(k, [T(1)*I(r) for n in occupation_qn(N,d,k)])
end

"""
    Array(tt::TTvector{T,N,d}, [sizes::NTuple{d2,Int}])

Return a dense Array representing the same object as `tt`. Optionally you can specify `sizes` to reshape the returned Array.
"""
function Base.Array(tt::TTvector{T,N,d}, sizes::NTuple{d2,Int}) where {T<:Number,N,d,d2}
  @assert prod(sizes) == (2^d)*rank(tt,1,0)*rank(tt,d+1,N)
  e = [reshape([1,0], (1,2,1)), reshape([0,1], (1,2,1))]
  a = OffsetVector([ones(T,rank(tt,1,0))], axes(core(tt,1),1))
 
  for k=1:d
    A = core(tt,k)
    
    a = OffsetVector( [ rank(tt,k+1,n) > 0 ? 
                           sum(
                            reshape(Array(a[m] * A[m,1+n-m,n]), (:,1,rank(tt,k+1,n))) .* e[1+n-m] 
                            for m in axes(A,1) ∩ (n-1:n)
                           ) : zeros(rank(tt,1,0)*2^k, 0)
                        for n in axes(A,3) 
                      ], axes(A,3))
    a = [rank(tt,k+1,n) > 0 ? reshape(a[n], (:,rank(tt,k+1,n))) : a[n] for n in axes(A,3)]
  end
  a = reshape(a[N], sizes)
end

function Base.Array(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  if (rank(tt,1,0) == 1) && (rank(tt,d+1,N) == 1)
    sizes = ntuple(k->2, d)
  elseif (rank(tt,1,0) == 1) && (rank(tt,d+1,N) != 1)
    sizes = ntuple(k->(k == d+1 ? rank(tt,d+1,N) : 2), d+1)
  elseif (rank(tt,1,0) != 1) && (rank(tt,d+1,N) == 1)
    sizes = ntuple(k->(k == 1 ? rank(tt,1,0) : 2), d+1)
  else
    sizes = ntuple(k->(k==1 ? rank(tt,1,0) : k==d+2 ? rank(tt,d+1,N) : 2), d+2)
  end
  return Array(tt, sizes)
end
