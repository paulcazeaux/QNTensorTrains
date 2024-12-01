"""
    TTvector{T <:Number,Nup,Ndn,d}

Implementation of Block Sparse TTvector class and associated core functions.
Mode sizes are assumed to be all 2 and total quantum number is N.
"""
mutable struct TTvector{T<:Number,Nup,Ndn,d,S<:AbstractMatrix{T}} # <: AbstractArray{T,d}
  r::Vector{Matrix{Int}}
  cores::Vector{SparseCore{T,Nup,Ndn,d,S}}
  orthogonal::Bool
  corePosition::Int

  function TTvector(r::Vector{Matrix{Int}}, 
                    cores::Vector{SparseCore{T,Nup,Ndn,d,S}}) where {T<:Number,Nup,Ndn,d,S<:AbstractMatrix{T}}
    @boundscheck begin
      @assert length(r) == d+1
      length(cores) == d || throw(DimensionMismatch("Trying to form $d-dimensional tensor train with only $(length(cores)) cores"))
      0≤Nup≤d && 0≤Ndn≤d || throw(DimensionMismatch("Total number of spin-up electrons $Nup, and spin-down $Ndn cannot be larger than dimension $d"))
    end
    
    for k=1:d
      @assert cores[k].k == k
      @assert row_ranks(cores[k]) == r[k]
      @assert col_ranks(cores[k]) == r[k+1]
    end
    return new{T,Nup,Ndn,d,S}(r,cores,false,0)
  end
end

function check(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  # Check rank consistency
  for k=1:d
    for (nup,ndn) in row_qn(core(tt,k))
      if row_rank(core(tt,k),nup,ndn) !== tt.r[k][nup,ndn]
        @warn "Row ranks at $((nup,ndn)) for $(k)-th block are inconsistent between core and tensor train: $(core(tt,k).row_ranks[nup,ndn]) instead of $(tt.r[k][nup,ndn])"
      end
    end
    for (nup,ndn) in col_qn(core(tt,k))
      if col_rank(core(tt,k),nup,ndn) !== tt.r[k+1][nup,ndn]
        @warn "Column ranks at $((nup,ndn)) for $(k)-th block are inconsistent between core and tensor train: $(core(tt,k).col_ranks[nup,ndn]) instead of $(tt.r[k+1][nup,ndn])"
      end
    end

    for (lup,ldn) in row_qn(core(tt,k)), (rup,rdn) in [r for r in ((lup,ldn), (lup+1,ldn), (lup,ldn+1), (lup+1,ldn+1)) if r in col_qn(core(tt,k))]
      if size(core(tt,k)[(lup,ldn),(rup,rdn)]) !== (row_rank(core(tt,k),lup,ldn), col_rank(core(tt,k),rup,rdn))
        @warn "Wrong $(k)-th core ($((lup,ldn)),$((rup,rdn))) block size: $(size(core(tt,k)[(lup,ldn),(rup,rdn)])) instead of ($(row_rank(core(tt,k),lup,ldn)), $(col_rank(core(tt,k),rup,rdn)))"
      end
    end
  end
end

function Base.show(io::IO, ::MIME"text/plain", tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  if get(io, :compact, true)
    str = "TTvector{$T,$Nup,$Ndn,$d}. Maximum rank $(maximum(sum.(tt.r)))"
  else
    # Manage some formatting and padding
    strr = ["r[i]=$(sum(r))" for r in tt.r]
    len = max(length.(strr)...)
    padr = len .- length.(strr)
    strr = ["$(s)"*" "^(pad+len+3)          for (s,pad) in zip(strr, padr)]

    str = string("TTvector{$T,$Nup,$Ndn,$d}. Ranks are:\n", strr...)
  end
    print(io, str)
end

function Base.show(io::IO, tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
    str = "TTvector{$T,$Nup,$Ndn,$d}. Maximum rank $(maximum(sum.(tt.r)))"
    println(io, str)
    for k=1:d
      println(io)
      show(io, core(tt,k))
    end
end

function Base.summary(io::IO, tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  show(summary(tt))
end

function Base.summary(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  if d>0
    return string("$(d)-dimensional TTvector object with eltype $T, ranks $(tt.r)")
  else
    return string("0-dimensional tensor train object with eltype $T")
  end
end

"""
    tt = tt_zeros(d::Int, N::Int, Sz::Rational, [T=Float64])

Compute the d-dimensional sTT-tensor of all zeros.
"""
function tt_zeros(d::Int, N::Int, Sz::Rational, ::Type{T}=Float64) where T<:Number
  Nup = Int(N+2Sz)÷2
  Ndn = Int(N-2Sz)÷2
  r = [zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  r[  1][state_qn(Nup,Ndn,d,  1)[1]...] = 1
  r[d+1][state_qn(Nup,Ndn,d,d+1)[1]...] = 1
  cores = [SparseCore{T,Nup,Ndn,d}(k, r[k], r[k+1]) for k=1:d]

  tt = TTvector(r, cores)
  check(tt)

  return tt
end
function tt_zeros(::Val{d}, ::Val{Nup}, ::Val{Ndn}, ::Type{T}=Float64) where {T<:Number,Nup,Ndn,d}
  r = [zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  r[  1][state_qn(Nup,Ndn,d,  1)[1]...] = 1
  r[d+1][state_qn(Nup,Ndn,d,d+1)[1]...] = 1
  cores = [SparseCore{T,Nup,Ndn,d}(k, r[k], r[k+1]) for k=1:d]

  tt = TTvector(r, cores)
  check(tt)

  return tt
end

function tt_zeros(::Val{d}, ::Val{Nup}, ::Val{Ndn}, r::Vector{Matrix{Int}}, ::Type{T}=Float64) where {T<:Number,Nup,Ndn,d}
  @boundscheck (length(r) == d+1) || (length(r) == d-1)

  if length(r) == d+1
    for k=1:d+1
      @boundscheck size(r[k]) == (Nup+1,Ndn+1) && findall(>(0), r[k]) ⊆ state_qn(Nup,Ndn,d,k)
    end
    cores = [SparseCore{T,Nup,Ndn,d}(k, r[k], r[k+1]) for k=1:d]

  else # length(r) == d-1
    for k=1:d-1
      @boundscheck size(r[k]) == (Nup+1,Ndn+1) && findall(>(0), r[k]) ⊆ state_qn(Nup,Ndn,d,k+1)
    end
    r₀ = zeros(Int,Nup+1,Ndn+1); r₀[state_qn(Nup,Ndn,d,  1)[1]...] = 1
    rₑ = zeros(Int,Nup+1,Ndn+1); rₑ[state_qn(Nup,Ndn,d,d+1)[1]...] = 1
    cores = [SparseCore{T,Nup,Ndn,d}(k, k>1 ? r[k-1] : r₀, k<d ? r[k] : rₑ) for k=1:d]
  end
  tt = TTvector(r, cores)
  check(tt)

  return(tt)
end

"""
    tt = tt_ones(d::Int, N::Int, Sz::Rational, [T=Float64])

Compute the d-dimensional sTT-tensor of all zeros.
"""
function tt_ones(d::Int, N::Int, Sz::Rational, ::Type{T}=Float64) where T<:Number
  Nup = Int(N+2Sz)÷2
  Ndn = Int(N-2Sz)÷2
  r = [zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for k=1:d+1, (nup,ndn) in state_qn(Nup,Ndn,d,k)
    r[k][nup,ndn] = 1
  end
  cores = [SparseCore{T,Nup,Ndn,d}(k, r[k], r[k+1]) for k=1:d]

  for k=1:d, (nup,ndn) in row_qn(cores[k])
    (nup  ,ndn  ) in col_qn(cores[k]) && fill!(○○(cores[k],nup,ndn), 1)
    (nup+1,ndn  ) in col_qn(cores[k]) && fill!(up(cores[k],nup,ndn), 1)
    (nup  ,ndn+1) in col_qn(cores[k]) && fill!(dn(cores[k],nup,ndn), 1)
    (nup+1,ndn+1) in col_qn(cores[k]) && fill!(●●(cores[k],nup,ndn), 1)
  end

  tt = TTvector(r, cores)
  check(tt)

  return tt
end

function tt_ones(::Val{d}, ::Val{Nup}, ::Val{Ndn}, ::Type{T}=Float64) where {T<:Number,Nup,Ndn,d}
  r = [zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for k=1:d+1, (nup,ndn) in state_qn(Nup,Ndn,d,k)
    r[k][nup,ndn] = 1
  end
  cores = [SparseCore{T,Nup,Ndn,d}(k, r[k], r[k+1]) for k=1:d]

  for k=1:d, (nup,ndn) in row_qn(cores[k])
    (nup  ,ndn  ) in col_qn(cores[k]) && fill!(○○(cores[k],nup,ndn), 1)
    (nup+1,ndn  ) in col_qn(cores[k]) && fill!(up(cores[k],nup,ndn), 1)
    (nup  ,ndn+1) in col_qn(cores[k]) && fill!(dn(cores[k],nup,ndn), 1)
    (nup+1,ndn+1) in col_qn(cores[k]) && fill!(●●(cores[k],nup,ndn), 1)
  end

  tt = TTvector(r, cores)
  check(tt)

  return tt
end



function tt_state(state_up::NTuple{Nup, Int}, state_dn::NTuple{Ndn, Int}, ::Val{d}, ::Type{T}=Float64) where {T<:Number,Nup,Ndn,d}
  @assert all(1 .≤ state_up .≤ d) && all(1 .≤ state_dn .≤ d)
  @assert allunique(state_up) && allunique(state_dn)
  return tt_state([k∈state_up for k=1:d], [k∈state_dn for k=1:d], T)
end

function tt_state(state_up::Vector{Bool}, state_dn::Vector{Bool}, ::Type{T}=Float64) where T<:Number
  d = length(state_up)
  @boundscheck @assert length(state_dn) == d
  Nup = sum(state_up)
  Ndn = sum(state_dn)

  nup = [0;cumsum(state_up)]
  ndn = [0;cumsum(state_dn)]
  cores = [SparseCore{T,Nup,Ndn,d}(state_up[k], nup[k], state_dn[k], ndn[k], k) for k=1:d]
  r = [[deepcopy(cores[k].row_ranks) for k=1:d];[deepcopy(cores[d].col_ranks)]]

  for k=1:d-1
    @assert r[k+1] == col_ranks(cores[k])
  end

  tt = TTvector(r, cores)
  check(tt)

  return tt
end

"""
    add_non_essential_dims(tt::TTvector{T,Nup,Ndn,d}, new_n::NTuple{newd,Int}, old_vars_pos::NTuple{d,Int})

Create `enlarged_tt` object with new 'dummy' modes. Mode sizes of `enlarged_tt` are equal to `new_n`, and the old modal indices positions should be given in the sorted list `old_vars_pos`.
"""
function add_non_essential_dims(tt::TTvector{T,Nup,Ndn,d,Matrix{T}}, newd::Int, old_vars_pos::NTuple{d,Int}) where {T<:Number,Nup,Ndn,d}
  @assert issorted(old_vars_pos)
  @assert length(old_vars_pos) == d
  @assert newd ≥ d

  r = [zeros(Int,Nup+1,Ndn+1) for k=1:newd+1]
  cores = Vector{SparseCore{T,Nup,Ndn,newd,Matrix{T}}}(undef, newd)

  # New dimensions corresponding to real ones
  for (k, kk) = enumerate(old_vars_pos)
    Ck = core(tt, k)

    # Compute new tensor ranks
    r[kk  ] .= rank(tt,k)
    r[kk+1] .= rank(tt,k+1)

    # Reconcile tensor ranks with cores
    cores[kk] = SparseCore{T,Nup,Ndn,newd}(kk, r[kk], r[kk+1])

    for (lup,ldn) in row_qn(core(tt,k)), (rup,rdn) in ((lup,ldn),(lup+1,ldn),(lup,ldn+1),(lup+1,ldn+1))∩col_qn(core(tt,k))
      if in_col_qn(rup,rdn,core(tt,k))
        cores[kk][(lup,ldn),(rup,rdn)] = core(tt,k)[(lup,ldn),(rup,rdn)]
      end
    end
  end

  # New dimensions before the first real dimension
  for kk in 1:old_vars_pos[1]-1
    # Compute new tensor ranks
    r[kk  ] .= rank(tt,1)
    r[kk+1] .= rank(tt,1)

    # Reconcile tensor ranks with cores
    l = (1,1)
    cores[kk] = SparseCore{T,Nup,Ndn,newd}(kk, rank(tt,1), rank(tt,1))
    copyto!(cores[kk][l,l], I(rank(tt,1,l)))
  end

  # New dimensions between real ones
  for k=2:d
    for kk=old_vars_pos[k-1]+1:old_vars_pos[k]-1
      # Compute new tensor ranks
      r[kk  ] .= rank(tt,k)
      r[kk+1] .= rank(tt,k)

      # Reconcile tensor ranks with cores
      cores[kk] = SparseCore{T,Nup,Ndn,newd}(kk, rank(tt,k), rank(tt,k))

      # Add appropriate cores
      for l in row_qn(core(tt,k))
        copyto!(cores[kk][l,l], I(rank(tt,k,l)))
      end
    end
  end

  # New dimensions after the last real dimension
  for kk in old_vars_pos[d]+1:newd
    # Compute new tensor ranks
    r[kk  ] .= rank(tt,d+1)
    r[kk+1] .= rank(tt,d+1)

    # Reconcile tensor ranks with cores
    cores[kk] = SparseCore{T,Nup,Ndn,newd}(kk, rank(tt,d+1), rank(tt,d+1))

    # Add appropriate cores
    l = (1+Nup,1+Ndn)
    copyto!(cores[kk][l,l], I(rank(tt,d+1,l)))
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
function cores2tensor(cores::Vector{SparseCore{T,Nup,Ndn,d,S}}) where {T<:Number,Nup,Ndn,d,S<:AbstractMatrix{T}}
  @assert d == length(cores)
  for k=1:d
    @assert cores[k].k == k
  end

  r = [[deepcopy(cores[k].row_ranks) for k=1:d]..., deepcopy(cores[d].col_ranks)]

  for k=2:d
    @assert cores[k-1].col_ranks == cores[k].row_ranks
  end

  tt = TTvector(r, copy(cores))
  check(tt)

  return tt
end

"""
    tensor2cores(tt::TTvector{T,Nup,Ndn,d})

Create a view on the list of SparseCores from `tt`.
"""
function tensor2cores(tt::TTvector)
  return tt.cores
end


function core(tt::TTvector, k::Int)
  return tt.cores[k]
end

function set_core!!(tt::TTvector{T,Nup,Ndn,d}, new_core::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  tt.cores[new_core.k] = new_core
  return tt
end

function set_cores!(tt::TTvector{T,Nup,Ndn,d,S}, first_core::SparseCore{T,Nup,Ndn,d,S}, second_core::SparseCore{T,Nup,Ndn,d,S}) where {T<:Number,Nup,Ndn,d,S<:AbstractMatrix{T}}
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
    ndims(tt::TTvector{T,Nup,Ndn,d})

Compute the number of dimensions of `tt`.
"""
function Base.ndims(::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return d
end

"""
    size(tt::TTvector{T,Nup,Ndn,d}, [dim])

Return a tuple containing the dimensions of `tt`. Optionally you can specify a dimension `dim` to just get the length of that dimension.
"""
function Base.size(::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return ntuple(k->2, d)
end

function Base.size(::TTvector{T,Nup,Ndn,d}, dim::Int) where {T<:Number,Nup,Ndn,d}
  return 4
end

"""
    Base.length(tt::TTvector{T,Nup,Ndn,d})

Compute the number of elements in `tt`, viewed as a `d`-dimensional Array.
"""
function Base.length(::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return 4^d
end


"""
    rank(tt::TTvector{T,Nup,Ndn,d}, [ind::Int], [n::Int])

Return a tuple containing the TT-ranks of `tt`. Optionally you can specify an index `ind` to just get the rank of that edge, and additionally the block index `n`.
"""
function LinearAlgebra.rank(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return tt.r
end

function LinearAlgebra.rank(tt::TTvector{T,Nup,Ndn,d}, k::Int) where {T<:Number,Nup,Ndn,d}
  @boundscheck 1 ≤ k ≤ d+1 || throw(BoundsError(tt, k))
  return tt.r[k]
end

function LinearAlgebra.rank(tt::TTvector{T,Nup,Ndn,d}, k::Int, n::Int) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    1 ≤ k ≤ d+1                     || throw(BoundsError(tt, k))
    if k ≤ d
      n in axes(row_qn(core(tt,k)),1) || throw(BoundsError(row_qn(core(tt,k)), n))
    else
      n in axes(col_qn(core(tt,d)),1) || throw(BoundsError(col_qn(core(tt,d)), n))
    end
  end
  return k ≤ d ? rank(tt,k)[row_qn(core(tt,k))[n]...] : rank(tt,k)[col_qn(core(tt,d))[n]...]
end

function LinearAlgebra.rank(tt::TTvector{T,Nup,Ndn,d}, k::Int, qn::Tuple{Int,Int}) where {T<:Number,Nup,Ndn,d}
  nup,ndn = qn
  @boundscheck begin
    1 ≤ k ≤ d+1 || throw(BoundsError(tt, k))
    qn ∈ state_qn(Nup,Ndn,d,k) || throw(BoundsError(tt.r[k], qn))
    if k ≤ d
      rank(tt,k)[nup,ndn] == core(tt,k).row_ranks[nup,ndn] || throw(DimensionMismatch("Tensor ranks $(tt.r[k]) do not match $k-th core row ranks $(core(tt,k).row_ranks)"))
    end
    if k ≥ 2
      rank(tt,k)[nup,ndn] == core(tt,k-1).col_ranks[nup,ndn] || throw(DimensionMismatch("Tensor ranks $(tt.r[k]) do not match $(k-1)-th core column ranks $(core(tt,k-1).col_ranks)"))
    end
  end
  return rank(tt,k)[nup,ndn]
end

function IdFrame(k::Int, tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  blocks = Matrix{Matrix{T}}(undef, Nup+1,Ndn+1)
  for (nup,ndn) in state_qn(Nup,Ndn,d,k)
    blocks[nup,ndn] = Matrix{T}(I(rank(tt,k,n)))
  end
  return Frame{T,Nup,Ndn,d}(k,blocks)
end

function IdFrame(::Val{d}, ::Val{Nup}, ::Val{Ndn}, k::Int, r::Int=1, ::Type{T}=Float64) where {T<:Number,Nup,Ndn,d}
  blocks = Matrix{Matrix{T}}(undef, Nup+1,Ndn+1)
  for (nup,ndn) in state_qn(Nup,Ndn,d,k)
    blocks[nup,ndn] = Matrix{T}(I(r))
  end
  return Frame{T,Nup,Ndn,d}(k, blocks)
end

"""
    Array(tt::TTvector{T,Nup,Ndn,d}, [sizes::NTuple{d2,Int}])

Return a dense Array representing the same object as `tt`. Optionally you can specify `sizes` to reshape the returned Array.
"""
function Base.Array(tt::TTvector{T,Nup,Ndn,d}, sizes::NTuple{d2,Int}) where {T<:Number,Nup,Ndn,d,d2}
  @assert prod(sizes) == (4^d)*rank(tt,1,1)*rank(tt,d+1,1)

  a = Matrix{Matrix{T}}(undef,Nup+1,Ndn+1)
  a[1,1] = Matrix{T}(I(rank(tt,1,(1,1))))
  for k=1:d
    Ak = core(tt,k)
    for rup in reverse(1:Nup+1), rdn in reverse(1:Ndn+1)
      if (rup,rdn) in col_qn(Ak)
        if rank(tt,k+1,(rup,rdn)) > 0
          if (rup  ,rdn  ) in row_qn(Ak)
            a[rup,rdn] = reshape( reshape(a[rup  ,rdn  ] * ○○(Ak,rup  ,rdn  ), (:,1,rank(tt,k+1,(rup,rdn)))) .* reshape([1,0,0,0], (1,4,1)), (:,rank(tt,k+1,(rup,rdn))))
          else
            a[rup,rdn] = zeros(T,rank(tt,1,1)*4^(k-1)*4,rank(tt,k+1,(rup,rdn)))
          end
          (rup-1,rdn  ) in row_qn(Ak) && (a[rup,rdn] .+= reshape( reshape(a[rup-1,rdn  ] * up(Ak,rup-1,rdn  ), (:,1,rank(tt,k+1,(rup,rdn)))) .* reshape([0,1,0,0], (1,4,1)), (:,rank(tt,k+1,(rup,rdn)))))
          (rup  ,rdn-1) in row_qn(Ak) && (a[rup,rdn] .+= reshape( reshape(a[rup  ,rdn-1] * dn(Ak,rup  ,rdn-1), (:,1,rank(tt,k+1,(rup,rdn)))) .* reshape([0,0,1,0], (1,4,1)), (:,rank(tt,k+1,(rup,rdn)))))
          (rup-1,rdn-1) in row_qn(Ak) && (a[rup,rdn] .+= reshape( reshape(a[rup-1,rdn-1] * ●●(Ak,rup-1,rdn-1), (:,1,rank(tt,k+1,(rup,rdn)))) .* reshape([0,0,0,1], (1,4,1)), (:,rank(tt,k+1,(rup,rdn)))))
        else
          a[rup,rdn] = zeros(T,rank(tt,1,1)*4^(k-1)*4,0)
        end
      end
    end
  end
  return reshape(a[Nup+1,Ndn+1], sizes)
end

function Base.Array(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  if (rank(tt,1,1) == 1 && rank(tt,d+1,1) == 1)
    sizes = ntuple(k->4, d)
  elseif (rank(tt,1,1) == 1 && rank(tt,d+1,1) != 1)
    sizes = ntuple(k->(k == d+1 ? rank(tt,d+1,1) : 4), d+1)
  elseif (rank(tt,1,1) != 1 && rank(tt,d+1,1) == 1)
    sizes = ntuple(k->(k == 1 ? rank(tt,1,1) : 4), d+1)
  else
    sizes = ntuple(k->(k==1 ? rank(tt,1,1) : k==d+2 ? rank(tt,d+1,1) : 4), d+2)
  end
  return Array(tt, sizes)
end
