using LinearAlgebra, OffsetArrays

########################################################
### Implement two-body second quantization operators ###
########################################################

##############################
### View operators ###
##############################

"""
  Adag_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements second quantization creation operator :math: (a_k^*) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function Adag_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  # Ensure that there is room for `nl` electrons to the left of core `A`
  # as well as `nr-1` electrons to the right of core `A` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux  , nl  , nr  , N)
  qr = shift_qn(A.col_qn, flux+1, nl+1, nr-1, N)
  qn = axes(A.occupied,1)∩ql∩(qr.-1)
  B = UnsafeSparseCore{T,N,d,M}(occupied = (qn, A.unoccupied[qn.-flux]))

  return B, flux+1, nl+1, nr-1
end

"""
  A_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements second quantization annihilation operator :math: (a_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function A_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux  , nl, nr, N)
  qr = shift_qn(A.col_qn, flux-1, nl, nr, N)
  qn = axes(A.unoccupied,1)∩ql∩qr
  B = UnsafeSparseCore{T,N,d,M}(unoccupied = (qn, A.occupied[qn.-flux]))

  return B, flux-1, nl, nr
end

"""
  AdagA_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements annihilation/creation block :math: (a^*_ka_k) on core `k`.
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function AdagA_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr-1` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux, nl  , nr  , N)
  qr = shift_qn(A.col_qn, flux, nl+1, nr-1, N)
  qn = axes(A.occupied,1)∩ql∩(qr.-1)
  B = UnsafeSparseCore{T,N,d,M}(occupied = (qn, A.occupied[qn.-flux]))

  return B, flux, nl+1, nr-1
end

"""
  S_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements Jordan-Wigner component :math: (s_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and odd, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function S_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))
  @boundscheck @assert isodd(flux)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux, nl, nr, N)
  qr = shift_qn(A.col_qn, flux, nl, nr, N)
  qn_unocc = axes(A.unoccupied,1)∩ql∩ qr
  qn_occ   = axes(A.occupied,1)  ∩ql∩(qr.-1)
  B = UnsafeSparseCore{T,N,d,M}( unoccupied = (qn_unocc, A.unoccupied[qn_unocc.-flux]),
                               occupied   = (qn_occ,   A.occupied[    qn_occ.-flux]), 
                               jw=true)

  return B, flux, nl, nr
end

"""
  Id_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements Identity component :math: (i_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and even, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function Id_view(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))
  @assert iseven(flux)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux, nl, nr, N)
  qr = shift_qn(A.col_qn, flux, nl, nr, N)
  qn_unocc = axes(A.unoccupied,1)∩ql∩ qr
  qn_occ   = axes(A.occupied,1)  ∩ql∩(qr.-1)
  B = UnsafeSparseCore{T,N,d,M}( unoccupied = (qn_unocc, A.unoccupied[qn_unocc.-flux]),
                               occupied   = (qn_occ,   A.occupied[    qn_occ.-flux])
                             )
  return B, flux, nl, nr
end

function AdagᵢAⱼ_view(tt_in::TTvector{T,N,d,M}, i::Int, j::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck 1 ≤ i ≤ d && 1 ≤ j ≤ d

  flux = 0
  nl = 0
  nr = 1

  cores = Vector{SparseCore{T,N,d,M}}(undef, d)
  ranks = deepcopy(rank(tt_in))

  shift_ranks!(ranks[1], rank(tt_in, 1), flux, nl, nr, N)

  for n=1:d
    if n == i && n == j
      unsafe_core, flux, nl, nr = AdagA_view(core(tt_in,n), flux, nl, nr)
    elseif n == i # Creation operator
      unsafe_core, flux, nl, nr = Adag_view( core(tt_in,n), flux, nl, nr)
    elseif n == j # Annihilation operator
      unsafe_core, flux, nl, nr = A_view(    core(tt_in,n), flux, nl, nr)
    elseif isodd(flux)
      unsafe_core, flux, nl, nr = S_view(    core(tt_in,n), flux, nl, nr)
    else # if iseven(flux)
      unsafe_core, flux, nl, nr = Id_view(   core(tt_in,n), flux, nl, nr)
    end
    # Adjust row ranks using flux to determine shift
    shift_ranks!(ranks[n+1], rank(tt_in, n+1), flux, nl, nr, N)
    # Reconstruct a SparseCore from the unsafe view obtained above
    cores[n] = SparseCore(n,ranks[n],ranks[n+1],unsafe_core)
  end

  tt_out = TTvector(ranks, cores)

  # Sanity check for the ranks
  check(tt_out)
  return tt_out
end


function AdagᵢAdagⱼAₖAₗ_view(tt_in::TTvector{T,N,d,M}, i::Int, j::Int, k::Int, l::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck 1 ≤ i < j ≤ d && 1 ≤ k < l ≤ d

  flux = 0
  nl = 0
  nr = 2

  cores = Vector{SparseCore{T,N,d,M}}(undef, d)
  ranks = deepcopy(rank(tt_in))

  shift_ranks!(ranks[1], rank(tt_in, 1), flux, nl, nr, N)

  for n=1:d
    if n ∈ (i,j)∩(k,l)
      unsafe_core, flux, nl, nr = AdagA_view(core(tt_in,n), flux, nl, nr)
    elseif n ∈ (i,j) # Creation operator
      unsafe_core, flux, nl, nr = Adag_view( core(tt_in,n), flux, nl, nr)
    elseif n ∈ (k,l) # Annihilation operator
      unsafe_core, flux, nl, nr = A_view(    core(tt_in,n), flux, nl, nr)
    elseif isodd(flux)
      unsafe_core, flux, nl, nr = S_view(    core(tt_in,n), flux, nl, nr)
    else # if iseven(flux)
      unsafe_core, flux, nl, nr = Id_view(   core(tt_in,n), flux, nl, nr)
    end

    shift_ranks!(ranks[n+1], rank(tt_in, n+1), flux, nl, nr, N)
    cores[n] = SparseCore(n,ranks[n],ranks[n+1],unsafe_core)
  end

  tt_out = TTvector(ranks, cores)
  # Sanity check for the ranks
  check(tt_out)

  return tt_out
end

##############################
### Out-of-place operators ###
##############################

"""
  Adag(A, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements second quantization creation operator :math: (a_k^*) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function Adag(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr-1` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns

  ql, row_ranks = shift_ranks(A.row_ranks, flux, nl, nr, N)
  qr, col_ranks = shift_ranks(A.col_ranks, flux+1, nl+1, nr-1, N)
  B = SparseCore{T,N,d}(k,row_ranks,col_ranks)

  for l in axes(A,1) ∩ (axes(A,3).-1) ∩ ql ∩ (qr.-1)
    B[l,2,l+1] = A[l-flux,1,l-flux]
  end

  return B, flux+1, nl+1, nr-1
end

"""
  A(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements second quantization annihilation operator :math: (a_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function A(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns

  ql, row_ranks = shift_ranks(A.row_ranks, flux,   nl, nr, N)
  qr, col_ranks = shift_ranks(A.col_ranks, flux-1, nl, nr, N)
  B = SparseCore{T,N,d}(k,row_ranks,col_ranks)

  for l in axes(B,1) ∩ axes(B,3) ∩ ql ∩ qr
    B[l,1,l] = A[l-flux,2,l-flux+1]
  end

  return B, flux-1, nl, nr
end

"""
  AdagA(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements annihilation/creation block :math: (a^*_ka_k) on core `k`.
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function AdagA(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr-1` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql, row_ranks = shift_ranks(A.row_ranks, flux, nl, nr, N)
  qr, col_ranks = shift_ranks(A.col_ranks, flux, nl+1, nr-1, N)
  B = SparseCore{T,N,d}(k, row_ranks, col_ranks)

  for l in axes(B,1) ∩ (axes(B,3).-1) ∩ ql ∩ (qr.-1)
    B[l,2,l+1] = A[l-flux,2,l-flux+1]
  end

  return B, flux, nl+1, nr-1
end

"""
  S(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements Jordan-Wigner component :math: (s_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and odd, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function S(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))
  @assert isodd(flux)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql, row_ranks = shift_ranks(A.row_ranks, flux, nl, nr, N)
  qr, col_ranks = shift_ranks(A.col_ranks, flux, nl, nr, N)
  B = SparseCore{T,N,d}(k, row_ranks, col_ranks)

  for l in axes(B,1) ∩ axes(B,3) ∩ ql ∩ qr
    B[l,1,l] = A[l-flux,1,l-flux]
  end
  for l ∈ axes(B,1) ∩ (axes(B,3).-1) ∩ ql ∩ (qr.-1)
    B[l,2,l+1] = A[l-flux,2,l-flux+1]
    lmul!(-1, B[l,2,l+1])
  end
  return B, flux, nl, nr
end

"""
  Id(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}

Implements Identity component :math: (i_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and even, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function Id(A::SparseCore{T,N,d,M}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))
  @assert iseven(flux)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql, row_ranks = shift_ranks(A.row_ranks, flux, nl, nr, N)
  qr, col_ranks = shift_ranks(A.col_ranks, flux, nl, nr, N)
  B = SparseCore{T,N,d}(k, row_ranks, col_ranks)

  for l in axes(B,1) ∩ axes(B,3) ∩ ql ∩ qr
    B[l,1,l] = A[l-flux,1,l-flux]
  end
  for l ∈ axes(B,1) ∩ (axes(B,3).-1) ∩ ql ∩ (qr.-1)
    B[l,2,l+1] = A[l-flux,2,l-flux+1]
  end

  return B, flux, nl, nr
end


function AdagᵢAⱼ(tt_in::TTvector{T,N,d,M}, i::Int, j::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck 1 ≤ i ≤ d && 1 ≤ j ≤ d

  flux = 0
  nl = 0
  nr = 1

  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef, d)
  ranks = deepcopy(rank(tt_in))

  shift_ranks!(ranks[1], rank(tt_in, 1), flux, nl, nr, N)
  for n=1:d
    if n == i && n == j
      cores[n], flux, nl, nr = AdagA(core(tt_in,n), flux, nl, nr)
    elseif n == i # Creation operator
      cores[n], flux, nl, nr = Adag( core(tt_in,n), flux, nl, nr)
    elseif n == j # Annihilation operator
      cores[n], flux, nl, nr = A(    core(tt_in,n), flux, nl, nr)
    elseif isodd(flux)
      cores[n], flux, nl, nr = S(    core(tt_in,n), flux, nl, nr)
    else # if iseven(flux)
      cores[n], flux, nl, nr = Id(   core(tt_in,n), flux, nl, nr)
    end
    # Adjust row ranks using flux to determine shift
    shift_ranks!(ranks[n+1], rank(tt_in, n+1), flux, nl, nr, N)
  end

  tt_out = TTvector(ranks, cores)
  # Sanity check for the ranks
  check(tt_out)
  return tt_out
end


function AdagᵢAdagⱼAₖAₗ(tt_in::TTvector{T,N,d,M}, i::Int, j::Int, k::Int, l::Int) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck 1 ≤ i < j ≤ d && 1 ≤ k < l ≤ d

  flux = 0
  nl = 0
  nr = 2

  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef, d)
  ranks = deepcopy(rank(tt_in))

  for n=1:d
    shift_ranks!(ranks[n], rank(tt_in, n), flux, nl, nr, N)

    if n ∈ (i,j)∩(k,l)
      cores[n], flux, nl, nr = AdagA(core(tt_in,n), flux, nl, nr)
    elseif n ∈ (i,j) # Creation operator
      cores[n], flux, nl, nr = Adag( core(tt_in,n), flux, nl, nr)
    elseif n ∈ (k,l) # Annihilation operator
      cores[n], flux, nl, nr = A(    core(tt_in,n), flux, nl, nr)
    elseif isodd(flux)
      cores[n], flux, nl, nr = S(    core(tt_in,n), flux, nl, nr)
    else # if iseven(flux)
      cores[n], flux, nl, nr = Id(   core(tt_in,n), flux, nl, nr)
    end
  end

  shift_ranks!(ranks[d+1], rank(tt_in, d+1), flux, nl, nr, N)

  tt_out = TTvector(ranks, cores)
  # Sanity check for the ranks
  check(tt_out)

  return tt_out
end

# Convenience function
function shift_ranks!(ranks::AbstractVector{Int}, flux::Int, nl::Int, nr::Int, N::Int)
  @boundscheck @assert length(ranks) ≥ flux 
  @boundscheck @assert nl ≥ 0 && nr ≥ 0 && nl + nr ≤ N

  start = min(max(nl,   firstindex(ranks)+(flux>0 ? flux : 0)), lastindex(ranks )+1)
  stop  = max(min(N-nr, lastindex( ranks)+(flux<0 ? flux : 0)), firstindex(ranks)-1)
  qn = start:stop

  ranks[qn] = ranks[qn.-flux]
  ranks[begin:start-1]  .= 0
  ranks[stop+1:end] .= 0

  return qn
end

function shift_ranks!(new_ranks::AbstractVector{Int}, ranks::AbstractVector{Int}, 
                      flux::Int, nl::Int, nr::Int, N::Int)
  @boundscheck @assert nl ≥ 0 && nr ≥ 0 && nl + nr ≤ N

  start = min(max(nl,   firstindex(ranks)+(flux>0 ? flux : 0)), lastindex(ranks )+1)
  stop  = max(min(N-nr, lastindex( ranks)+(flux<0 ? flux : 0)), firstindex(ranks)-1)
  qn = start:stop

  new_ranks[qn] = ranks[qn.-flux]
  new_ranks[begin:start-1]  .= 0
  new_ranks[stop+1:end] .= 0

  return qn
end

function shift_ranks( ranks::AbstractVector{Int}, 
                      flux::Int, nl::Int, nr::Int, N::Int)
  new_ranks = deepcopy(ranks)
  qn = shift_ranks!(new_ranks, ranks, flux, nl, nr, N)
  return qn, new_ranks
end
