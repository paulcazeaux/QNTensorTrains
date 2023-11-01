using LinearAlgebra, OffsetArrays

########################################################
### Implement two-body second quantization operators ###
########################################################

##############################
### View operators ###
##############################

"""
  Adag_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements second quantization creation operator :math: (a_k^*) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function Adag_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  # Ensure that there is room for `nl` electrons to the left of core `A`
  # as well as `nr-1` electrons to the right of core `A` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux  , nl  , nr  , N)
  qr = shift_qn(A.col_qn, flux+1, nl+1, nr-1, N)

  B = UnsafeSparseCore{T,N,d}(1:0, axes(A.occupied,1)∩ql∩(qr.-1))

  for l in axes(B.occupied,1)
    B[l,2,l+1] = A[l-flux,1,l-flux]
  end

  return B, flux+1, nl+1, nr-1
end

"""
  A_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements second quantization annihilation operator :math: (a_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function A_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux  , nl, nr, N)
  qr = shift_qn(A.col_qn, flux-1, nl, nr, N)

  B = UnsafeSparseCore{T,N,d}(axes(A.unoccupied,1)∩ql∩qr, 1:0)

  for l in axes(B.unoccupied,1)
    B[l,1,l] = A[l-flux,2,l-flux+1]
  end

  return B, flux-1, nl, nr
end

"""
  AdagA_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements annihilation/creation block :math: (a^*_ka_k) on core `k`.
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function AdagA_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr-1` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux, nl  , nr  , N)
  qr = shift_qn(A.col_qn, flux, nl+1, nr-1, N)

  B = UnsafeSparseCore{T,N,d}(1:0, axes(A.occupied,1)∩ql∩(qr.-1))

  for l in axes(B.occupied,1)
    B[l,2,l+1] = A[l-flux,2,l-flux+1]
  end

  return B, flux, nl+1, nr-1
end

"""
  S_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements Jordan-Wigner component :math: (s_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and odd, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function S_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))
  @boundscheck @assert isodd(flux)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux, nl, nr, N)
  qr = shift_qn(A.col_qn, flux, nl, nr, N)

  B = UnsafeSparseCore{T,N,d}(axes(A.unoccupied,1)∩ql∩qr, axes(A.occupied,1)∩ql∩(qr.-1))

  for l in axes(B.unoccupied,1)
    B[l,1,l] = A[l-flux,1,l-flux]
  end
  for l in axes(B.occupied,1)
    B[l,2,l+1] = copy(A[l-flux,2,l-flux+1])
    lmul!(-1, B[l,2,l+1])
  end

  return B, flux, nl, nr
end

"""
  Id_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements Identity component :math: (i_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and even, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function Id_view(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))
  @assert iseven(flux)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_qn(A.row_qn, flux, nl, nr, N)
  qr = shift_qn(A.col_qn, flux, nl, nr, N)

  B = UnsafeSparseCore{T,N,d}(axes(A.unoccupied,1)∩ql∩qr, axes(A.occupied,1)∩ql∩(qr.-1))

  for l in axes(B.unoccupied,1)
    B[l,1,l] = A[l-flux,1,l-flux]
  end
  for l ∈ axes(B.occupied,1)
    B[l,2,l+1] = A[l-flux,2,l-flux+1]
  end

  return B, flux, nl, nr
end

function AdagᵢAⱼ_view(tt_in::TTvector{T,N,d}, i::Int, j::Int) where {T<:Number,N,d}
  @boundscheck 1 ≤ i ≤ d && 1 ≤ j ≤ d

  flux = 0
  nl = 0
  nr = 1

  # ranks = deepcopy(rank(tt_in))
  cores = [SparseCore{T,N,d}(n) for n=1:d]
  ranks::Vector{OffsetVector{Int, Vector{Int}}} = [(n ≤ d ? cores[n].row_ranks : cores[d].col_ranks) for n=1:d+1]

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

  tt_out = TTvector{T,N,d}(ranks, cores)

  # Sanity check for the ranks
  check(tt_out)
  return tt_out
end


function AdagᵢAdagⱼAₖAₗ_view(tt_in::TTvector{T,N,d}, i::Int, j::Int, k::Int, l::Int) where {T<:Number,N,d}
  @boundscheck 1 ≤ i < j ≤ d && 1 ≤ k < l ≤ d

  flux = 0
  nl = 0
  nr = 2

  # ranks = deepcopy(rank(tt_in))
  cores = [SparseCore{T,N,d}(n) for n=1:d]
  ranks = [(n ≤ d ? cores[n].row_ranks : cores[d].col_ranks) for n=1:d+1]

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


  tt_out = TTvector{T,N,d}(ranks, cores)
  # Sanity check for the ranks
  check(tt_out)

  return tt_out
end

##############################
### Out-of-place operators ###
##############################

"""
  Adag(A, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements second quantization creation operator :math: (a_k^*) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function Adag(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  B = SparseCore{T,N,d}(k)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr-1` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns

  ql = shift_ranks!(B.row_ranks, A.row_ranks, flux, nl, nr, N)
  qr = shift_ranks!(B.col_ranks, A.col_ranks, flux+1, nl+1, nr-1, N)

  for l in axes(A,1) ∩ (axes(A,3).-1)
    if l∈ql && l+1∈qr
      B[l,2,l+1] = deepcopy(A[l-flux,1,l-flux])
    else
      B[l,2,l+1] = zeros_block(T,B.row_ranks[l],B.col_ranks[l+1])
    end
  end

  # Reshape unoccupied blocks
  for l in axes(A,1) ∩ axes(A,3)
    B[l,1,l] = zeros_block(T, B.row_ranks[l], B.col_ranks[l])
  end


  return B, flux+1, nl+1, nr-1
end

"""
  A(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements second quantization annihilation operator :math: (a_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function A(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  B = SparseCore{T,N,d}(k)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns

  ql = shift_ranks!(B.row_ranks, A.row_ranks, flux,   nl, nr, N)
  qr = shift_ranks!(B.col_ranks, A.col_ranks, flux-1, nl, nr, N)

  for l in axes(B,1) ∩ axes(B,3)
    if l ∈ ql ∩ qr
      B[l,1,l] = deepcopy(A[l-flux,2,l-flux+1])
    else
      B[l,1,l] = zeros_block(T, B.row_ranks[l], B.col_ranks[l])
    end
  end

  # Reshape occupied blocks
  for l in axes(A,1) ∩ (axes(A,3).-1)
    B[l,2,l+1] = zeros_block(T, B.row_ranks[l], B.col_ranks[l+1])
  end

  return B, flux-1, nl, nr
end

"""
  AdagA(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements annihilation/creation block :math: (a^*_ka_k) on core `k`.
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function AdagA(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  B = SparseCore{T,N,d}(k)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr-1` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_ranks!(B.row_ranks, A.row_ranks, flux, nl, nr, N)
  qr = shift_ranks!(B.col_ranks, A.col_ranks, flux, nl+1, nr-1, N)

  for l in axes(B,1) ∩ (axes(B,3).-1)
    if l∈ql && l+1∈qr
      B[l,2,l+1] = deepcopy(A[l-flux,2,l-flux+1])
    else
      B[l,2,l+1] = zeros_block(T, B.row_ranks[l], B.col_ranks[l+1])
    end
  end

  # Reshape unoccupied blocks
  for l in axes(B,1) ∩ axes(B,3)
    B[l,1,l] = zeros_block(T, B.row_ranks[l], B.col_ranks[l])
  end


  return B, flux, nl+1, nr-1
end

"""
  S(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements Jordan-Wigner component :math: (s_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and odd, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function S(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))
  @assert isodd(flux)

  B = SparseCore{T,N,d}(k)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_ranks!(B.row_ranks, A.row_ranks, flux, nl, nr, N)
  qr = shift_ranks!(B.col_ranks, A.col_ranks, flux, nl, nr, N)

  for l in axes(B,1) ∩ axes(B,3)
    if l ∈ ql ∩ qr
      B[l,1,l] = deepcopy(A[l-flux,1,l-flux])
    else
      B[l,1,l] = zeros_block(T,B.row_ranks[l],B.col_ranks[l])
    end
  end
  for l ∈ axes(B,1) ∩ (axes(B,3).-1)
    if l ∈ ql && l+1 ∈ qr
      B[l,2,l+1] = -1 * A[l-flux,2,l-flux+1]
    else
      B[l,2,l+1] = zeros_block(T,B.row_ranks[l],B.col_ranks[l+1])
    end
  end

  return B, flux, nl, nr
end

"""
  Id(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements Identity component :math: (i_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and even, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function Id(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))
  @assert iseven(flux)

  B = SparseCore{T,N,d}(k)

  # Ensure that there is room for `nl` electrons to the left of core `k`
  # as well as `nr` electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
  ql = shift_ranks!(B.row_ranks, A.row_ranks, flux, nl, nr, N)
  qr = shift_ranks!(B.col_ranks, A.col_ranks, flux, nl, nr, N)

  for l in axes(B,1) ∩ axes(B,3)
    if l ∈ ql ∩ qr
      B[l,1,l] = deepcopy(A[l-flux,1,l-flux])
    else
      B[l,1,l] = zeros_block(T,B.row_ranks[l],B.col_ranks[l])
    end
  end
  for l ∈ axes(B,1) ∩ (axes(B,3).-1)
    if l ∈ ql && l+1 ∈ qr
      B[l,2,l+1] = deepcopy(A[l-flux,2,l-flux+1])
    else
      B[l,2,l+1] = zeros_block(T,B.row_ranks[l],B.col_ranks[l+1])
    end
  end

  return B, flux, nl, nr
end


function AdagᵢAⱼ(tt_in::TTvector{T,N,d}, i::Int, j::Int) where {T<:Number,N,d}
  @boundscheck 1 ≤ i ≤ d && 1 ≤ j ≤ d

  flux = 0
  nl = 0
  nr = 1

  # ranks = deepcopy(rank(tt_in))
  cores = [SparseCore{T,N,d}(n) for n=1:d]
  ranks = [(n ≤ d ? cores[n].row_ranks : cores[d].col_ranks) for n=1:d+1]

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

  tt_out = TTvector{T,N,d}(ranks, cores)
  # Sanity check for the ranks
  check(tt_out)
  return tt_out
end


function AdagᵢAdagⱼAₖAₗ(tt_in::TTvector{T,N,d}, i::Int, j::Int, k::Int, l::Int) where {T<:Number,N,d}
  @boundscheck 1 ≤ i < j ≤ d && 1 ≤ k < l ≤ d

  flux = 0
  nl = 0
  nr = 2

  # ranks = deepcopy(rank(tt_in))
  cores = [SparseCore{T,N,d}(n) for n=1:d]
  ranks = [(n ≤ d ? cores[n].row_ranks : cores[d].col_ranks) for n=1:d+1]

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

  tt_out = TTvector{T,N,d}(ranks, cores)
  # Sanity check for the ranks
  check(tt_out)

  return tt_out
end

##########################
### In-place operators ###
##########################

"""
  Adag!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements second quantization creation operator :math: (a_k^*) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function Adag!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k

  # Shift ranks
  ql = shift_ranks!(A.row_ranks, flux,   nl,   nr,   N)
  qr = shift_ranks!(A.col_ranks, flux+1, nl+1, nr-1, N)

  # Move local unoccupied blocks to occupied
  for l in axes(A,1)∩(axes(A,3).-1)
    n = l-flux
    if l ∈ ql && l+1 ∈ qr
      A[l,2,l+1] = A[l-flux,1,l-flux]
    else
      A[l,2,l+1] = zeros_block(T,A.row_ranks[l],A.col_ranks[l+1])
    end
  end

  # Zero out all unoccupied blocks on this core
  for l in axes(A,1)∩axes(A,3)
    A[l,1,l] = zeros_block(T,A.row_ranks[l],A.col_ranks[l])
  end

  return flux+1, nl+1, nr-1
end

"""
  A!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements second quantization annihilation operator :math: (a_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function A!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k

  # Shift ranks
  ql = shift_ranks!(A.row_ranks, flux,   nl, nr, N)
  qr = shift_ranks!(A.col_ranks, flux-1, nl, nr, N)

  # Move local occupied blocks to unoccupied
  for l in axes(A,1) ∩ axes(A,3)
    if l ∈ ql && l ∈ qr
      A[l,1,l] = A[l-flux,2,l-flux+1]
    else
      A[l,1,l] = zeros_block(T,A.row_ranks[l],A.col_ranks[l])
    end
  end

  # Zero out all occupied blocks on this core
  for l in axes(A,1) ∩ (axes(A,3).-1)
    A[l,2,l+1] = zeros_block(T,A.row_ranks[l],A.col_ranks[l+1])
  end

  return flux-1, nl, nr
end

"""
  AdagA!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements annihilation/creation block :math: (a^*_ka_k) on core `k`.
assuming the quantum number flux in the chain up to core 'k' is `flux`, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k` (included).
"""
function AdagA!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  k = A.k
  
  # Shift ranks
  ql = shift_ranks!(A.row_ranks, flux, nl,   nr,   N)
  qr = shift_ranks!(A.col_ranks, flux, nl+1, nr-1, N)

  for l ∈ (flux > 0 ? reverse : identity)(axes(A,1) ∩ (axes(A,3).-1))
    if l∈ql && l+1∈qr
      A[l,2,l+1] = A[l-flux,2,l-flux+1]
    else
      A[l,2,l+1] = zeros_block(T,A.row_ranks[l],A.col_ranks[l+1])
    end
  end

  # Zero out and resize unoccupied blocks
  for l in axes(A,1) ∩ axes(A,3)
    A[l,1,l] = zeros_block(T,A.row_ranks[l], A.col_ranks[l])
  end

  return flux, nl+1, nr-1
end

"""
  S!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements Jordan-Wigner component :math: (s_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and odd, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function S!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  @boundscheck @assert isodd(flux)
  k = A.k

  # Shift ranks
  ql = shift_ranks!(A.row_ranks, flux, nl, nr, N)
  qr = shift_ranks!(A.col_ranks, flux, nl, nr, N)

  for l in (flux > 0 ? reverse : identity)(axes(A,1) ∩ axes(A,3))
    if l ∈ ql && l ∈ qr
    # Shift blocks diagonally
      A[l,1,l] = A[l-flux,1,l-flux]
    else
      A[l,1,l] = zeros_block(T, A.row_ranks[l], A.col_ranks[l])
    end
  end
  for l in (flux > 0 ? reverse : identity)(axes(A,1) ∩ (axes(A,3).-1))
    if l ∈ ql && l+1 ∈ qr
      A[l,2,l+1] = lmul!(-1, A[l-flux,2,l-flux+1])
    else
      A[l,2,l+1] = zeros_block(T, A.row_ranks[l], A.col_ranks[l+1])
    end
  end

  return flux, nl, nr
end

"""
  Id!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}

Implements Identity component :math: (i_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux` and even, 
and we must fit `nl` electrons to the left and `nr` electrons to the right of core `k`.
"""
function Id!(A::SparseCore{T,N,d}, flux::Int, nl::Int, nr::Int) where {T<:Number,N,d}
  @boundscheck @assert iseven(flux)
  k = A.k

  # Shift ranks
  ql = shift_ranks!(A.row_ranks, flux, nl, nr, N)
  qr = shift_ranks!(A.col_ranks, flux, nl, nr, N)

  for l in (flux > 0 ? reverse : identity)(axes(A,1) ∩ axes(A,3))
    if l ∈ ql && l ∈ qr
    # Shift blocks diagonally
      A[l,1,l] = A[l-flux,1,l-flux]
    else
      A[l,1,l] = zeros_block(T, A.row_ranks[l], A.col_ranks[l])
    end
  end
  for l in (flux > 0 ? reverse : identity)(axes(A,1) ∩ (axes(A,3).-1))
    if l ∈ ql && l+1 ∈ qr
      A[l,2,l+1] = A[l-flux,2,l-flux+1]
    else
      A[l,2,l+1] = zeros_block(T, A.row_ranks[l], A.col_ranks[l+1])
    end
  end

  return flux, nl, nr
end

function AdagᵢAⱼ!(tt::TTvector{T,N,d}, i::Int, j::Int) where {T<:Number,N,d}
  @boundscheck @assert 1 ≤ i ≤ d && 1 ≤ j ≤ d

  flux = 0
  nl = 0
  nr = 1

  for n=1:d
    # Adjust row ranks using flux to determine shift
    shift_ranks!(rank(tt,n), flux, nl, nr, N)

    if n == i == j
      flux, nl, nr = AdagA!(core(tt,n), flux, nl, nr)
    elseif n == i # Creation operator
      flux, nl, nr = Adag!( core(tt,n), flux, nl, nr)
    elseif n == j # Annihilation operator
      flux, nl, nr = A!(    core(tt,n), flux, nl, nr)
    elseif isodd(flux)
      flux, nl, nr = S!(    core(tt,n), flux, nl, nr)
    else # if iseven(it)
      flux, nl, nr = Id!(   core(tt,n), flux, nl, nr)
    end
  end

  # Sanity check for the ranks
  check(tt)
end

function AdagᵢAdagⱼAₖAₗ!(tt::TTvector{T,N,d}, i::Int, j::Int, k::Int, l::Int) where {T<:Number,N,d}
  @boundscheck @assert 1 ≤ i < j ≤ d && 1 ≤ k < l ≤ d

  flux = 0
  nl = 0
  nr = 2

  for n=1:d
    # Adjust row ranks using flux to determine shift
    shift_ranks!(rank(tt,n), flux, nl, nr, N)

    if n ∈ (i,j)∩(k,l)
      flux, nl, nr = AdagA!(core(tt,n), flux, nl, nr)
    elseif n ∈ (i,j) # Creation operator
      flux, nl, nr = Adag!( core(tt,n), flux, nl, nr)
    elseif n ∈ (k,l) # Annihilation operator
      flux, nl, nr = A!(    core(tt,n), flux, nl, nr)
    elseif isodd(flux)
      flux, nl, nr = S!(    core(tt,n), flux, nl, nr)
    else # if iseven(it)
      flux, nl, nr = Id!(   core(tt,n), flux, nl, nr)
    end
  end

  # Sanity check for the ranks
  check(tt)
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


########################################################
### Implement one-body second quantization operators ###
########################################################

module OneBody
using ..QNTensorTrains: SparseCore, TTvector, core, check, zeros_block
using LinearAlgebra

"""
  Adag_left!(A::SparseCore{T,N,d}) where {T<:Number,N,d}

Implements second quantization creation operator :math: (a_k^*) on core `k`,
assuming it comes *left* from the corresponding  annihilation operator in the 
number-conserving block :math: (a_k^* a_j), i.e. :math: (k < j).
"""
function Adag_left!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = A.k
  @boundscheck @assert k < d

  A.col_ranks[begin] = 0

# Case n ∉ axes(A,3) - the unoccupied block A[n,1,n] does not exist
  if first(axes(A,1)) < first(axes(A,3))
    n = first(axes(A,1))
    @boundscheck @assert n+1 == first(axes(A,3)) && d+1-k == N-n

    A.row_ranks[n  ] = 0
    A[n,2,n+1] = zeros_block(T,0,0)
  end


# Generic case : both occupied and unoccupied are allowed
  qn = first(axes(A,3)):(last(axes(A,3))-1)
  @boundscheck @assert qn == axes(A.unoccupied,1) ∩ axes(A.occupied,1)
  for n in qn
    A.col_ranks[n+1] = size(A[n,1,n],2)
    A[n,2,n+1] = A[n,1,n]
    A[n,1,n  ] = zeros_block(T,A.row_ranks[n],A.col_ranks[n])
  end

  n = last(axes(A,3))
  if n ∈ axes(A,1)
    A.row_ranks[n] = 0
    A[n,1,n] = zeros_block(T,0,A.col_ranks[n])
  end

  return A
end

"""
  Adag_right!(A::SparseCore{T,N,d}) where {T<:Number,N,d}

Implements second quantization creation operator :math: (a_k^*) on core `k`,
assuming it comes *right* from the corresponding annihilation operator in the 
number-conserving block :math: (a_k^* a_j), i.e. :math: (j < k).
"""
function Adag_right!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = A.k
  @boundscheck @assert k > 1

  A.row_ranks[end] = 0

# Case n ∉ axes(A,1), i.e. k==n - the unoccupied block A[n,1,n] does not exist
  n = last(axes(A,3))
  if n > last(axes(A,1))
    @boundscheck @assert n-1 == last(axes(A,1)) && n == k
    A.col_ranks[n] = 0
    A[n-1,2,n] = zeros_block(T,0,0)
  end

# Generic case : both occupied and unoccupied are allowed
  qn = (first(axes(A,1))+1):last(axes(A,1))
  @boundscheck @assert qn == axes(A.unoccupied,1) ∩ (axes(A.occupied,1).+1)
  for n in reverse(qn)
    A.row_ranks[n-1] = size(A[n,1,n],1)
    A[n-1,2,n] = A[n,1,n]
    A[n,1,n] = zeros_block(T,A.row_ranks[n],A.col_ranks[n])
  end

  n = first(axes(A,1))
  if n ∈ axes(A,3)
    A.col_ranks[n] = 0
    A[n,1,n] = zeros_block(T,A.row_ranks[n],0)
  end

  return A
end

"""
  A_left!(A::SparseCore{T,N,d}) where {T<:Number,N,d}

Implements second quantization annihilation operator :math: (a_k) on core `k`,
assuming it comes *left* from the corresponding  creation operator in the 
number-conserving block :math: (a_i^* a_k), i.e. :math: (k < i).
"""
function A_left!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = A.k
  @boundscheck @assert k < d

  n = last(axes(A,3))
  A.col_ranks[n] = 0
  if n ∈ axes(A,1)
    A.row_ranks[n] = 0
    A[n,1,n] = zeros_block(T,0,0)
  end

# Generic case : both occupied and unoccupied are allowed
  qn = first(axes(A,3)):(last(axes(A,3))-1)
  @boundscheck @assert qn == axes(A.unoccupied,1) ∩ axes(A.occupied,1)
  for n in reverse(qn) 
    A.col_ranks[n] = size(A[n,2,n+1],2)
    A[n,1,n  ] = A[n,2,n+1]
    A[n,2,n+1] = zeros_block(T,A.row_ranks[n],A.col_ranks[n+1])
  end

# Case n ∉ axes(A,3) - the unoccupied block A[n,1,n] does not exist
  if first(axes(A,1)) < first(axes(A,3))
    n = first(axes(A,1))
    @boundscheck @assert n+1 == first(axes(A,3)) && d+1-k == N-n

    A.row_ranks[n] = 0
    A[n,2,n+1] = zeros_block(T,A.row_ranks[n],A.col_ranks[n+1])
  end

  return A
end

"""
  A_right!(A::SparseCore{T,N,d}) where {T<:Number,N,d}

Implements second quantization annihilation operator :math: (a_k) on core `k`,
assuming it comes *right* from the corresponding creation operator in the 
number-conserving block :math: (a_i^* a_k), i.e. :math: (i < k).
"""
function A_right!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = A.k
  @boundscheck @assert k > 1

  n = first(axes(A,1))
  A.row_ranks[n] = 0
  if n ∈ axes(A,3)
    A.col_ranks[n] = 0
    A[n,1,n] = zeros_block(T,0,0)
  end

# Generic case : both occupied and unoccupied are allowed on this column
  qn = (first(axes(A,1))+1):last(axes(A,1))
  @boundscheck @assert qn == axes(A.unoccupied,1) ∩ (axes(A.occupied,1).+1)
  for n in qn
    A.row_ranks[n] = size(A[n-1,2,n],1)
    A[n  ,1,n] = A[n-1,2,n]
    A[n-1,2,n] = zeros_block(T,A.row_ranks[n-1],A.col_ranks[n])
  end

# Case n ∉ axes(A,1), i.e. k==n - the unoccupied block A[n,1,n] does not exist
  if last(axes(A,3)) > last(axes(A,1))
    n = last(axes(A,3))
    @boundscheck @assert n-1 == last(axes(A,1)) && n == k

    A.col_ranks[n] = 0
    A[n-1,2,n] = zeros_block(T,A.row_ranks[n-1],A.col_ranks[n])
  end

  return A
end

"""
  S⁺!(A::SparseCore{T,N,d}) where {T<:Number,N,d}

Implements Jordan-Wigner component :math: (s_k) on core `k`,
assuming it comes in between second quantization creation (to the left) and 
annihilation (to the right) operators in a number-conserving block :math: (a_i^* a_j), 
i.e. :math: (i < j < k).
"""
function S⁺!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = A.k
  @boundscheck @assert 1 < k < d

  A.row_ranks[begin+1:end] = A.row_ranks[begin:end-1]
  A.col_ranks[begin+1:end] = A.col_ranks[begin:end-1]
  A.row_ranks[begin] = 0
  A.col_ranks[begin] = 0

  qn = axes(A.unoccupied,1)
  for n in last(qn):-1:(first(qn)+1)
    A.unoccupied[n] = A.unoccupied[n-1]
  end
  n = first(qn)
  A.unoccupied[n] = zeros_block(T,A.row_ranks[n],A.col_ranks[n])

  qn = axes(A.occupied,1)
  for n in last(qn):-1:(first(qn)+1)
    A.occupied[n] = lmul!(-1, A.occupied[n-1])
  end
  n = first(qn)
  A.occupied[n] = zeros_block(T,A.row_ranks[n],A.col_ranks[n+1])

  return A
end

"""
  S⁻!(A::SparseCore{T,N,d}) where {T<:Number,N,d}

Implements Jordan-Wigner component :math: (s_k) on core `k`,
assuming it comes in between second quantization annihilation (to the left) and 
creation (to the right) operators in a number-conserving block :math: (a_i^* a_j), 
i.e. :math: (j < k < i).
"""
function S⁻!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = A.k
  @boundscheck @assert 1 < k < d

  A.row_ranks[begin:end-1] = A.row_ranks[begin+1:end]
  A.col_ranks[begin:end-1] = A.col_ranks[begin+1:end]
  A.row_ranks[end] = 0
  A.col_ranks[end] = 0

  qn = axes(A.unoccupied,1)
  for n in first(qn):(last(qn)-1)
    A.unoccupied[n] = A.unoccupied[n+1]
  end
  n = last(qn)
  A.unoccupied[n] = zeros_block(T,A.row_ranks[n],A.col_ranks[n])

  qn = axes(A.occupied,1)
  for n in first(qn):(last(qn)-1)
    A.occupied[n] = lmul!(-1, A.occupied[n+1])
  end
  n = last(qn)
  A.occupied[n] = zeros_block(T,A.row_ranks[n],A.col_ranks[n+1])

  return A
end

"""
  I_left!(A::SparseCore{T,N,d}) where {T<:Number,N,d}

Implements Identity component :math: (i_k) on core `k`,
assuming it comes to the left of both second quantization annihilation and 
creation operators in a number-conserving block :math: (a_i^* a_j), i.e. :math: (k < i,j).
"""
function I_left!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = A.k
  @boundscheck @assert k < d

  if N == last(axes(A,1))
# Case N ∈ axes(A,1), but there has to be at least one electron on the right now
    A.row_ranks[N] = 0
    A.col_ranks[N] = 0
    A[N-1,2,N] = zeros_block(T,A.row_ranks[N-1],0)
    A[N,1,N] = zeros_block(T,0,0)
  elseif N == last(axes(A,3))
# Case N ∈ axes(A,3) but not axes(A,1), but there has to be at least one electron on the right now
    A.col_ranks[N] = 0
    A[N-1,2,N] = zeros_block(T,A.row_ranks[N-1],A.col_ranks[N])
  end

  return A
end

"""
  I_right!(A::SparseCore{T,N,d}) where {T<:Number,N,d}

Implements Identity component :math: (i_k) on core `k`,
assuming it comes to the right of both second quantization annihilation and 
creation operators in a number-conserving block :math: (a_i^* a_j), i.e. :math: (i,j < k).
"""
function I_right!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = A.k
  @boundscheck @assert k > 1

  if 0 == first(axes(A,3))
# Case 0 ∈ axes(A,3), but there has to be at least one electron on the left now
    A.row_ranks[0] = 0
    A.col_ranks[0] = 0
    A[0,1,0] = zeros_block(T,0,0)
    A[0,2,1] = zeros_block(T,0,A.col_ranks[1])
  elseif 0 == first(axes(A,1))
# Case 0 ∈ axes(A,1) but not axes(A,3), but there has to be at least one electron on the left now
    A.row_ranks[0] = 0
    A[0,2,1] = zeros_block(T,A.row_ranks[0],A.col_ranks[1])
  end

  return A
end

"""
  AdagA!(A::SparseCore{T,N,d}) where {T<:Number,N,d}

Implements annihilation/creation block :math: (a^*_ka_k) on core `k`.
"""
function AdagA!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  k = A.k
  if 0 == first(axes(A,3))
    A.col_ranks[0] = 0
  end
  if N == last(axes(A,1))
    A.row_ranks[N] = 0
  end
  for n in axes(A.unoccupied, 1)
    if size(A.unoccupied[n]) == (A.row_ranks[n], A.col_ranks[n])
      lmul!(0, A.unoccupied[n])
    else
      A.unoccupied[n] = zeros_block(T,A.row_ranks[n], A.col_ranks[n])
    end
  end

  return A
end

function AdagᵢAⱼ!(tt::TTvector{T,N,d}, i::Int, j::Int) where {T<:Number,N,d}
  @boundscheck @assert 1 ≤ i ≤ d && 1 ≤ j ≤ d

  for k=1:min(i,j)-1
  # Identity operator to the left of a pair creation / annihilation
    I_left!(core(tt,k))
    # Adjust ranks
    N ∈ axes(core(tt,k),1) && (rank(tt,k)[N] = 0)
  end

# Case i < j : Op = Iₗ ⊗ ⋯ ⊗ Iₗ ⊗ A† ⊗ S⁺ ⊗ ⋯ ⊗ S⁺ ⊗ A ⊗ Iᵣ ⊗ ⋯ ⊗ Iᵣ
  if (i<j) 
  # Creation operator at i
    Adag_left!(core(tt,i))
    # Adjust ranks
    N ∈ axes(core(tt,i),1) && (rank(tt,i)[N] = 0)
    rank(tt,i+1)[begin+1 : end] = rank(tt,i+1)[begin : end-1]
    rank(tt,i+1)[begin] = 0

  # Jordan-Wigner chain
    for k=i+1:j-1
  # Phase factors
      S⁺!(core(tt,k))
    # Adjust ranks
      rank(tt,k+1)[begin+1 : end] = rank(tt,k+1)[begin : end-1]
      rank(tt,k+1)[begin] = 0
    end
  # Annihilation operator at j
    # Adjust ranks
    A_right!(core(tt,j))
    0 ∈ axes(core(tt,j),3) && (rank(tt,j+1)[0] = 0)

# Case i = j : Op = Iₗ ⊗ ⋯ ⊗ Iₗ ⊗ A†A ⊗ Iᵣ ⊗ ⋯ ⊗ Iᵣ
  elseif (i==j)
  # Annihilation then creation at i=j
    AdagA!(core(tt,i))
    # Adjust ranks
    N ∈ axes(core(tt,i),1) && (rank(tt,i  )[N] = 0)
    0 ∈ axes(core(tt,i),3) && (rank(tt,i+1)[0] = 0)

# Case i < j : Op = Iₗ ⊗ ⋯ ⊗ Iₗ ⊗ A ⊗ S⁻ ⊗ ⋯ ⊗ S⁻ ⊗ A† ⊗ Iᵣ ⊗ ⋯ ⊗ Iᵣ
  elseif (j<i)
  # Annihilation operator at j
    A_left!(core(tt,j))
    # Adjust ranks
    N ∈ axes(core(tt,j),1) && (rank(tt,j)[N] = 0)
    rank(tt,j+1)[begin : end-1] = rank(tt,j+1)[begin+1 : end]
    rank(tt,j+1)[end] = 0

  # Jordan-Wigner chain
    for k=j+1:i-1
  # Phase factors
      S⁻!(core(tt,k))
    # Adjust ranks
      rank(tt,k+1)[begin : end-1] = rank(tt,k+1)[begin+1 : end]
      rank(tt,k+1)[end] = 0
    end
    Adag_right!(core(tt,i))
    # Adjust ranks
    0 ∈ axes(core(tt,i),3) && (rank(tt,i+1)[0] = 0)
  end

  for k=max(i,j)+1:d
  # Identity operator to the right of a pair creation / annihilation
    I_right!(core(tt,k))
    # Adjust ranks
    0 ∈ axes(core(tt,k),3) && (rank(tt,k+1)[0] = 0)
  end

  # Sanity check for the ranks
  check(tt)
end

end
