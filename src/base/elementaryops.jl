########################################################
### Implement two-body second quantization operators ###
########################################################


"""
  Adag(A, spin, flux_up::Int, nl_up::Int, nr_up::Int, 
                flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements second quantization creation operator :math: (a_k↑↓^*) on core `k`,
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function Adag(A::SparseCore{T,Nup,Ndn,d,M}, spin::Spin,
              flux_up::Int, nl_up::Int, nr_up::Int, 
              flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  ql, rowranks = shift_ranks(row_qn(A), row_ranks(A), Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)

  if spin == Up
    # Ensure that there is room for `nl_up` spin up electrons to the left of core `k`
    # as well as `nr_up-1` spin up electrons to the right of core `k` (excluded) by 
    # allowing only certain rows and columns

    qr, colranks = shift_ranks(col_qn(A), col_ranks(A), Nup, Ndn, flux_up+1, nl_up+1, nr_up-1, flux_dn, nl_dn, nr_dn)
    
    B = SparseCore{T,Nup,Ndn,d}(k,rowranks,colranks)
    for (lup,ldn) in row_qn(A) ∩ ql
      jw = isodd(lup-flux_up+ldn-flux_dn) ? -1 : 1
      if (lup+1,ldn) in col_qn(A) ∩ qr
        up(B,lup,ldn) .= jw .* ○○(A,lup-flux_up,ldn-flux_dn)
        # @show k, (up=:Adag,dn=:Id), (lup-flux_up,ldn-flux_dn), :○○, jw
      end
      if (lup+1,ldn+1) in col_qn(A) ∩ qr
        ●●(B,lup,ldn) .= jw .* dn(A,lup-flux_up,ldn-flux_dn)
        # @show k, (up=:Adag,dn=:Id), (lup-flux_up,ldn-flux_dn), :dn, jw
      end
    end
    return B, flux_up+1, nl_up+1, nr_up-1, flux_dn, nl_dn, nr_dn

  elseif spin == Dn
    # Ensure that there is room for `nl_dn` spin down electrons to the left of core `k`
    # as well as `nr_dn-1` spin down electrons to the right of core `k` (excluded) by 
    # allowing only certain rows and columns

    qr, colranks = shift_ranks(col_qn(A), col_ranks(A), Nup, Ndn, flux_up, nl_up, nr_up, flux_dn+1, nl_dn+1, nr_dn-1)
    
    B = SparseCore{T,Nup,Ndn,d}(k,rowranks,colranks)
    for (lup,ldn) in row_qn(A) ∩ ql
      if (lup,ldn+1) in col_qn(A) ∩ qr
        jw = isodd(lup-flux_up+ldn-flux_dn) ? -1 : 1
        dn(B,lup,ldn) .= jw .* ○○(A,lup-flux_up,ldn-flux_dn)
        # @show k, (up=:Id,dn=:Adag), (lup-flux_up,ldn-flux_dn), :○○, jw
      end
      if (lup+1,ldn+1) in col_qn(A) ∩ qr
        jw = isodd(lup+1-flux_up+ldn-flux_dn) ? -1 : 1
        ●●(B,lup,ldn) .= jw .* up(A,lup-flux_up,ldn-flux_dn)
        # @show k, (up=:Id,dn=:Adag), (lup-flux_up,ldn-flux_dn), :up, jw
      end
    end
    return B, flux_up, nl_up, nr_up, flux_dn+1, nl_dn+1, nr_dn-1
  end
end


"""
  A(A, spin::Spin, flux_up::Int, nl_up::Int, nr_up::Int, 
                   flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements second quantization annihilation operator :math: (a_k↑) on core `k`,
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function A(A::SparseCore{T,Nup,Ndn,d,M}, spin::Spin, 
                flux_up::Int, nl_up::Int, nr_up::Int, 
                flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))


  ql, rowranks = shift_ranks(row_qn(A), row_ranks(A), Nup, Ndn, flux_up,   nl_up,   nr_up,   flux_dn, nl_dn, nr_dn)
  if spin==Up
  # Ensure that there is room for `nl_up` spin up electrons to the left of core `k`
  # as well as `nr_up-1` spin up electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns
    qr, colranks = shift_ranks(col_qn(A), col_ranks(A), Nup, Ndn, flux_up-1, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
    B = SparseCore{T,Nup,Ndn,d}(k,rowranks,colranks)

    for (lup,ldn) in row_qn(A) ∩ ql
      jw = isodd(lup-flux_up+ldn-flux_dn) ? -1 : 1
      if (lup,ldn) in col_qn(A) ∩ qr
        ○○(B,lup,ldn) .= jw .* up(A,lup-flux_up,ldn-flux_dn)
        # @show k, (up=:A,dn=:Id), (lup-flux_up,ldn-flux_dn), :up, jw
      end
      if (lup,ldn+1) in col_qn(A) ∩ qr
        dn(B,lup,ldn) .= jw .* ●●(A,lup-flux_up,ldn-flux_dn)
        # @show k, (up=:A,dn=:Id), (lup-flux_up,ldn-flux_dn), :●●, jw
      end
    end

    return B, flux_up-1, nl_up, nr_up, flux_dn, nl_dn, nr_dn
  elseif spin==Dn
    # Ensure that there is room for `nl_up` spin up electrons to the left of core `k`
    # as well as `nr_up-1` spin up electrons to the right of core `k` (excluded) by 
    # allowing only certain rows and columns
    qr, colranks = shift_ranks(col_qn(A), col_ranks(A), Nup, Ndn, flux_up, nl_up, nr_up, flux_dn-1, nl_dn, nr_dn)
    B = SparseCore{T,Nup,Ndn,d}(k,rowranks,colranks)

    for (lup,ldn) in row_qn(A) ∩ ql
      if (lup,ldn) in col_qn(A) ∩ qr
        jw = isodd(lup-flux_up+ldn-flux_dn) ? -1 : 1
        ○○(B,lup,ldn) .= jw .* dn(A,lup-flux_up,ldn-flux_dn)
        # @show k, (up=:Id,dn=:A), (lup-flux_up,ldn-flux_dn), :dn, jw
      end
      if (lup+1,ldn) in col_qn(A) ∩ qr
        jw = isodd(lup+1-flux_up+ldn-flux_dn) ? -1 : 1
        up(B,lup,ldn) .= jw .* ●●(A,lup-flux_up,ldn-flux_dn)
        # @show k, (up=:Id,dn=:A), (lup-flux_up,ldn-flux_dn), :●●, jw
      end
    end

    return B, flux_up, nl_up, nr_up, flux_dn-1, nl_dn, nr_dn
  end
end

"""
  AdagupAdagdn(A, flux_up::Int, nl_up::Int, nr_up::Int, 
                  flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements second quantization creation operator :math: (a_k↑^*a_k↓^*) on core `k`,
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function AdagupAdagdn(A::SparseCore{T,Nup,Ndn,d,M}, 
                flux_up::Int, nl_up::Int, nr_up::Int, 
                flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  # Ensure that there is room for `nl_dn` spin down electrons to the left of core `k`
  # as well as `nr_dn-1` spin down electrons to the right of core `k` (excluded) by 
  # allowing only certain rows and columns

  ql, rowranks = shift_ranks(row_qn(A), row_ranks(A), Nup, Ndn, flux_up,   nl_up,   nr_up,   flux_dn,   nl_dn,   nr_dn  )
  qr, colranks = shift_ranks(col_qn(A), col_ranks(A), Nup, Ndn, flux_up+1, nl_up+1, nr_up-1, flux_dn+1, nl_dn+1, nr_dn-1)
  
  B = SparseCore{T,Nup,Ndn,d}(k,rowranks,colranks)
  for (lup,ldn) in row_qn(A) ∩ ql
    if (lup+1,ldn+1) in col_qn(A) ∩ qr
      ●●(B,lup,ldn) .= ○○(A,lup-flux_up,ldn-flux_dn)
    end
  end

  return B, flux_up+1, nl_up+1, nr_up-1, flux_dn+1, nl_dn+1, nr_dn-1
end

"""
  AupAdn(A, flux_up::Int, nl_up::Int, nr_up::Int, 
            flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements second quantization annihilation operator :math: (a_k↑a_k↓) on core `k`,
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function AupAdn(A::SparseCore{T,Nup,Ndn,d,M}, 
                flux_up::Int, nl_up::Int, nr_up::Int, 
                flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  ql, rowranks = shift_ranks(row_qn(A), row_ranks(A), Nup, Ndn, flux_up,   nl_up, nr_up, flux_dn,   nl_dn, nr_dn)
  qr, colranks = shift_ranks(col_qn(A), col_ranks(A), Nup, Ndn, flux_up-1, nl_up, nr_up, flux_dn-1, nl_dn, nr_dn)
  
  B = SparseCore{T,Nup,Ndn,d}(k,rowranks,colranks)
  for (lup,ldn) in row_qn(A) ∩ ql
    if (lup,ldn) in col_qn(A) ∩ qr
      ○○(B,lup,ldn) .= (-1) .* ●●(A,lup-flux_up,ldn-flux_dn)
    end
  end

  return B, flux_up-1, nl_up, nr_up, flux_dn-1, nl_dn, nr_dn
end

"""
  N(A, spin::Spin, flux_up::Int, nl_up::Int, nr_up::Int, 
                           flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements spin-polarized number operator :math: (a_k↑^*a_k↑) or :math: (a_k↓^*a_k↓) on core `k`.
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function N(A::SparseCore{T,Nup,Ndn,d,M}, spin::Spin, 
                flux_up::Int, nl_up::Int, nr_up::Int, 
                flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  ql, rowranks = shift_ranks(row_qn(A), A.row_ranks, Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
  if spin==Up
    qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up, nl_up+1, nr_up-1, flux_dn, nl_dn, nr_dn)
    
    B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
    for (lup,ldn) in row_qn(A) ∩ ql
      if (lup+1,ldn) in col_qn(A) ∩ qr
        up(B,lup,ldn) .= up(A,lup-flux_up,ldn-flux_dn)
      end
      if (lup+1,ldn+1) in col_qn(A) ∩ qr
        ●●(B,lup,ldn) .= ●●(A,lup-flux_up,ldn-flux_dn)
      end
    end
    return B, flux_up, nl_up+1, nr_up-1, flux_dn, nl_dn, nr_dn

  elseif spin==Dn
    qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn+1, nr_dn-1)
    
    B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
    for (lup,ldn) in row_qn(A) ∩ ql
      if (lup,ldn+1) in col_qn(A) ∩ qr
        dn(B,lup,ldn) .= dn(A,lup-flux_up,ldn-flux_dn)
      end
      if (lup+1,ldn+1) in col_qn(A) ∩ qr
        ●●(B,lup,ldn) .= ●●(A,lup-flux_up,ldn-flux_dn)
      end
    end
    return B, flux_up, nl_up, nr_up, flux_dn, nl_dn+1, nr_dn-1
  end
end

"""
  S₊(A, flux_up::Int, nl_up::Int, nr_up::Int, 
        flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements J₊ ladder operator :math: (a_k↑^*a_k↓) on core `k`.
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function S₊(A::SparseCore{T,Nup,Ndn,d,M}, 
              flux_up::Int, nl_up::Int, nr_up::Int, 
              flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  ql, rowranks = shift_ranks(row_qn(A), A.row_ranks, Nup, Ndn, flux_up,   nl_up,   nr_up,   flux_dn,   nl_dn, nr_dn)
  qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up+1, nl_up+1, nr_up-1, flux_dn-1, nl_dn, nr_dn)
  
  B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
  for (lup,ldn) in row_qn(A) ∩ ql
    if (lup+1,ldn) in col_qn(A) ∩ qr
      up(B,lup,ldn) .= dn(A,lup-flux_up,ldn-flux_dn)
    end
  end

  return B, flux_up+1, nl_up+1, nr_up-1, flux_dn-1, nl_dn, nr_dn
end


"""
  S₋(A, flux_up::Int, nl_up::Int, nr_up::Int, 
               flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements J₋ ladder operator :math: (a_k↑a_k↓^*) on core `k`.
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function S₋(A::SparseCore{T,Nup,Ndn,d,M}, 
              flux_up::Int, nl_up::Int, nr_up::Int, 
              flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  ql, rowranks = shift_ranks(row_qn(A), A.row_ranks, Nup, Ndn, flux_up,   nl_up, nr_up, flux_dn,   nl_dn,   nr_dn  )
  qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up-1, nl_up, nr_up, flux_dn+1, nl_dn+1, nr_dn-1)
  
  B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
  for (lup,ldn) in row_qn(A) ∩ ql
    if (lup,ldn+1) in col_qn(A) ∩ qr
      dn(B,lup,ldn) .= (-1) .* up(A,lup-flux_up,ldn-flux_dn)
    end
  end

  return B, flux_up-1, nl_up, nr_up, flux_dn+1, nl_dn+1, nr_dn-1
end

"""
  AN(A, spin::Spin, flux_up::Int, nl_up::Int, nr_up::Int, 
                    flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements density times annihilation operator with spin `spin` :math: (a_k↑^*a_k↑a_k↓) (`spin`=`up`) or :math: (a_k↑a_k↓^*a_k↓) (`spin`=`dn`) on core `k`.
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function AN(A::SparseCore{T,Nup,Ndn,d,M}, spin::Spin,
                      flux_up::Int, nl_up::Int, nr_up::Int, 
                      flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  ql, rowranks = shift_ranks(row_qn(A), A.row_ranks, Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
  if spin == Up
    qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up-1, nl_up, nr_up, flux_dn, nl_dn+1, nr_dn-1)
    
    B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
    for (lup,ldn) in row_qn(A) ∩ ql
      jw = isodd(lup-flux_up+ldn-flux_dn) ? -1 : 1
      if (lup,ldn+1) in col_qn(A) ∩ qr
        dn(B,lup,ldn) .= jw .* ●●(A,lup-flux_up,ldn-flux_dn)
      end
    end
    return B, flux_up-1, nl_up, nr_up, flux_dn, nl_dn+1, nr_dn-1
  elseif spin == Dn
    qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up, nl_up+1, nr_up-1, flux_dn-1, nl_dn, nr_dn)
    
    B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
    for (lup,ldn) in row_qn(A) ∩ ql
      jw = isodd(lup+1-flux_up+ldn-flux_dn) ? -1 : 1
      if (lup+1,ldn) in col_qn(A) ∩ qr
        up(B,lup,ldn) .= jw .* ●●(A,lup-flux_up,ldn-flux_dn)
      end
    end
    return B, flux_up, nl_up+1, nr_up-1, flux_dn-1, nl_dn, nr_dn
  end
end


"""
  AdagN(A, spin::Spin, 
           flux_up::Int, nl_up::Int, nr_up::Int, 
           flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements density times creation operator with spin `spin` :math: (a_k↑^*a_k↑a_k↓^*) (`spin`=`dn`) or :math: (a_k↑^*a_k↓^*a_k↓) (`spin`=`up`) on core `k`.
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function AdagN(A::SparseCore{T,Nup,Ndn,d,M}, spin::Spin,
                flux_up::Int, nl_up::Int, nr_up::Int, 
                flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  ql, rowranks = shift_ranks(row_qn(A), A.row_ranks, Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
  if spin==Dn
    qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up, nl_up+1, nr_up-1, flux_dn+1, nl_dn+1, nr_dn-1)
    
    B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
    for (lup,ldn) in row_qn(A) ∩ ql
      jw = isodd(lup+1-flux_up+ldn-flux_dn) ? -1 : 1
      if (lup+1,ldn+1) in col_qn(A) ∩ qr
        ●●(B,lup,ldn) .= jw .* up(A,lup-flux_up,ldn-flux_dn)
      end
    end
    return B, flux_up, nl_up+1, nr_up-1, flux_dn+1, nl_dn+1, nr_dn-1
  elseif spin==Up
    qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up+1, nl_up+1, nr_up-1, flux_dn, nl_dn+1, nr_dn-1)
    
    B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
    for (lup,ldn) in row_qn(A) ∩ ql
      jw = isodd(lup-flux_up+ldn-flux_dn) ? -1 : 1
      if (lup+1,ldn+1) in col_qn(A) ∩ qr
        ●●(B,lup,ldn) .= jw .* dn(A,lup-flux_up,ldn-flux_dn)
      end
    end
    return B, flux_up+1, nl_up+1, nr_up-1, flux_dn, nl_dn+1, nr_dn-1
  end
end

"""
  NN(A, flux_up::Int, nl_up::Int, nr_up::Int, 
            flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements spin up+down density operator :math: (a_k↑^*a_k↑a_k↓^*a_k↓) on core `k`.
assuming the quantum number fluxes in the chain up to core 'k' are `flux_up` and `flux_dn`, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function NN(A::SparseCore{T,Nup,Ndn,d,M}, 
                            flux_up::Int, nl_up::Int, nr_up::Int, 
                            flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  ql, rowranks = shift_ranks(row_qn(A), A.row_ranks, Nup, Ndn, flux_up, nl_up,   nr_up,   flux_dn, nl_dn,   nr_dn  )
  qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up, nl_up+1, nr_up-1, flux_dn, nl_dn+1, nr_dn-1)
  
  B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
  for (lup,ldn) in row_qn(A) ∩ ql
    if (lup+1,ldn+1) in col_qn(A) ∩ qr
      ●●(B,lup,ldn) .= ●●(A,lup-flux_up,ldn-flux_dn)
    end
  end

  return B, flux_up, nl_up+1, nr_up-1, flux_dn, nl_dn+1, nr_dn-1
end

"""
  Id(A, flux_up::Int, nl_up::Int, nr_up::Int, 
        flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}

Implements Identity component :math: (i_k) on core `k`,
assuming the quantum number flux in the chain up to core 'k' is `flux=flux_up+flux_dn` and even, 
and we must fit `nl_up`/`nr_dn` electrons to the left and `nr_up`/`nr_dn` electrons to the right of core `k` (included).
"""
function Id(A::SparseCore{T,Nup,Ndn,d,M},
            flux_up::Int, nl_up::Int, nr_up::Int, 
            flux_dn::Int, nl_dn::Int, nr_dn::Int) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  k = A.k
  @boundscheck 1 ≤ k ≤ d || throw(BoundsError(A))

  ql, rowranks = shift_ranks(row_qn(A), A.row_ranks, Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
  qr, colranks = shift_ranks(col_qn(A), A.col_ranks, Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
  
  B = SparseCore{T,Nup,Ndn,d}(k, rowranks, colranks)
  for (lup,ldn) in row_qn(A) ∩ ql
    if (lup,ldn) in col_qn(A) ∩ qr
      ○○(B,lup,ldn) .= ○○(A,lup-flux_up,ldn-flux_dn)
    end
    if (lup+1,ldn  ) in col_qn(A) ∩ qr
      up(B,lup,ldn) .= up(A,lup-flux_up,ldn-flux_dn)
    end
    if (lup  ,ldn+1) in col_qn(A) ∩ qr
      dn(B,lup,ldn) .= dn(A,lup-flux_up,ldn-flux_dn)
    end
    if (lup+1,ldn+1) in col_qn(A) ∩ qr
      ●●(B,lup,ldn) .= ●●(A,lup-flux_up,ldn-flux_dn)
    end
  end
  return B, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn
end


function AdagᵢAⱼ(tt_in::TTvector{T,Nup,Ndn,d,M}, i::Orbital, j::Orbital, t::T=T(1)) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  @boundscheck 1 ≤ i.site ≤ d && 1 ≤ j.site ≤ d && i.spin == j.spin

  # Corner case
  if (i.spin == Up && Nup < 1) || (i.spin == Dn && Ndn < 1)
    return tt_zeros(Val(d),Val(Nup),Val(Ndn),T)
  else 
    if i.spin == j.spin == Up
      spin = Up
      flux_up, nl_up, nr_up= 0, 0, 1
      flux_dn, nl_dn, nr_dn= 0, 0, 0
    elseif i.spin == j.spin == Dn
      spin = Dn
      flux_up, nl_up, nr_up= 0, 0, 0
      flux_dn, nl_dn, nr_dn= 0, 0, 1
    end

    cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef, d)
    ranks = deepcopy(rank(tt_in))

    for site=1:d
      # Adjust row ranks using flux to determine shift
      shift_ranks!(row_qn(core(tt_in,site)), ranks[site], rank(tt_in, site), Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)

      if site == i.site == j.site # Density operator
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = N(    core(tt_in,site), spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site == i.site            # Creation operator
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = Adag( core(tt_in,site), spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site == j.site            # Annihilation operator
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = A(    core(tt_in,site), spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      else
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = Id(   core(tt_in,site),       flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      end
    end
    shift_ranks!(col_qn(core(tt_in,d)), ranks[d+1], rank(tt_in, d+1), Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
    tt_out = lmul!(ε(i,j)*t, TTvector(ranks, cores))
    # Sanity check for the ranks
    check(tt_out)
    return tt_out
  end
end

function AdagᵢAdagₖAₗAⱼ(tt_in::TTvector{T,Nup,Ndn,d,M}, i::Orbital, j::Orbital, k::Orbital, l::Orbital, w::T=T(1)) where {T<:Number,Nup,Ndn,d,M<:AbstractMatrix{T}}
  @boundscheck begin
    @assert i.spin == j.spin && k.spin == l.spin
    @assert i≠k && j≠l
  end

   # Corner case
  if (i.spin == k.spin == Up && Nup < 2) || (i.spin == k.spin == Dn && Ndn < 2) || (i.spin ≠ k.spin && (Nup < 1 || Ndn < 1))
    return tt_zeros(Val(d),Val(Nup),Val(Ndn),T)
  else
    cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef, d)
    ranks = deepcopy(rank(tt_in))

    flux_up, nl_up, nr_up = 0, 0, (i.spin == Up ? 1 : 0) + (k.spin == Up ? 1 : 0)
    flux_dn, nl_dn, nr_dn = 0, 0, (i.spin == Dn ? 1 : 0) + (k.spin == Dn ? 1 : 0)

    for site=1:d
      shift_ranks!(row_qn(core(tt_in,site)), ranks[site], rank(tt_in, site), Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)

      if     site == i.site == j.site == k.site == l.site
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = NN(    core(tt_in,site),         flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site == i.site == j.site == k.site
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = AdagN( core(tt_in,site), k.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site == i.site == j.site           == l.site
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = AN(    core(tt_in,site), l.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site == i.site ==           k.site == l.site
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = AdagN( core(tt_in,site), i.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site ==           j.site == k.site == l.site
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = AN(    core(tt_in,site), j.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site == i.site == j.site
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = N(     core(tt_in,site), i.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site ==                     k.site == l.site
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = N(     core(tt_in,site), k.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site == i.site ==                     l.site
        if i.spin == l.spin
          cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = N(   core(tt_in,site), i.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
        elseif i.spin == Up && l.spin == Dn
          cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = S₊(  core(tt_in,site),         flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
        elseif i.spin == Dn && l.spin == Up
          cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = S₋(  core(tt_in,site),         flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
        end
      elseif site ==           j.site == k.site
        if k.spin == j.spin
          cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = N(   core(tt_in,site), k.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
        elseif k.spin == Up && j.spin == Dn
          cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = S₊(  core(tt_in,site),         flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
        elseif k.spin == Dn && j.spin == Up
          cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = S₋(  core(tt_in,site),         flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
        end
      elseif site == i.site ==           k.site
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = AdagupAdagdn( core(tt_in,site),  flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site ==           j.site ==           l.site
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = AupAdn(core(tt_in,site),         flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site == i.site                                # Creation operator
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = Adag(  core(tt_in,site), i.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site ==           j.site                      # Annihilation operator
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = A(     core(tt_in,site), j.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site ==                     k.site            # Creation operator
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = Adag(  core(tt_in,site), k.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      elseif site ==                               l.site  # Annihilation operator
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = A(     core(tt_in,site), l.spin, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      else
        cores[site], flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn = Id(    core(tt_in,site),         flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
      end
    end
    shift_ranks!(col_qn(core(tt_in,d)), ranks[d+1], rank(tt_in, d+1), Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)

    tt_out = lmul!(ε(i,j,k,l)*w, TTvector(ranks, cores))
    # Sanity check for the ranks
    check(tt_out)

    return tt_out
  end
end

# Convenience function
@inline function shift_qn(qns::Vector{Tuple{Int,Int}}, Nup::Int, Ndn::Int, 
                          flux_up::Int, nl_up::Int, nr_up::Int, 
                          flux_dn::Int, nl_dn::Int, nr_dn::Int)
  return filter(
      qn -> ( 
        (nup,ndn) = qn; 
        return nl_up ≤ nup-1 ≤ Nup-nr_up && nl_dn ≤ ndn-1 ≤ Ndn-nr_dn && (nup-flux_up, ndn-flux_dn) in qns
      ), 
      qns)
end

function shift_ranks!(qns::Vector{Tuple{Int,Int}}, ranks::Matrix{Int}, Nup::Int, Ndn::Int, 
                          flux_up::Int, nl_up::Int, nr_up::Int, 
                          flux_dn::Int, nl_dn::Int, nr_dn::Int)
  @boundscheck @assert size(ranks,1) ≥ flux_up
  @boundscheck @assert nl_up ≥ 0 && nr_up ≥ 0 && nl_up + nr_up ≤ Nup
  @boundscheck @assert size(ranks,2) ≥ flux_dn
  @boundscheck @assert nl_dn ≥ 0 && nr_dn ≥ 0 && nl_dn + nr_dn ≤ Ndn

  shifted_qns = shift_qn(qns,Nup,Ndn,flux_up,nl_up,nr_up,flux_dn,nl_dn,nr_dn)
  for (nup,ndn) in shifted_qns
    ranks[nup,ndn] = ranks[nup-flux_up,ndn-flux_dn]
  end
  for (nup,ndn) in setdiff(qns, shifted_qns)
    ranks[nup,ndn] = 0
  end
  return shifted_qns
end

function shift_ranks!(qns::Vector{Tuple{Int,Int}}, new_ranks::AbstractMatrix{Int}, ranks::AbstractMatrix{Int}, Nup::Int, Ndn::Int, 
                          flux_up::Int, nl_up::Int, nr_up::Int, 
                          flux_dn::Int, nl_dn::Int, nr_dn::Int)
  @boundscheck @assert size(ranks,1) ≥ flux_up
  @boundscheck @assert nl_up ≥ 0 && nr_up ≥ 0 && nl_up + nr_up ≤ Nup
  @boundscheck @assert size(ranks,2) ≥ flux_dn
  @boundscheck @assert nl_dn ≥ 0 && nr_dn ≥ 0 && nl_dn + nr_dn ≤ Ndn

  new_ranks .= 0
  shifted_qns = shift_qn(qns,Nup,Ndn,flux_up,nl_up,nr_up,flux_dn,nl_dn,nr_dn)
  for (nup,ndn) in shifted_qns
    new_ranks[nup,ndn] = ranks[nup-flux_up,ndn-flux_dn]
  end

  return shifted_qns
end

function shift_ranks(qns::Vector{Tuple{Int,Int}}, ranks::AbstractMatrix{Int}, Nup::Int, Ndn::Int, 
                          flux_up::Int, nl_up::Int, nr_up::Int, 
                          flux_dn::Int, nl_dn::Int, nr_dn::Int)
  new_ranks = deepcopy(ranks)
  shifted_qns = shift_ranks!(qns, new_ranks, ranks, Nup, Ndn, flux_up, nl_up, nr_up, flux_dn, nl_dn, nr_dn)
  return shifted_qns, new_ranks
end

# Fermionic anticommutation dictates we fix an order for the operators, 
# which implies changing sign according to the permutation parity.
function ε(i::Orbital, j::Orbital)
  return ( i.site<j.site || i == j || (i.site == j.site && i.spin==Dn && j.spin==Up) ? 1 : -1 )
end

function ε(i::Orbital, j::Orbital, k::Orbital, l::Orbital)
  p = sortperm([i,k,l,j], lt=(i,j) -> (i.site<j.site || (i.site == j.site && i.spin==Up && j.spin==Dn)) )
  return ε!(p)
end

function ε!(p::Vector{Int})
  n = length(p)
  if n==1
    return 1
  else
    i = findfirst(isequal(n),p)
    popat!(p,i)
    return ( isodd(n-i) ? -1 : 1 ) * ε!(p)
  end
end
