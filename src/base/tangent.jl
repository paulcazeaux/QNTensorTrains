using LinearAlgebra, OffsetArrays
"""
    TTtangent{T<:Number,Nup,Ndn,d}

Implementation of tangent vectors of the manifold of constant TT-rank.
"""
mutable struct TTtangent{T<:Number,Nup,Ndn,d}
  r          :: Vector{Matrix{Int}}

  baseL      :: TTvector{T,Nup,Ndn,d}
  baseR      :: TTvector{T,Nup,Ndn,d}
  components :: Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}

  """
      TTtangent(baseL::TTvector{T,Nup,Ndn,d}, baseR::TTvector{T,Nup,Ndn,d}, direction::TTvector{T,Nup,Ndn,d})

  Lazy constructor that does not make a copy of the base vector, but takes as input two versions of the TTtensor,
  `baseL` which should be left-orthogonal and `baseR` which should be right-orthogonal.

  No effort is made to check that the two TTtensors do in fact represent the same tensor, except for checking their modal size and ranks.
  """
  function TTtangent(baseL::TTvector{T,Nup,Ndn,d}, baseR::TTvector{T,Nup,Ndn,d}, direction::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
    @assert rank(baseL) == rank(baseR)
    @assert rank(baseL,  1,1) == rank(direction,  1,1) == 1
    @assert rank(baseL,d+1,1) == rank(direction,d+1,1) == 1
    @assert baseR.orthogonal && baseR.corePosition == 1
    @assert baseL.orthogonal && baseL.corePosition == d

    # Compute the left projection matrices: Pi = U^T_{k-1} ... U^T_1 W_1 ... W_{k-1}, a block-diagonal r[k] × rank(direction,k) matrix
    P = Vector{Frame{T,Nup,Ndn,d,Matrix{T}}}(undef, d)
    P[1] = IdFrame(Val(d), Val(Nup), Val(Ndn), 1)
    for k=1:d-1
      P[k+1] = (adjoint(core(baseL,k)) * P[k]) * core(direction,k)
    end

    # Compute the right projection matrices: Q = W_{i+1} ... W_d V^T_d ... V^T_{i+1}, a direction.r[i+1] × r[i+1] matrix
    # Assemble the components of the tangent vector for i=d,...,1
    Qₖ = IdFrame(Val(d), Val(Nup), Val(Ndn), d+1)
    components = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef, d)

    for k=d:-1:1
      DₖQₖ = core(direction,k) * Qₖ
      components[k] = P[k] * DₖQₖ
      if k < d
        # Apply gauge condition
        for r in axes(components[k],3)
          L = core(baseL,k)[r,:vertical]
          components[k][r,:vertical] = (I - L * adjoint(L)) * components[k][r,:vertical]
        end
      end
      Qₖ = DₖQₖ * adjoint(core(baseR,k))
    end

    return new{T,Nup,Ndn,d}(deepcopy(rank(baseL)),baseL,baseR,components)
  end
end

"""
    TTtangent(base::TTvector{T,Nup,Ndn,d}, direction::TTvector{T,Nup,Ndn,d})

Compute the tangent vector at `base` to the manifold of constant ranks `base.r` in the direction `direction`.
Makes two copies of the TTtensor `base`.
"""
function TTtangent(base::TTvector{T,Nup,Ndn,d}, direction::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  # Create left-orthogonalized copy of base TTtensor U_1 ... U_d
  baseL = deepcopy(base)
  leftOrthogonalize!(baseL, keepRank=true)
  # Create right-orthogonalized copy of base TTtensor V_1 ... V_d
  baseR = deepcopy(base)
  rightOrthogonalize!(baseR, keepRank=true)

  return TTtangent(baseL, baseR, direction)
end

function component(dx::TTtangent{T,Nup,Ndn,d}, k::Int) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert 1 ≤ k ≤ d
  return dx.components[k]
end

function Base.show(io::IO, dx::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  if get(io, :compact, true)
    str = "TTtangent{$T,$d}. Maximum rank $(maximum(sum.(dx.r)))"
  else
    # Manage some formatting and padding
    strr = ["r[k]=$(r)" for r in dx.r]
    len = max(length.(strr)...)
    padr = len .- length.(strr)
    strr = ["$(s)"*" "^(pad+len+3)          for (s,pad) in zip(strr, padr)]


    str = string("TTtangent{$T,$d}. Ranks are:\n", strr...)
  end
    print(io, str)
end

function Base.:+(dx::TTtangent{T,Nup,Ndn,d}, dy::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @assert rank(dx) == rank(dy)

  result = deepcopy(dx)
  for k=1:d
    component(result,k) .+= component(dy,k)
  end
  return result
end

function Base.:-(dx::TTtangent{T,Nup,Ndn,d}, dy::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @assert rank(dx) == rank(dy)

  result = deepcopy(dx)
  for k=1:d
    component(result,k) .-= component(dy,k)
  end
  return result
end

LinearAlgebra.rank(dx::TTtangent) = dx.r
LinearAlgebra.rank(dx::TTtangent, k::Int) = dx.r[k]
LinearAlgebra.rank(dx::TTtangent, k::Int, qn::NTuple{2,Int}) = dx.r[k][qn[1],qn[2]]

function LinearAlgebra.axpby!(a::Number, dx::TTtangent{T,Nup,Ndn,d}, b::Number, dy::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @assert rank(dx) == rank(dy)

  for k=1:d
    axpby!(T(a), dx.components[k], T(b), component(dy,k))
  end
  return dy
end

function LinearAlgebra.axpy!(a::Number, dx::TTtangent{T,Nup,Ndn,d}, dy::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @assert rank(dx) == rank(dy)

  for k=1:d
    axpby!(T(a), dx.components[k], component(dy,k))
  end
  return dy
end

function LinearAlgebra.lmul!(a::Number, dx::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  for k=1:d
    lmul!(a,component(dx,k))
  end
  return dx
end

function LinearAlgebra.rmul!(dx::TTtangent{T,Nup,Ndn,d}, b::Number) where {T<:Number,Nup,Ndn,d}
  for k=1:d
    rmul!(component(dx,k),b)
  end
  return dx
end

function Base.:*(a::Number, dx::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return lmul!(a, deepcopy(dx))
end

function Base.:*(dx::TTtangent{T,Nup,Ndn,d}, b::Number) where {T<:Number,Nup,Ndn,d}
  return rmul!(deepcopy(dx), b)
end

function Base.:/(dx::TTtangent{T,Nup,Ndn,d}, b::Number) where {T<:Number,Nup,Ndn,d}
  return rmul!(deepcopy(dx), inv(b))
end

"""
    TTtensor(dx::TTtangent{T,Nup,Ndn,d})

Compute the TTtensor representation of a tangent vector.
"""
function TTvector(dx::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}

  cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef,d)
  for k=1:d
    row_ranks = (k==1 ? 1 : 2 ) .* rank(dx,k)
    col_ranks = (k==d ? 1 : 2 ) .* rank(dx,k+1)
    cores[k] = SparseCore{T,Nup,Ndn,d}(k, row_ranks, col_ranks)
  end

  for (lup,ldn) in row_qn(cores[1]), (rup,rdn) in [(rup,rdn) for (rup,rdn) in ((lup,ldn),(lup+1,ldn),(lup,ldn+1),(lup+1,ldn+1)) if in_col_qn(rup,rdn,cores[1])]
    cores[1][(lup,ldn),(rup,rdn)][:, 1:rank(dx,2,(rup,rdn))      ] = component(dx, 1)[(lup,ldn),(rup,rdn)]
    cores[1][(lup,ldn),(rup,rdn)][:,   rank(dx,2,(rup,rdn))+1:end] = core(dx.baseL,1)[(lup,ldn),(rup,rdn)]
  end

  for k=2:d-1, (lup,ldn) in row_qn(cores[k]), (rup,rdn) in [(rup,rdn) for (rup,rdn) in ((lup,ldn),(lup+1,ldn),(lup,ldn+1),(lup+1,ldn+1)) if in_col_qn(rup,rdn,cores[k])]
    cores[k][(lup,ldn),(rup,rdn)][1:rank(dx,k,(lup,ldn)),      1:rank(dx,k+1,(rup,rdn))      ] = core(dx.baseR,k)[(lup,ldn),(rup,rdn)]
    cores[k][(lup,ldn),(rup,rdn)][  rank(dx,k,(lup,ldn))+1:end,1:rank(dx,k+1,(rup,rdn))      ] = component(dx, k)[(lup,ldn),(rup,rdn)]
    cores[k][(lup,ldn),(rup,rdn)][  rank(dx,k,(lup,ldn))+1:end,  rank(dx,k+1,(rup,rdn))+1:end] = core(dx.baseL,k)[(lup,ldn),(rup,rdn)]
  end

  for (lup,ldn) in row_qn(cores[d]), (rup,rdn) in [(rup,rdn) for (rup,rdn) in ((lup,ldn),(lup+1,ldn),(lup,ldn+1),(lup+1,ldn+1)) if in_col_qn(rup,rdn,cores[d])]
    cores[d][(lup,ldn),(rup,rdn)][1:rank(dx,d,(lup,ldn)),      :] = core(dx.baseR,d)[(lup,ldn),(rup,rdn)]
    cores[d][(lup,ldn),(rup,rdn)][  rank(dx,d,(lup,ldn))+1:end,:] = component(dx, d)[(lup,ldn),(rup,rdn)]
  end

  return cores2tensor(cores)
end

function LinearAlgebra.dot(dx::TTtangent{T,Nup,Ndn,d}, dy::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @assert rank(dx) == rank(dy)
  return sum(k->dot(dx.components[k], component(dy,k)), 1:d)
end

function LinearAlgebra.norm(dx::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return sqrt(sum(k->norm2(dx.components[k]), 1:d))
end

"""
    retractHOSVD(dx::TTtangent{T,Nup,Ndn,d}, α::Number=one(T))

Compute a retraction of `dx.base` + `α`⋅`dx` onto the manifold with ranks 'dx.r' using the HOSVD based rounding.
"""
function retractHOSVD(dx::TTtangent{T,Nup,Ndn,d}, α::Number=one(T)) where {T<:Number,Nup,Ndn,d}
  cores = [SparseCore{T,Nup,Ndn,d}(k, (k>1 ? 2 : 1 ) .* rank(dx,k), (k<d ? 2 : 1 ) .* rank(dx,k+1)) for k=1:d]

  for (lup,ldn) in row_qn(cores[1]), (rup,rdn) in [(rup,rdn) for (rup,rdn) in ((lup,ldn),(lup+1,ldn),(lup,ldn+1),(lup+1,ldn+1)) if in_col_qn(rup,rdn,cores[1])]
    X = cores[1][(lup,ldn),r]
    axpy!(α, component(dx, 1)[(lup,ldn),(rup,rdn)], @view X[:, 1:rank(dx,2,(rup,rdn))      ])
    axpy!(1, core(dx.baseL,1)[(lup,ldn),(rup,rdn)], @view X[:,   rank(dx,2,(rup,rdn))+1:end])
  end

  for k=2:d-1, (lup,ldn) in row_qn(cores[k]), (rup,rdn) in [(rup,rdn) for (rup,rdn) in ((lup,ldn),(lup+1,ldn),(lup,ldn+1),(lup+1,ldn+1)) if in_col_qn(rup,rdn,cores[k])]
    X = cores[k][l,r]
    axpy!(1, core(dx.baseR,k)[(lup,ldn),(rup,rdn)], @view X[1:rank(dx,k)[(lup,ldn)],      1:rank(dx,k+1,(rup,rdn))      ])
    axpy!(α, component(dx, k)[(lup,ldn),(rup,rdn)], @view X[  rank(dx,k)[(lup,ldn)]+1:end,1:rank(dx,k+1,(rup,rdn))      ])
    axpy!(1, core(dx.baseL,k)[(lup,ldn),(rup,rdn)], @view X[  rank(dx,k)[(lup,ldn)]+1:end,  rank(dx,k+1,(rup,rdn))+1:end])
  end

  for (lup,ldn) in row_qn(cores[d]), (rup,rdn) in [(rup,rdn) for (rup,rdn) in ((lup,ldn),(lup+1,ldn),(lup,ldn+1),(lup+1,ldn+1)) if in_col_qn(rup,rdn,cores[d])]
    X = cores[d][l,r]
    axpy!(1, core(dx.baseR,d)[(lup,ldn),(rup,rdn)], @view X[1:rank(dx,d,(lup,ldn)),      :])
    axpy!(1, core(dx.baseL,d)[(lup,ldn),(rup,rdn)], @view X[  rank(dx,d,(lup,ldn))+1:end,:])
    axpy!(α, component(dx, d)[(lup,ldn),(rup,rdn)], @view X[  rank(dx,d,(lup,ldn))+1:end,:])
  end

  return round!(cores2tensor(cores), dx.r) # No chopping.
end

"""
    projectiveTransport(dx::TTtangent{T,Nup,Ndn,d}, y::TTvector{T,Nup,Ndn,d})

Compute the vector transport of a tangent vector `dx` ``\\in T_x(M_r) \\to T_y(M_r)``
"""
function projectiveTransport(y::TTvector{T,Nup,Ndn,d}, dx::TTtangent{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  z = TTvector(dx)
  return TTtangent(y, z)
end
