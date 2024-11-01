using LinearAlgebra, OffsetArrays
"""
    TTtangent{T<:Number,N,d}

Implementation of tangent vectors of the manifold of constant TT-rank.
"""
mutable struct TTtangent{T<:Number,N,d}
  r::Vector{OffsetVector{Int,Vector{Int}}}

  baseL::TTvector{T,N,d}
  baseR::TTvector{T,N,d}
  components::Vector{SparseCore{T,N,d,Matrix{T}}}

  """
      TTtangent(baseL::TTvector{T,N,d}, baseR::TTvector{T,N,d}, direction::TTvector{T,N,d})

  Lazy constructor that does not make a copy of the base vector, but takes as input two versions of the TTtensor,
  `baseL` which should be left-orthogonal and `baseR` which should be right-orthogonal.

  No effort is made to check that the two TTtensors do in fact represent the same tensor, except for checking their modal size and ranks.
  """
  function TTtangent(baseL::TTvector{T,N,d}, baseR::TTvector{T,N,d}, direction::TTvector{T,N,d}) where {T<:Number,N,d}
    @assert rank(baseL) == rank(baseR)
    @assert rank(baseL,  1,0) == rank(direction,  1,0) == 1
    @assert rank(baseL,d+1,N) == rank(direction,d+1,N) == 1
    @assert baseR.orthogonal && baseR.corePosition == 1
    @assert baseL.orthogonal && baseL.corePosition == d

    # Compute the left projection matrices: Pi = U^T_{k-1} ... U^T_1 W_1 ... W_{k-1}, a block-diagonal r[k] × rank(direction,k) matrix
    P = Vector{Frame{T,N,d,Matrix{T}}}(undef, d)
    P[1] = IdFrame(Val(N), Val(d), 1)
    for k=1:d-1
      P[k+1] = (adjoint(core(baseL,k)) * P[k]) * core(direction,k)
    end

    # Compute the right projection matrices: Q = W_{i+1} ... W_d V^T_d ... V^T_{i+1}, a direction.r[i+1] × r[i+1] matrix
    # Assemble the components of the tangent vector for i=d,...,1
    Qₖ = IdFrame(Val(N), Val(d), d+1)
    components = Vector{SparseCore{T,N,d,Matrix{T}}}(undef, d)

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

    return new{T,N,d}(deepcopy(rank(baseL)),baseL,baseR,components)
  end
end

"""
    TTtangent(base::TTvector{T,N,d}, direction::TTvector{T,N,d})

Compute the tangent vector at `base` to the manifold of constant ranks `base.r` in the direction `direction`.
Makes two copies of the TTtensor `base`.
"""
function TTtangent(base::TTvector{T,N,d}, direction::TTvector{T,N,d}) where {T<:Number,N,d}
  # Create left-orthogonalized copy of base TTtensor U_1 ... U_d
  baseL = deepcopy(base)
  leftOrthogonalize!(baseL, keepRank=true)
  # Create right-orthogonalized copy of base TTtensor V_1 ... V_d
  baseR = deepcopy(base)
  rightOrthogonalize!(baseR, keepRank=true)

  return TTtangent(baseL, baseR, direction)
end

function component(dx::TTtangent{T,N,d}, k::Int) where {T<:Number,N,d}
  @boundscheck @assert 1 ≤ k ≤ d
  return dx.components[k]
end

function Base.show(io::IO, dx::TTtangent{T,N,d}) where {T<:Number,N,d}
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

function Base.:+(dx::TTtangent{T,N,d}, dy::TTtangent{T,N,d}) where {T<:Number,N,d}
  @assert rank(dx) == rank(dy)

  result = deepcopy(dx)
  for k=1:d
    component(result,k) .+= component(dy,k)
  end
  return result
end

function Base.:-(dx::TTtangent{T,N,d}, dy::TTtangent{T,N,d}) where {T<:Number,N,d}
  @assert rank(dx) == rank(dy)

  result = deepcopy(dx)
  for k=1:d
    component(result,k) .-= component(dy,k)
  end
  return result
end

LinearAlgebra.rank(dx::TTtangent) = dx.r
LinearAlgebra.rank(dx::TTtangent, k::Int) = dx.r[k]
LinearAlgebra.rank(dx::TTtangent, k::Int, n::Int) = dx.r[k][n]

function LinearAlgebra.axpby!(a::Number, dx::TTtangent{T,N,d}, b::Number, dy::TTtangent{T,N,d}) where {T<:Number,N,d}
  @assert rank(dx) == rank(dy)

  for k=1:d
    axpby!(T(a), dx.components[k], T(b), component(dy,k))
  end
  return dy
end

function LinearAlgebra.axpy!(a::Number, dx::TTtangent{T,N,d}, dy::TTtangent{T,N,d}) where {T<:Number,N,d}
  @assert rank(dx) == rank(dy)

  for k=1:d
    axpby!(T(a), dx.components[k], component(dy,k))
  end
  return dy
end

function LinearAlgebra.lmul!(a::Number, dx::TTtangent{T,N,d}) where {T<:Number,N,d}
  for k=1:d
    lmul!(a,component(dx,k))
  end
  return dx
end

function LinearAlgebra.rmul!(dx::TTtangent{T,N,d}, b::Number) where {T<:Number,N,d}
  for k=1:d
    rmul!(component(dx,k),b)
  end
  return dx
end

function Base.:*(a::Number, dx::TTtangent{T,N,d}) where {T<:Number,N,d}
  return lmul!(a, deepcopy(dx))
end

function Base.:*(dx::TTtangent{T,N,d}, b::Number) where {T<:Number,N,d}
  return rmul!(deepcopy(dx), b)
end

function Base.:/(dx::TTtangent{T,N,d}, b::Number) where {T<:Number,N,d}
  return rmul!(deepcopy(dx), inv(b))
end

"""
    TTtensor(dx::TTtangent{T,N,d})

Compute the TTtensor representation of a tangent vector.
"""
function TTvector(dx::TTtangent{T,N,d}) where {T<:Number,N,d}

  cores = Vector{SparseCore{T,N,d,Matrix{T}}}(undef,d)
  for k=1:d
    row_ranks = (k==1 ? 1 : 2 ) .* rank(dx,k)
    col_ranks = (k==d ? 1 : 2 ) .* rank(dx,k+1)
    cores[k] = SparseCore{T,N,d}(k, row_ranks, col_ranks)
  end

  for l in axes(cores[1],1), r in (l:l+1)∩axes(cores[1],3)
    cores[1][l,r][:, 1:rank(dx,2,r)    ] = component(dx, 1)[l,r]
    cores[1][l,r][:, rank(dx,2,r)+1:end] = core(dx.baseL,1)[l,r]
  end

  for k=2:d-1
    for l in axes(cores[k],1), r in (l:l+1)∩axes(cores[k],3)
      cores[k][l,r][1:rank(dx,k)[l],    1:rank(dx,k+1)[r]    ] = core(dx.baseR,k)[l,r]
      cores[k][l,r][rank(dx,k)[l]+1:end,1:rank(dx,k+1)[r]    ] = component(dx, k)[l,r]
      cores[k][l,r][rank(dx,k)[l]+1:end,rank(dx,k+1)[r]+1:end] = core(dx.baseL,k)[l,r]
    end
  end

  for l in axes(cores[d],1), r in (l:l+1)∩axes(cores[d],3)
    cores[d][l,r][1:rank(dx,d)[l],    :] = core(dx.baseR,d)[l,r]
    cores[d][l,r][rank(dx,d)[l]+1:end,:] = component(dx, d)[l,r]
  end

  return cores2tensor(cores)
end

function LinearAlgebra.dot(dx::TTtangent{T,N,d}, dy::TTtangent{T,N,d}) where {T<:Number,N,d}
  @assert rank(dx) == rank(dy)
  return sum(k->dot(dx.components[k], component(dy,k)), 1:d)
end

function LinearAlgebra.norm(dx::TTtangent{T,N,d}) where {T<:Number,N,d}
  return sqrt(sum(k->norm2(dx.components[k]), 1:d))
end

"""
    retractHOSVD(dx::TTtangent{T,N,d}, α::Number=one(T))

Compute a retraction of `dx.base` + `α`⋅`dx` onto the manifold with ranks 'dx.r' using the HOSVD based rounding.
"""
function retractHOSVD(dx::TTtangent{T,N,d}, α::Number=one(T)) where {T<:Number,N,d}
  cores = [SparseCore{T,N,d}(
                  k, 
                  (k==1 ? 1 : 2 ) .* rank(dx,k), 
                  (k==d ? 1 : 2 ) .* rank(dx,k+1)
                            ) for k=1:d]

  for l in axes(cores[1],1), r in (l:l+1)∩axes(cores[1],3)
    X = cores[1][l,r]
    axpy!(α, component(dx, 1)[l,r], @view X[:, 1:rank(dx,2,r)    ])
    axpy!(1, core(dx.baseL,1)[l,r], @view X[:, rank(dx,2,r)+1:end])
  end

  for k=2:d-1
    for l in axes(cores[k],1), r in (l:l+1)∩axes(cores[k],3)
      X = cores[k][l,r]
      axpy!(1, core(dx.baseR,k)[l,r], @view X[1:rank(dx,k)[l],    1:rank(dx,k+1)[r]    ])
      axpy!(α, component(dx, k)[l,r], @view X[rank(dx,k)[l]+1:end,1:rank(dx,k+1)[r]    ])
      axpy!(1, core(dx.baseL,k)[l,r], @view X[rank(dx,k)[l]+1:end,rank(dx,k+1)[r]+1:end])
    end
  end

  for l in axes(cores[d],1), r in (l:l+1)∩axes(cores[d],3)
    X = cores[d][l,r]
    axpy!(1, core(dx.baseR,d)[l,r], @view X[1:rank(dx,d)[l],    :])
    axpy!(1, core(dx.baseL,d)[l,r], @view X[rank(dx,d)[l]+1:end,:])
    axpy!(α, component(dx, d)[l,r], @view X[rank(dx,d)[l]+1:end,:])
  end

  return round!(cores2tensor(cores), dx.r) # No chopping.
end

"""
    projectiveTransport(dx::TTtangent{T,N,d}, y::TTvector{T,N,d})

Compute the vector transport of a tangent vector `dx` ``\\in T_x(M_r) \\to T_y(M_r)``
"""
function projectiveTransport(y::TTvector{T,N,d}, dx::TTtangent{T,N,d}) where {T<:Number,N,d}
  z = TTvector(dx)
  return TTtangent(y, z)
end
