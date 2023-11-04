using LinearAlgebra, OffsetArrays
"""
    TTtangent{T<:Number,N,d}

Implementation of tangent vectors of the manifold of constant TT-rank.
"""
mutable struct TTtangent{T<:Number,N,d}
  r::Vector{OffsetVector{Int,Vector{Int}}}

  baseL::TTvector{T,N,d}
  baseR::TTvector{T,N,d}
  components::Vector{SparseCore{T,N,d}}

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

    r = deepcopy(rank(baseL))

    # Compute the left projection matrices: Pi = U^T_{k-1} ... U^T_1 W_1 ... W_{k-1}, a block-diagonal r[k] × rank(direction,k) matrix
    P = Vector{OffsetVector{Matrix{T},Vector{Matrix{T}}}}(undef, d)
    P[1] = [n == 0 ? ones(T,1,1) : zeros(T,0,0) for n in axes(core(baseL,1),1)]
    for k=1:d-1
      Xₖ = core(baseL,k)
      Dₖ = core(direction,k)

      P[k+1] = [zeros(T,Xₖ.col_ranks[r],Dₖ.col_ranks[r]) for r in axes(Xₖ,3)]
      for r in axes(Xₖ,3), l in (r-1:r)∩axes(Xₖ,1)
        if isnonzero(Xₖ,l,r) && isnonzero(Dₖ,l,r)
          mul!( P[k+1][r], 
                adjoint(data(Xₖ[l,r])) * P[k][l],
                data(Dₖ[l,r]),
                conj(factor(Xₖ[l,r])) * factor(Dₖ[l,r]),
                T(1)
              )
        end
      end
    end

    # Compute the right projection matrices: Q = W_{i+1} ... W_d V^T_d ... V^T_{i+1}, a direction.r[i+1] × r[i+1] matrix
    # Assemble the components of the tangent vector for i=d,...,1
    Qₖ = [l == N ? ones(T,1,1) : zeros(T,0,0) for l in axes(core(baseR,d),3)]
    components = Vector{SparseCore{T,N,d}}(undef, d)
    
    for k=d:-1:1
      Xₖ = core(baseR,k)
      Dₖ = core(direction,k)

      tmp = Dₖ * Qₖ
      components[k] = P[k] * tmp
      if k < d
        # Apply gauge condition
        for r in axes(components[k],3)
          L = Array(core(baseL,k)[r,:vertical])
          components[k][r,:vertical] = (I - L * adjoint(L)) * components[k][r,:vertical]
        end
      end

      Qₖ = [zeros(T,Dₖ.row_ranks[l],Xₖ.row_ranks[l]) for l in axes(Xₖ,1)]
      for l in axes(Xₖ,1), r in (l:l+1)∩axes(Xₖ,3)
        if isnonzero(Xₖ,l,r) && isnonzero(tmp,l,r)
          mul!( Qₖ[l], 
                data(tmp[l,r]),
                adjoint(data(Xₖ[l,r])),
                factor(tmp[l,r]) * conj(factor(Xₖ[l,r])),
                T(1)
              )
        end
      end
    end

    return new{T,N,d}(r,baseL,baseR,components)
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

  cores = [SparseCore{T,N,d}(k) for k=1:d]
  for k=1:d
    cores[k].row_ranks .= (k==1 ? 1 : 2 ) .* rank(dx,k)
    cores[k].col_ranks .= (k==d ? 1 : 2 ) .* rank(dx,k+1)
  end

  for l in axes(cores[1],1), r in (l:l+1)∩axes(cores[1],3)
    if isnonzero(component(dx,1),l,r) || isnonzero(core(dx.baseL,1),l,r)
      X = zeros(T,rank(dx,1,l),2*rank(dx,2,r))
      X[:, 1:rank(dx,2,r)    ] = component(dx, 1)[l,r]
      X[:, rank(dx,2,r)+1:end] = core(dx.baseL,1)[l,r]
    else
      X = zeros_block(T,rank(dx,1,l),2*rank(dx,2,r))
    end
    cores[1][l,r] = X
  end

  for k=2:d-1
    for l in axes(cores[k],1), r in (l:l+1)∩axes(cores[k],3)
      if isnonzero(core(dx.baseR, k),l,r) || isnonzero(component(dx,k),l,r) || isnonzero(core(dx.baseL,k),l,r)
        X = zeros(T,2*rank(dx,k,l),2*rank(dx,k+1,r))
        X[1:rank(dx,k)[l],    1:rank(dx,k+1)[r]    ] = core(dx.baseR,k)[l,r]
        X[rank(dx,k)[l]+1:end,1:rank(dx,k+1)[r]    ] = component(dx, k)[l,r]
        X[rank(dx,k)[l]+1:end,rank(dx,k+1)[r]+1:end] = core(dx.baseL,k)[l,r]
      else
        X = zeros_block(T,2*rank(dx,k,l),2*rank(dx,k+1,r))
      end
      cores[k][l,r] = X
    end
  end

  for l in axes(cores[d],1), r in (l:l+1)∩axes(cores[d],3)
    if isnonzero(core(dx.baseR,d),l,r) || isnonzero(component(dx,d),l,r)
      X = zeros(T,2*rank(dx,d,l),rank(dx,d+1,r))
      X[1:rank(dx,d)[l],    :] = core(dx.baseR,d)[l,r]
      X[rank(dx,d)[l]+1:end,:] = component(dx, d)[l,r]
    else
      X = zeros_block(T,2*rank(dx,d,l),rank(dx,d+1,r))
    end
    cores[d][l,r] = X
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
  cores = [SparseCore{T,N,d}(k) for k=1:d]
  for k=1:d
    cores[k].row_ranks .= (k==1 ? 1 : 2 ) .* rank(dx,k)
    cores[k].col_ranks .= (k==d ? 1 : 2 ) .* rank(dx,k+1)
  end

  for l in axes(cores[1],1), r in (l:l+1)∩axes(cores[1],3)
    if isnonzero(component(dx,1),l,r) || isnonzero(core(dx.baseL,1),l,r)
      X = zeros(T,rank(dx,1,l),2*rank(dx,2,r))
      axpy!(α, component(dx, 1)[l,r], @view X[:, 1:rank(dx,2,r)    ])
      axpy!(1, core(dx.baseL,1)[l,r], @view X[:, rank(dx,2,r)+1:end])
    else
      X = zeros_block(T,rank(dx,1,l),2*rank(dx,2,r))
    end
    cores[1][l,r] = X
  end

  for k=2:d-1
    for l in axes(cores[k],1), r in (l:l+1)∩axes(cores[k],3)
      if isnonzero(core(dx.baseR, k),l,r) || isnonzero(component(dx,k),l,r) || isnonzero(core(dx.baseL,k),l,r)
        X = zeros(T,2*rank(dx,k,l),2*rank(dx,k+1,r))
        axpy!(1, core(dx.baseR,k)[l,r], @view X[1:rank(dx,k)[l],    1:rank(dx,k+1)[r]    ])
        axpy!(α, component(dx, k)[l,r], @view X[rank(dx,k)[l]+1:end,1:rank(dx,k+1)[r]    ])
        axpy!(1, core(dx.baseL,k)[l,r], @view X[rank(dx,k)[l]+1:end,rank(dx,k+1)[r]+1:end])
      else
        X = zeros_block(T,2*rank(dx,k,l),2*rank(dx,k+1,r))
      end
      cores[k][l,r] = X
    end
  end

  for l in axes(cores[d],1), r in (l:l+1)∩axes(cores[d],3)
    if isnonzero(core(dx.baseR,d),l,r) && isnonzero(component(dx,d),l,r)
      X = zeros(T,2*rank(dx,d,l),rank(dx,d+1,r))
      axpy!(1, core(dx.baseR,d)[l,r], @view X[1:rank(dx,d)[l],    :])
      axpy!(1, core(dx.baseL,d)[l,r], @view X[rank(dx,d)[l]+1:end,:])
      axpy!(α, component(dx, d)[l,r], @view X[rank(dx,d)[l]+1:end,:])
    else
      X = zeros_block(T,2*rank(dx,d,l),rank(dx,d+1,r))
    end
    cores[d][l,r] = X
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
