function LinearAlgebra.lmul!(a::Number, tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  if tt.orthogonal
    lmul!(a, core(tt,tt.corePosition))
  else
    lmul!(a, core(tt,1))
  end
  return tt
end

function LinearAlgebra.rmul!(tt::TTvector{T,Nup,Ndn,d}, b::Number) where {T<:Number,Nup,Ndn,d}
  if tt.orthogonal
    rmul!(core(tt,tt.corePosition),b)
  else
    rmul!(core(tt,d),b)
  end
  return tt
end

function LinearAlgebra.lmul!(A::Matrix{Number}, tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @assert rank(tt,1,0) .== size(A,2)

  tt.r[1] .= size(A,1)
  lmul!(A, core(tt,1))

  if tt.orthogonal && tt.corePosition != 1
    tt.orthogonal = false
  end
  return tt
end

function LinearAlgebra.rmul!(tt::TTvector{T,Nup,Ndn,d}, B::Matrix{Number}) where {T<:Number,Nup,Ndn,d}
  @assert rank(tt,d+1,N) == size(B,1)

  tt.r[d+1] .= size(B,2)
  rmul!(core(tt,d),B)

  if tt.orthogonal && tt.corePosition != d
    tt.orthogonal = false
  end
  return tt
end

function Base.:*(a::Number, tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return lmul!(a, deepcopy(tt))
end

function Base.:*(tt::TTvector{T,Nup,Ndn,d}, b::Number) where {T<:Number,Nup,Ndn,d}
  return rmul!(deepcopy(tt), b)
end

function Base.:/(tt::TTvector{T,Nup,Ndn,d}, b::Number) where {T<:Number,Nup,Ndn,d}
  return rmul!(deepcopy(tt), inv(b))
end

function Base.:*(A::Matrix{Number}, tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return lmul!(A, deepcopy(tt))
end

function Base.:*(tt::TTvector{T,Nup,Ndn,d}, B::Matrix{Number}) where {T<:Number,Nup,Ndn,d}
  return rmul!(deepcopy(tt), B)
end

"""
    times(b::TTvector{T,Nup,Ndn,d}, c::TTvector{T,Nup,Ndn,d})

Compute the Hadamard product of `a` and `b`
"""
function times(b::TTvector{T,Nup,Ndn,d}, c::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}

  r = [rank(b,k).*rank(c,k) for k=1:d+1]
  cores = [core(b,k)⊗core(c,k) for k = 1:d]

  return TTvector(r, cores)
end

"""
    power(b::TTvector{T,Nup,Ndn,d}, n::Int)

Compute the elementwise power of `a`.
"""
function power(a::TTvector{T,Nup,Ndn,d}, n::Int) where {T<:Number,Nup,Ndn,d}
  @assert n ≥ 0
  if n == 0
    return tt_ones(a.n, T)
  else
    c = deepcopy(a)
    for i=2:n
      c=times(c,a)
    end
    return c
  end
end