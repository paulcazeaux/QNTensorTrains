using LinearAlgebra

function Base.:+(b::TTvector{T,N,d}, c::TTvector{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert rank(b,1,0) == rank(c,1,0)
    @assert rank(b,d+1,N) == rank(c,d+1,N)
  end

  r = rank(b).+rank(c)
  r[1] .= rank(b,1)
  r[d+1] .= rank(b,d+1)

  cores = [core(b,k)⊕core(c,k) for k=1:d]
  a = TTvector(r, cores)

  return a
end

function Base.:+(b::TTvector{T,N,d}, c::Number) where {T<:Number,N,d}
  r = rank(b).+1
  r[1][0] = rank(b,1,0)
  r[d+1][N] = rank(b,d+1,N)

  cores = [core(b,k)⊕c for k=1:d]
  a = TTvector{T,N,d}(r, cores)

  return a
end

Base.:+(b::Number, c::TTvector) = c+b

function Base.:+(a::TTvector)
  return deepcopy(a)
end

function Base.:-(a::TTvector)
  return (-1)*a
end

function Base.:-(b::TTvector{T,N,d}, c::TTvector{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert rank(b,1,0) == rank(c,1,0)
    @assert rank(b,d+1,N) == rank(c,d+1,N)
  end
  
  r = [rank(b,k) .+ rank(c,k) for k=1:d+1]
  r[1][0] = rank(b,1,0)
  r[d+1][N] = rank(b,d+1,N)

  cores = [core(b,k)⊕( k>1 ? core(c,k) : -core(c,1)) for k=1:d]
  a = TTvector{T,N,d}(r, cores)

  return a
end

function Base.:-(b::TTvector{T,N,d}, c::Number) where {T<:Number,N,d}
  return b+(-c)
end

function Base.:-(b::Number, c::TTvector{T,N,d}) where {T<:Number,N,d}
  r = [rank(b,k).+1 for k=1:d+1]
  r[  1][0] = rank(b,  1,0)
  r[d+1][N] = rank(b,d+1,N)

  cores = [b⊕(k > 1 ? core(c,k) : -core(c,1)) for k=1:d]
  a = TTvector{T,N,d}(r, cores)

  return a
end

"""
    sum(tt::TTvector{T,N,d})

Sum all elements in `tt`.
"""
function Base.sum(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  e = tt_ones(size(tt),T)
  return dot(e,tt)
end
