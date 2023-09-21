using LinearAlgebra, OffsetArrays

"""
  Block{T<:Number} <: AbstractMatrix{T}

Elementary structure holding data for an `m×n` block in a Sparse Tensor Train core.
The structure allows for scaling without data copy (for use in matrix-free assembly)
as well as implicit zero blocks being represented with uninitialized `factor` and `data` fields.

Care must be taken to check during use using the `isnonzero(a::Block)` function to avoid 
undefined references when using low-level `factor(b)` and `data(b)` field accessor functions.
"""
mutable struct Block{T<:Number} <: AbstractMatrix{T}
  const m::Int
  const n::Int
  factor::T
  const data::Matrix{T}

  Block{T}(factor::Number, data::Matrix{T}) where {T<:Number} = new(size(data,1),size(data,2),T(factor),data)
  Block{T}(data::Matrix{T}) where {T<:Number}                 = new(size(data,1),size(data,2),T(1),data)
  Block{T}(m,n) where {T<:Number}                             = new(m,n)
  Block{T}(m,n,factor,data) where {T<:Number}                 = new(m,n,factor,data)
end

Block(factor::Number, data::Matrix{T}) where {T<:Number} = Block{T}(factor, data)
Block(data::Matrix{T}) where {T<:Number} = Block{T}(data)

iszero(a::Block) = !isdefined(a,:data)
isnonzero(a::Block) = isdefined(a,:data)

function Base.convert(::Type{Block{T}}, b::Tuple{Number,Matrix{T}}) where T<:Number
  return Block(b[1],b[2])
end

function Base.convert(::Type{Block{T}}, b::Matrix{T}) where T<:Number
  if length(b) == 0
    return Block{T}(size(b,1),size(b,2))
  else
    return Block(b)
  end
end

function Base.copy(a::Block{T}) where T<:Number
  if isnonzero(a)
    return Block{T}(a.m,a.n,a.factor,a.data)
  else
    return Block{T}(a.m,a.n)
  end
end

function ones_block(::Type{T}, m::Int, n::Int) where T<:Number
  return Block{T}(m,n,T(1),ones(T,m,n))
end

function factor(a::Block)
  return a.factor
end

function data(a::Block{T}) where T<:Number
  return a.data
end

function Base.size(a::Block)
  return (a.m,a.n)
end

function Base.axes(a::Block)
  return (Base.OneTo(a.m),Base.OneTo(a.n))
end

function Base.Array(b::Block{T}) where T<:Number
  if isnonzero(b)
    return factor(b).*data(b)
  else
    return zeros(T,b.m,b.n)
  end
end

function Base.getindex(a::Block{T}, i::Int, j::Int) where T<:Number
  if isnonzero(a)
    return factor(a).*data(a)[i,j]
  else
    return T(0)
  end
end

function Base.getindex(b::Block{T}, I::AbstractRange, J::AbstractRange) where T<:Number
  return Block{T}(b.m,b.n,b.factor, data(b)[I,J])
end

function Base.setindex!(x::AbstractMatrix{T}, b::Block{T}, I...) where T<:Number
  if isnonzero(b)
    setindex!(x, data(b), I...)
    lmul!(factor(b), view(x, I...))
    return b
  else
    x[I...] .= 0
    return x
  end
end

function zeros_block(::Type{T}, m::Int, n::Int) where T<:Number
  return Block{T}(m,n)
end

function LinearAlgebra.axpy!(α::Number, x::Block{T}, y::Block{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  if isnonzero(x)
    axpby!(α*factor(x), data(x), factor(y), data(y))
    y.factor = T(1)
  end
  return y
end

function LinearAlgebra.axpby!(α::Number, x::Block{T}, β::Number, y::Block{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  if isnonzero(x)
    axpby!(α*factor(x), data(x), β*factor(y), data(y))
    y.factor = T(1)
  else
    lmul!(β,y)
  end
  return y
end


function LinearAlgebra.axpy!(α::Number, x::Block{T}, y::AbstractMatrix{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  if isnonzero(x)
    axpy!(α*factor(x),data(x),y)
  end
  return y
end

function LinearAlgebra.axpby!(α::Number, x::Block{T}, β::Number, y::AbstractMatrix{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  if isnonzero(x)
    axpby!(α*factor(x),data(x),β,y)
  else
    lmul!(β,y)
  end
  return y
end

function Base.:+(x::Block{T}, y::Block{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  if isnonzero(x) && isnonzero(y)
    return Block(factor(x).*data(x).+factor(y).*data(y))
  elseif isnonzero(x)
    return deepcopy(x)
  else
    return deepcopy(y)
  end
end

function Base.:+(x::Block{T}, β::Number) where T<:Number
  if isnonzero(x)
    return Block(factor(x).*data(x).+β)
  else
    return Block(β,ones(T,x.m,x.n))
  end
end

function Base.:+(α::Number, y::Block{T}) where T<:Number
  return y + α
end


function LinearAlgebra.lmul!(α::Number, b::Block{T}) where T<:Number
  b.factor *= T(α)
  return b
end

function LinearAlgebra.rmul!(a::Block{T}, β::Number) where T<:Number
  a.factor *= T(β)
  return a
end

function LinearAlgebra.:*(α::Number, b::Block)
  c = deepcopy(b)
  return lmul!(α, c)
end

function LinearAlgebra.:*(a::Block, β::Number)
  c = deepcopy(a)
  return rmul!(c, β)
end

function LinearAlgebra.mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::Block{T}, α::Number=1, β::Number=0) where T<:Number
  if isnonzero(B)
    return mul!(C, A, data(B), α*factor(B), β)
  else
    return rmul!(C,β)
  end
end

function LinearAlgebra.mul!(C::AbstractMatrix{T}, A::Block{T}, B::AbstractMatrix{T}, α::Number=1, β::Number=0) where T<:Number
  if isnonzero(A)
    mul!(C, data(A), B, α*factor(A), β)
  else
    rmul!(C,β)
  end
  return C
end

function LinearAlgebra.mul!(C::Block{T}, A::AbstractMatrix{T}, B::Block{T}, α::Number=1, β::Number=0) where T<:Number
  if isnonzero(B)
    mul!(data(C), A, data(B), α*factor(B), β*factor(C))
    C.factor = T(1) 
  else
    rmul!(C,β)
  end
  return C
end

function LinearAlgebra.mul!(C::Block{T}, A::Block{T}, B::AbstractMatrix{T}, α::Number=1, β::Number=0) where T<:Number
  if isnonzero(A)
    mul!(C.array, data(A), B, α*factor(A), β*factor(C))
    C.factor = T(1)
  else
    rmul!(C,β)
  end
  return C
end

function contract!(C::AbstractMatrix{T}, A::Block{T}, B::Block{T},α::Number=1,β::Number=0) where T<:Number
  if isnonzero(A) && isnonzero(B)
    return mul!(C, data(A),data(B),factor(A)*factor(B)*T(α),T(β))
  else
    return rmul!(C,β)
  end
end

function LinearAlgebra.:*(A::Union{Matrix{T},Diagonal{T}}, B::Block{T}) where T<:Number
  if isnonzero(B)
    return Block{T}(size(A,1),size(B,2),B.factor,A*data(B))
  else
    return zeros_block(T,size(A,1),size(B,2))
  end
end

function LinearAlgebra.:*(A::Block{T}, B::Union{Matrix{T},Diagonal{T}}) where T<:Number
  if isnonzero(A)
    return Block{T}(size(A,1),size(B,2),A.factor,data(A)*B)
  else
    return zeros_block(T,size(A,1),size(B,2))
  end
end

function LinearAlgebra.norm(b::Block{T}) where T<:Number
  if isnonzero(b)
    return norm(factor(b)) * norm(data(b))
  else
    return abs(T(0))
  end
end

function norm2(x::Block{T}) where T<:Number
  if isnonzero(x)
    return abs2(factor(x)) * sum(abs2, data(x))
  else
    return abs(T(0))
  end
end

function LinearAlgebra.dot(x::Block{T},y::Block{T}) where T<:Number
  if isnonzero(x) && isnonzero(y)
    return conj(factor(x))*factor(y) * dot(data(x), data(y))
  else
    return T(0)
  end
end

function ⊗(a::Block{T}, b::Block{T}) where T<:Number
  m = a.m*b.m
  n = a.n*b.n
  if isnonzero(a) && isnonzero(b)
    f = factor(a)*factor(b)
    d = reshape( 
            reshape(data(a), (size(a,1),1,size(a,2),1)) .* 
            reshape(data(b), (1,size(b,1),1,size(b,2))),
            (size(a,1)*size(b,1),size(a,2)*size(b,2))
                )
    return Block{T}(m,n,f,d)
  else
    return Block{T}(m,n)
  end
end
