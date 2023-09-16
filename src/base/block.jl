using LinearAlgebra, OffsetArrays

abstract type Block{T} <: AbstractMatrix{T} end

struct ZeroBlock{T<:Number} <: Block{T}
  m::Int
  n::Int
end

mutable struct NonZeroBlock{T<:Number} <: Block{T}
  factor::T
  const data::Matrix{T}
end

function Block(factor::Number, data::Matrix{T}) where T<:Number
  if factor == 0
    return ZeroBlock{T}(size(data,1),size(data,2))
  else
    return NonZeroBlock{T}(T(factor),data)
  end
end

function Block(data::Matrix{T}) where T<:Number
  return NonZeroBlock{T}(T(1),data)
end

iszero(a::ZeroBlock) = true
iszero(a::NonZeroBlock) = false
isnonzero(a::ZeroBlock) = false
isnonzero(a::NonZeroBlock) = true

function Base.convert(::Type{Block{T}}, b::Tuple{Number,Matrix{T}}) where T<:Number
  return Block(b[1],b[2])
end

function Base.convert(::Type{Block{T}}, b::Matrix{T}) where T<:Number
  if length(b) == 0
    return ZeroBlock{T}(size(b,1),size(b,2))
  else
    return NonZeroBlock{T}(T(1),b)
  end
end

function Base.copy(a::ZeroBlock{T}) where T<:Number
  return a
end

function Base.copy(a::NonZeroBlock{T}) where T<:Number
  return NonZeroBlock{T}(a.factor, a.data)
end

function ones_block(::Type{T}, m::Int, n::Int) where T<:Number
  return NonZeroBlock{T}(T(1), ones(T,m,n))
end

function factor(a::NonZeroBlock)
  return a.factor
end

function data(a::NonZeroBlock)
  return a.data
end

function Base.size(a::NonZeroBlock)
  return size(a.data)
end

function Base.axes(a::NonZeroBlock)
  return axes(a.data)
end

function Base.Array(b::NonZeroBlock{T}) where T<:Number
  return factor(b).*data(b)
end

function Base.getindex(a::NonZeroBlock{T}, i::Int, j::Int) where T<:Number
  return factor(a).*data(a)[i,j]
end

function Base.getindex(b::NonZeroBlock{T}, I::AbstractRange, J::AbstractRange) where T<:Number
  return NonZeroBlock{T}(b.factor, b.data[I,J])
end

function Base.setindex!(x::AbstractMatrix{T}, b::NonZeroBlock{T}, I...) where T<:Number
  setindex!(x, data(b), I...)
  lmul!(factor(b), view(x, I...))
  return b
end

function zeros_block(::Type{T}, m::Int, n::Int) where T<:Number
  return ZeroBlock{T}(m,n)
end

function Base.size(a::ZeroBlock)
  return (a.m,a.n)
end

function Base.axes(a::ZeroBlock)
  return (1:a.m,1:a.n)
end

function Base.Array(b::ZeroBlock{T}) where T<:Number
  return zeros(T,b.m,b.n)
end

function Base.getindex(a::ZeroBlock{T}, i::Int, j::Int) where T<:Number
  return T(0)
end

function Base.getindex(b::ZeroBlock{T}, I::AbstractRange, J::AbstractRange) where T<:Number
    return ZeroBlock{T}(length(I),length(J))
end

function Base.setindex!(x::AbstractMatrix{T}, b::ZeroBlock{T}, I...) where T<:Number
  x[I...] .= 0
  return x
end

function LinearAlgebra.norm(b::ZeroBlock{T}) where T<:Number
  return norm(T(0))
end

function LinearAlgebra.norm(b::NonZeroBlock{T}) where T<:Number
  return norm(factor(b)) * norm(data(b))
end

function LinearAlgebra.lmul!(α::Number, b::ZeroBlock{T}) where T<:Number
  return b
end

function LinearAlgebra.rmul!(a::ZeroBlock{T}, β::Number) where T<:Number
  return a
end

function LinearAlgebra.lmul!(α::Number, b::NonZeroBlock{T}) where T<:Number
  b.factor *= T(α)
  return b
end

function LinearAlgebra.rmul!(a::NonZeroBlock{T}, β::Number) where T<:Number
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

function LinearAlgebra.mul!(C::NonZeroBlock{T}, A::AbstractMatrix{T}, B::ZeroBlock{T}, α::Number, β::Number) where T<:Number
  rmul!(C,β)
  return C
end

function LinearAlgebra.mul!(C::NonZeroBlock{T}, A::AbstractMatrix{T}, B::NonZeroBlock{T}, α::Number, β::Number) where T<:Number
  mul!(C.data, A, data(B), α*factor(B), β*factor(C))
  C.factor = T(1) 
  return C
end
function LinearAlgebra.mul!(C::NonZeroBlock{T}, A::ZeroBlock{T}, B::AbstractMatrix{T}, α::Number, β::Number) where T<:Number
  rmul!(C,β)
  return C
end

function LinearAlgebra.mul!(C::NonZeroBlock{T}, A::NonZeroBlock{T}, B::AbstractMatrix{T}, α::Number, β::Number) where T<:Number
  mul!(C.array, data(A), B, α*factor(A), β*factor(C))
  C.factor = T(1)
  return C
end

function LinearAlgebra.:*(A::Union{Matrix{T},Diagonal{T}}, B::ZeroBlock{T}) where T<:Number
  return ZeroBlock{T}(size(A,1), size(B,2))
end

function LinearAlgebra.:*(A::Union{Matrix{T},Diagonal{T}}, B::NonZeroBlock{T}) where T<:Number
    return NonZeroBlock{T}(B.factor,A*B.data)
end

function LinearAlgebra.:*(A::ZeroBlock{T}, B::Union{Matrix{T},Diagonal{T}}) where T<:Number
  return ZeroBlock{T}(size(A,1), size(B,2))
end

function LinearAlgebra.:*(A::NonZeroBlock{T}, B::Union{Matrix{T},Diagonal{T}}) where T<:Number
  return NonZeroBlock{T}(A.factor,A.data*B)
end


function LinearAlgebra.axpy!(α::Number, x::ZeroBlock{T}, y::NonZeroBlock{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  return y
end

function LinearAlgebra.axpy!(α::Number, x::NonZeroBlock{T}, y::NonZeroBlock{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  lmul!(factor(y), data(y))
  axpy!(α*factor(x), data(x), data(y))
  y.factor = T(1)
  return y
end

function LinearAlgebra.axpby!(α::Number, x::ZeroBlock{T}, β::Number, y::NonZeroBlock{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  lmul!(β,y)
  return y
end

function LinearAlgebra.axpby!(α::Number, x::NonZeroBlock{T}, β::Number, y::NonZeroBlock{T}) where T<:Number
  axpby!(α*factor(x), data(x), β*factor(y), data(y))
  y.factor = T(1)
  return y
end

function Base.:+(x::ZeroBlock{T}, y::ZeroBlock{T}) where T<:Number
  @boundscheck @assert x.m == y.m && x.n == y.n
  return ZeroBlock{T}(x.m, x.n)
end

function Base.:+(x::ZeroBlock{T}, y::NonZeroBlock{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  return deepcopy(y)
end

function Base.:+(x::NonZeroBlock{T}, y::ZeroBlock{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  return deepcopy(x)
end

function Base.:+(x::NonZeroBlock{T}, y::NonZeroBlock{T}) where T<:Number
  @boundscheck @assert size(x) == size(y)
  return NonZeroBlock{T}(T(1),factor(x).*data(x).+factor(y).*data(y))
end

function Base.:+(x::ZeroBlock{T}, β::Number) where T<:Number
  return NonZeroBlock{T}(T(β), ones(T,x.m,x.n))
end

function Base.:+(x::NonZeroBlock{T}, β::Number) where T<:Number
  return NonZeroBlock{T}(T(1), factor(x).*data(x).+β)
end

function Base.:+(α::Number, y::Block{T}) where T<:Number
  return y + α
end

function norm2(x::ZeroBlock{T}) where T<:Number
  return abs(T(0))
end

function norm2(x::NonZeroBlock{T}) where T<:Number
  return abs2(factor(x)) * sum(abs2, data(x)[:])
end

function LinearAlgebra.dot(x::ZeroBlock{T},y::ZeroBlock{T}) where T<:Number
  return T(0)
end
function LinearAlgebra.dot(x::NonZeroBlock{T},y::ZeroBlock{T}) where T<:Number
  return T(0)
end
function LinearAlgebra.dot(x::ZeroBlock{T},y::NonZeroBlock{T}) where T<:Number
  return T(0)
end
function LinearAlgebra.dot(x::NonZeroBlock{T},y::NonZeroBlock{T}) where T<:Number
  return conj(factor(x))*factor(y) * dot(x.data[:], y.data[:])
end

function ⊗(a::ZeroBlock{T}, b::ZeroBlock{T}) where T<:Number
  return ZeroBlock{T}(a.m*b.m,a.n*b.n)
end

function ⊗(a::ZeroBlock{T}, b::NonZeroBlock{T}) where T<:Number
  return ZeroBlock{T}(a.m*size(b,1),a.n*size(b,2))
end

function ⊗(a::NonZeroBlock{T}, b::ZeroBlock{T}) where T<:Number
  return ZeroBlock{T}(size(a,1)*size(b,1),size(a,2)*size(b,2))
end

function ⊗(a::NonZeroBlock{T}, b::NonZeroBlock{T}) where T<:Number
  f = factor(a)*factor(b)
  d = reshape( 
          reshape(data(a), (size(a,1),1,size(a,2),1)) .* 
          reshape(data(b), (1,size(b,1),1,size(b,2))),
          (size(a,1)*size(b,1),size(a,2)*size(b,2))
              )
  return NonZeroBlock{T}(f, d)
end
