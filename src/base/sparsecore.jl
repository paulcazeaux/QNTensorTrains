using LinearAlgebra, OffsetArrays

"""
  SparseCore{T<:Number,N,d,k}

Special bidiagonal sparse structure,
where diagonal  correspond to modal index = 1, 
and  just above the diagonal correspond to modal index 2

N is the total number of electrons and d is the overall tensor order; dictates structure
"""
struct SparseCore{T<:Number,N,d} <: AbstractArray{Matrix{T},3}
  k::Int        # Core index

  m::Int        # row size
  n::Int        # column size

  row_qn::OffsetArrays.IdOffsetRange{Int64,UnitRange{Int64}}
  col_qn::OffsetArrays.IdOffsetRange{Int64,UnitRange{Int64}}

  row_ranks::OffsetVector{Int, Vector{Int}}
  col_ranks::OffsetVector{Int, Vector{Int}}

  unoccupied::OffsetVector{Block{T}, Vector{Block{T}}}
  occupied::OffsetVector{Block{T}, Vector{Block{T}}}
end

function occupation_qn(N::Int, d::Int, k::Int)
  @boundscheck begin
    N ≤ d || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
    1 ≤ k ≤ d+1 || throw(BoundsError())
  end
  qn = min(max(N+k-1-d, 0), N):min(k-1, N)
  return OffsetArrays.IdOffsetRange(values=qn, indices=qn)
end

"""
  SparseCore{T,N,d}(k::Int)

Sparse Core initialization to zero

"""
function SparseCore{T,N,d}(k::Int) where {T<:Number,N,d}
  @boundscheck begin
    N ≤ d || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
    1 ≤ k ≤ d || throw(BoundsError())
  end
  m = min(k-1, N) - min(max(N+k-1-d, 0), N) + 1
  n = min(k  , N) - min(max(N+k  -d, 0), N) + 1

  row_qn = occupation_qn(N,d,k)
  col_qn = occupation_qn(N,d,k+1)

  row_ranks = OffsetVector(k > 1 ? zeros(Int,m) : [1], row_qn)
  col_ranks = OffsetVector(k < d ? zeros(Int,n) : [1], col_qn)

  qn = row_qn ∩ col_qn
  unoccupied = OffsetVector(zeros_block.(T, row_ranks[qn], col_ranks[qn   ]), qn)

  qn = row_qn ∩ (col_qn.-1)
  occupied   = OffsetVector(zeros_block.(T, row_ranks[qn], col_ranks[qn.+1]), qn)
  C = SparseCore{T,N,d}(k,m,n,row_qn,col_qn,row_ranks,col_ranks,unoccupied,occupied)
  return C
end

"""
  SparseCore{T,N,d}(s::Bool, Nl::Int, k::Int)

Core initialization for a pure, non-entangled state
  where s ∈ {true, false} denotes occupation or not,
  Nl is the number of occupied states on cores 1...k-1,
  N is the total number of particles
"""
function SparseCore{T,N,d}(s::Bool, Nl::Int, k::Int) where {T<:Number,N,d}
  @boundscheck begin
    Nl < k          || throw(DimensionMismatch("Number of occupied states on cores 1...$(k-1) cannot be larger than $(k-1)"))
    N-Nl-s ≤ (d-k)  || throw(DimensionMismatch("Number of occupied states on cores $(k+1)...$d cannot be larger than $(d-k)"))
    Nl ≤ N          || throw(DimensionMismatch("Number of occupied states on cores 1...$(k-1) cannot be larger than total number of electrons $N"))
    N ≤ d           || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
    1 ≤ k ≤ d       || throw(BoundsError())
  end

  m = min(k-1, N) - max(N+k-1-d, 0) + 1
  n = min(k  , N) - max(N+k  -d, 0) + 1

  row_qn = occupation_qn(N,d,k)
  col_qn = occupation_qn(N,d,k+1)

  row_ranks = OffsetVector(Int.(row_qn .== Nl  ), row_qn)
  col_ranks = OffsetVector(Int.(col_qn .== Nl+s), col_qn)

  qn = row_qn ∩ col_qn
  unoccupied = OffsetVector(ones_block.(T, row_ranks[qn], col_ranks[qn   ]), qn)

  qn = row_qn ∩ (col_qn.-1)
  occupied   = OffsetVector(ones_block.(T, row_ranks[qn], col_ranks[qn.+1]), qn)

  return SparseCore{T,N,d}(k,m,n,row_qn,col_qn,row_ranks,col_ranks,unoccupied,occupied)
end

function Base.similar(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  return SparseCore{T,N,d}(A.k,A.m,A.n,A.row_qn,A.col_qn,
                      deepcopy(A.row_ranks),deepcopy(A.col_ranks),
                      [zeros_block(T,A.row_ranks[n],A.col_ranks[n  ]) for n in axes(A.unoccupied,1)],
                      [zeros_block(T,A.row_ranks[n],A.col_ranks[n+1]) for n in axes(A.occupied,  1)])
end

function Base.size(A::SparseCore)
  return (A.m, 2, A.n)
end

function Base.axes(A::SparseCore)
  return (A.row_qn, 1:2, A.col_qn)
end

function Base.getindex(A::SparseCore{T}, l::Int, s::Int, r::Int) where T<:Number
  @boundscheck checkbounds(A, l,s,r)

  if s==1 && r == l # Unoccupied state
    return A.unoccupied[l]
  elseif s==2 && l+1 == r # Occupied state
    return A.occupied[l]
  else # Forbidden
    return zeros_block(T,A.row_ranks[l],A.col_ranks[r])
  end
end

function Base.getindex(A::SparseCore{T}, l::Int, r::Int) where T<:Number
  @boundscheck checkbounds(A, l,r-l+1,r)

  if r == l # Unoccupied state
    return A.unoccupied[l]
  elseif l+1 == r # Occupied state
    return A.occupied[l]
  else # Forbidden
    return zeros_block(T,A.row_ranks[l],A.col_ranks[r])
  end
end

function Base.getindex(A::SparseCore, n::Int, unfolding::Symbol)
  @assert unfolding == :horizontal || unfolding == :vertical ||
          unfolding == :R          || unfolding == :L

  if unfolding == :horizontal || unfolding == :R
    @boundscheck n ∈ axes(A,1) || throw(BoundsError(A, (n,:,:)))
    return hcat([A[n,r-n+1,r] for r∈axes(A,3)∩(n:n+1)]...)
  else # unfolding == :vertical || unfolding == :L
    @boundscheck n ∈ axes(A,3) || throw(BoundsError(A, (:,:,n)))
    return vcat([A[l,n-l+1,n] for l∈axes(A,1)∩(n-1:n)]...)
  end
end

function Base.setindex!(A::SparseCore{T}, X::Matrix{T}, l::Int,r::Int) where T<:Number
  Base.setindex!(A,Block(X),l,r-l+1,r)
end

function Base.setindex!(A::SparseCore{T}, X::Block{T}, l::Int,r::Int) where T<:Number
  Base.setindex!(A,X,l,r-l+1,r)
end

function Base.setindex!(A::SparseCore{T}, X::Matrix{T}, l::Int,s::Int,r::Int) where T<:Number
  Base.setindex!(A,Block(X),l,s,r)
end

function Base.setindex!(A::SparseCore{T}, X::Block{T}, l::Int,s::Int,r::Int) where T<:Number
  @boundscheck begin
    checkbounds(A, l,s,r)
    size(X) == (A.row_ranks[l],A.col_ranks[r]) || 
      throw(DimensionMismatch("Trying to assign block of size $(size(X)) to a block of prescribed ranks $((A.row_ranks[l], A.col_ranks[r]))"))
  end

  if s==1 && r == l # Unoccupied state
    A.unoccupied[l] = X
  elseif s==2 && l+1 == r # Occupied state
    A.occupied[l] = X
  else # Forbidden
    throw(BoundsError(A, (l,s,r)))
  end
end

function Base.setindex!(A::SparseCore{T,N,d}, X::Matrix{T}, n::Int, unfolding::Symbol) where {T<:Number,N,d}
  @assert unfolding == :horizontal || unfolding == :vertical ||
          unfolding == :R          || unfolding == :L

  if unfolding == :horizontal || unfolding == :R
    @boundscheck begin
      n ∈ axes(A,1) || throw(BoundsError(A, (n,:,:)))
      size(X,2) == sum(A.col_ranks[r] for r∈axes(A,3)∩(n:n+1)) || 
        throw(DimensionMismatch("Trying to assign block with incorrect number $(size(X,2)) of columns vs expected rank $(sum(A.col_ranks[r] for r∈axes(A,3)∩(n:n+1)))"))
    end
    A.row_ranks[n] = size(X,1)

    if n ∈ axes(A,3) && n+1 ∈ axes(A,3)
      A.unoccupied[n] = X[:,1:A.col_ranks[n]]
      A.occupied[n] = X[:,A.col_ranks[n]+1:end]
    elseif n ∈ axes(A,3)
      A.unoccupied[n] = X
    elseif n+1 ∈ axes(A,3)
      A.occupied[n] = X
    else
      throw(BoundsError(A, (n,:,n:n+1)))
    end

  elseif unfolding == :vertical || unfolding == :L
    @boundscheck begin
      n ∈ axes(A,3) || throw(BoundsError(A, (:,:,n)))
      size(X,1) == sum(A.row_ranks[l] for l∈axes(A,1)∩(n-1:n)) || 
        throw(DimensionMismatch("Trying to assign block with incorrect number $(size(X,1)) of rows vs expected rank $(sum(A.row_ranks[l] for l∈axes(A,1)∩(n-1:n)))"))
    end
    A.col_ranks[n] = size(X,2)

    if n-1 ∈ axes(A,1) && n ∈ axes(A,1)
      A.occupied[n-1] = X[1:A.row_ranks[n-1],:]
      A.unoccupied[n] = X[A.row_ranks[n-1]+1:end,:]
    elseif n-1 ∈ axes(A,1)
      A.occupied[n-1] = X
    elseif n ∈ axes(A,1)
      A.unoccupied[n] = X
    else
      throw(BoundsError(A, (n-1:n,:,n)))
    end
  end
end

function Base.show(io::IO, ::MIME"text/plain", A::SparseCore{T,N,d}) where {T<:Number,N,d}
  if get(io, :compact, true)
    str = "SparseCore{$T,$N,$d} with $(A.m)x2x$(A.n) block shape"
  else
    # Manage some formatting and padding
    strr = ["r[$(qn)]=$(A.row_ranks[qn])" for qn in axes(A,1)]
    len = max(length.(strr)...)
    padr = len .- length.(strr)
    strr_rows = ["$(s)"*" "^(pad+len+3)          for (s,pad) in zip(strr, padr)]

    strr = ["r[$(qn)]=$(A.col_ranks[qn])" for qn in axes(A,3)]
    len = max(length.(strr)...)
    padr = len .- length.(strr)
    strr_cols = ["$(s)"*" "^(pad+len+3)          for (s,pad) in zip(strr, padr)]

    str = string("SparseCore{$T,$N,$d} with $(A.m)x2x$(A.n) block shape and index $(A.k).",
      "\nRow ranks are ", strr_rows..., 
      "\nColumn ranks are ", strr_cols)
  end
  print(io, str)
end

function Base.show(io::IO, A::SparseCore)

    (size(A,1) == 0 || size(A,3) == 0) && return show(io, MIME("text/plain"), A)
    row_str = [" $(qn)‹$(A.row_ranks[qn]) " for qn in axes(A,1)]
    col_str = [" $(qn)‹$(A.col_ranks[qn]) " for qn in axes(A,3)]

    rw = maximum(length.(row_str))
    cw = maximum(length.(col_str))
    cpad(s,w) = rpad(lpad(s, div(w-length(s),2)+length(s)),w)
    row_str .= cpad.(row_str, rw)
    col_str .= cpad.(col_str, cw)

    Grid = fill(UInt16(10240), size(A,3)*(cw+1)+2+rw, 2size(A,1))
    Grid[rw+1,2:end] .= '⎢'
    Grid[end-1,2:end] .= '⎥'
    if size(A,1)>1
      Grid[rw+1,2] = '⎡'
      Grid[rw+1,end] = '⎣'
      Grid[end-1,2] = '⎤'
      Grid[end-1,end] = '⎦'
    else
      Grid[rw+1,2] = '['
      Grid[end-1,end] = ']'
    end
    Grid[end, :] .= '\n'

    for (j,n) in enumerate(axes(A,1))
      Grid[1:rw,2j]       .= collect(row_str[n])
    end
    for (i,n) in enumerate(axes(A,3))
      Grid[rw+1+(cw+1)*(i-1).+(1:cw),1]       .= collect(col_str[n])
    end

    for (j,n) in enumerate(axes(A,1))
      if n ∈ axes(A,3)
        i = findfirst(isequal(n), OffsetArrays.no_offset_view(axes(A,3)))
        if (A.row_ranks[n] > 0) && (A.col_ranks[n] > 0)
          Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j]   .= collect(cpad('○',cw))
        end
        if n+1 ∈ axes(A,3)
          Grid[rw+1+(cw+1)*i,2j] = '⎢'
          if n+1 ∈ axes(A,1)
            Grid[rw+1+(cw+1)*i,2j+1] = '+'
          end
        end
        if n-1 ∈ axes(A,3)
          Grid[rw+1+(cw+1)*(i-1),2j] = '⎢'
        end
        if n+1 ∈ axes(A,1)
          Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j+1] .= collect(cpad('⎯',cw))
        end
      end
      if n+1 ∈ axes(A,3)
        i = findfirst(isequal(n+1), OffsetArrays.no_offset_view(axes(A,3)))
        if (A.row_ranks[n] > 0) && (A.col_ranks[n+1] > 0)
          Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j]   .= collect(cpad('●',cw))
        end
        if n+1 ∈ axes(A,1)
          Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j+1]       .= collect(cpad('⎯',cw))
          if n+2 ∈ axes(A,3)
            Grid[rw+1+(cw+1)*i,2j+1] = '+'
          end
        end
        if n-1 ∈ axes(A,1)
          Grid[rw+1+(cw+1)*(i-1).+(1:cw),2j-1]       .= collect(cpad('⎯',cw))
        end
        if n+2 ∈ axes(A,3)
          Grid[rw+1+(cw+1)*i,2j] = '⎢'
        end
      end
    end

    foreach(c -> print(io, Char(c)), @view Grid[1:end-1])
end


function Base.summary(io::IO, A::SparseCore)
  show(io, summary(A))
end

function Base.summary(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  return string("[$(axes(A,1)[begin]):$(axes(A,1)[end])]x$(size(A,2))x[$(axes(A,3)[begin]):$(axes(A,3)[end])] SparseCore{$(T),$(N),$(d)})")
end

function LinearAlgebra.lmul!(a::Number, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  for n in axes(B.unoccupied,1)
    lmul!(T(a), B.unoccupied[n])
  end
  for n in axes(B.occupied,1)
    lmul!(T(a), B.occupied[n])
  end
  return B
end

function LinearAlgebra.rmul!(B::SparseCore{T,N,d}, a::Number) where {T<:Number,N,d}
  return lmul!(a,B)
end

function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::SparseCore{T,N,d}, b::Number) where {T<:Number,N,d}
  C.row_ranks .= A.row_ranks
  C.col_ranks .= A.col_ranks

  for n in axes(A.unoccupied,1)
    C.unoccupied[n] = T(b) * A.unoccupied[n]
  end
  for n in axes(A.occupied,1)
    C.occupied[n] = T(b) * A.occupied[n]
  end
  return C
end

function LinearAlgebra.mul!(C::SparseCore{T,N,d}, a::Number, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  C.row_ranks .= B.row_ranks
  C.col_ranks .= B.col_ranks

  for n in axes(B.unoccupied,1)
    C.unoccupied[n] = T(a) * B.unoccupied[n]
  end
  for n in axes(B.occupied,1)
    C.occupied[n] = T(a) * B.occupied[n]
  end
  return C
end

function Base.:*(a::Number, B::SparseCore)
  return lmul!(a,deepcopy(B))
end

Base.:*(A::SparseCore, b::Number) = b*A

function Base.:+(A::SparseCore)
  return deepcopy(A)
end

function Base.:-(A::SparseCore)
  return lmul!(-1, deepcopy(A))
end

function LinearAlgebra.lmul!(A::AbstractMatrix{T}, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  @assert B.k == 1
  @assert size(A,2) == B.row_ranks[0]

  B.row_ranks[0] = size(A,1)
  if 0 ∈ axes(B,3)
    B[0,1,0] = A * B[0,1,0]
  end
  if 1 ∈ axes(B,3)
    B[0,2,1] = A * B[0,2,1]
  end

  return B
end

function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::AbstractMatrix{T}, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  @assert C.k == B.k == 1
  @assert size(A,2) == B.row_ranks[0]

  C.row_ranks[0] = size(A,1)
  if 0 ∈ axes(B,3)
    C[0,1,0] = A * B[0,1,0]
  end
  if 1 ∈ axes(B,3)
    C[0,2,1] = A * B[0,2,1]
  end

  return C
end

function Base.:*(A::Mat, B::SparseCore{T,N,d}) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @assert B.k == 1
  return mul!(SparseCore{T,N,d}(1), A, B)
end

function LinearAlgebra.lmul!(A::OffsetVector{Mat, Vector{Mat}}, B::SparseCore{T,N,d}) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @boundscheck begin
    length(A) == B.m      || throw(DimensionMismatch("Number of matrices $(length(A)) does not match the number of block rows $(size(B,1))"))
    axes(A,1) == axes(B,1)|| throw(DimensionMismatch("Axes mismatch between matrices $(summary(axis(A,1))) and core row indices $(summary(axis(B,1)))"))
  end
  for n in axes(B,1)
    @assert size(A[n],2) == B.row_ranks[n]
  end

  for n in axes(B,1)
    B.row_ranks[n] = size(A[n],1)
    for m in axes(B,3) ∩ (n:n+1)
      B[n,m-n+1,m] = A[n] * B[n,m-n+1,m]
    end
  end

  return B
end


function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::OffsetVector{Mat, Vector{Mat}}, B::SparseCore{T,N,d}) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @boundscheck begin
    length(A) == B.m      || throw(DimensionMismatch("Number of matrices $(length(A)) does not match the number of block rows $(size(B,1))"))
    axes(A,1) == axes(B,1)|| throw(DimensionMismatch("Axes mismatch between matrices $(summary(axis(A,1))) and core row indices $(summary(axis(B,1)))"))
  end
  for n in axes(B,1)
    @assert size(A[n],2) == B.row_ranks[n]
  end

  C.col_ranks .= B.col_ranks

  for n in axes(B,1)
    C.row_ranks[n] = size(A[n],1)
    for m in axes(B,3) ∩ (n:n+1)
      C[n,m-n+1,m] = A[n] * B[n,m-n+1,m]
    end
  end

  return C
end

function Base.:*(A::OffsetVector{Mat, Vector{Mat}}, B::SparseCore{T,N,d}) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  return mul!(SparseCore{T,N,d}(B.k), A, B)
end

function LinearAlgebra.rmul!(A::SparseCore{T,N,d}, B::OffsetVector{Mat, Vector{Mat}}) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @boundscheck begin
    size(A,3) == length(B)|| throw(DimensionMismatch("Number of matrices $(length(B)) does not match the number of block columns $(size(B,3))"))
    axes(A,3) == axes(B,1)|| throw(DimensionMismatch("Axes mismatch between core column indices $(summary(axis(B,3))) and matrices $(summary(axis(A,1)))"))
  end
  for n in axes(A,3)
    @assert A.col_ranks[n] == size(B[n],1)
  end

  for n in axes(A,3)
    A.col_ranks[n] = size(B[n],2)
    for m in axes(A,1) ∩ (n-1:n)
      A[m,n-m+1,n] = A[m,n-m+1,n] * B[n]
    end
  end

  return A
end

function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::SparseCore{T,N,d}, B::OffsetVector{Mat, Vector{Mat}}) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @boundscheck begin
    size(A,3) == length(B)|| throw(DimensionMismatch("Number of matrices $(length(B)) does not match the number of block columns $(size(B,3))"))
    axes(A,3) == axes(B,1)|| throw(DimensionMismatch("Axes mismatch between core column indices $(summary(axis(B,3))) and matrices $(summary(axis(A,1)))"))
  end
  for n in axes(A,3)
    @assert A.col_ranks[n] == size(B[n],1)
  end

  C.row_ranks .= A.row_ranks

  for n in axes(A,3)
    C.col_ranks[n] = size(B[n],2)
    for m in axes(A,1) ∩ (n-1:n)
      C[m,n-m+1,n] = A[m,n-m+1,n] * B[n]
    end
  end

  return C
end

function Base.:*(A::SparseCore{T,N,d}, B::OffsetVector{Mat, Vector{Mat}}) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  return mul!(SparseCore{T,N,d}(A.k), A, B)
end

function LinearAlgebra.rmul!(A::SparseCore{T,N,d}, B::Mat) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @assert A.k == d
  @assert A.col_ranks[N] = size(B,1)

  for n in axes(A.unoccupied, 1)
    A.unoccupied[n] = A.unoccupied[n] * B
  end
  for n in axes(A.occupied, 1)
    A.occupied[n] = A.occupied[n] * B
  end

  A.col_ranks[N] = size(B,2)

  return A
end

function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::SparseCore{T,N,d}, B::Mat) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @assert C.k == A.k == d
  @assert A.col_ranks[N] = size(B,1)

  for n in axes(A.unoccupied, 1)
    C.unoccupied[n] = A.unoccupied[n] * B
  end
  for n in axes(A.occupied, 1)
    C.occupied[n] = A.occupied[n] * B
  end

  C.row_ranks = A.row_ranks
  C.col_ranks[N] = size(B,2)

  return C
end

function Base.:*(A::SparseCore{T,N,d}, B::Mat) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @assert A.k == d
  return mul!(SparseCore{T,N,d}(d), A, B)
end

function ⊕(A::SparseCore{T,N,d}, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  @assert A.k == B.k
  k = A.k
  if d>1
    if k==1
      row_ranks = A.row_ranks
      col_ranks = A.col_ranks + B.col_ranks

      unoccupied = hcat.(A.unoccupied, B.unoccupied)
      occupied   = hcat.(A.occupied,   B.occupied  )

    elseif 1<k<d
      row_ranks = A.row_ranks + B.row_ranks
      col_ranks = A.col_ranks + B.col_ranks

      unoccupied =  cat.(A.unoccupied, B.unoccupied, dims=(1,2))
      occupied   =  cat.(A.occupied,   B.occupied  , dims=(1,2))
    else # k == d
      row_ranks = A.row_ranks + B.row_ranks
      col_ranks = A.col_ranks

      unoccupied = vcat.(A.unoccupied, B.unoccupied)
      occupied   = vcat.(A.occupied,   B.occupied  )
    end
  else #d==1
    unoccupied = A.unoccupied .+ B.unoccupied
    occupied   = A.occupied   .+ B.occupied
  end

  return SparseCore{T,N,d}(k,size(A,1),size(A,3),axes(A,1),axes(A,3),row_ranks,col_ranks,unoccupied,occupied)
end

function LinearAlgebra.axpy!(α, v::SparseCore{T,N,d}, w::SparseCore{T,N,d}) where {T<:Number,N,d}
  @assert v.row_ranks == w.row_ranks
  @assert v.col_ranks == w.col_ranks

  axpy!.(α, v.unoccupied, w.unoccupied)
  axpy!.(α, v.occupied, w.occupied)
  return w
end

function LinearAlgebra.axpby!(α, v::SparseCore{T,N,d}, β, w::SparseCore{T,N,d}) where {T<:Number,N,d}
  @assert v.row_ranks == w.row_ranks
  @assert v.col_ranks == w.col_ranks

  axpy!.(α, v.unoccupied, β, w.unoccupied)
  axpy!.(α, v.occupied, β, w.occupied)
  return w
end

function LinearAlgebra.dot(v::SparseCore{T,N,d}, w::SparseCore{T,N,d}) where {T<:Number,N,d}
  @assert v.row_ranks == w.row_ranks
  @assert v.col_ranks == w.col_ranks

  s = T(0)
  for (V,W) in zip(v.unoccupied, w.unoccupied)
    s += dot(V,W)
  end
  for (V,W) in zip(v.occupied, w.occupied)
    s += dot(V,W)
  end
  return s
end

function norm2(v::SparseCore{T,N,d}) where {T<:Number,N,d}
  return sum(norm2, v.unoccupied) + sum(norm2, v.occupied)
end

function LinearAlgebra.norm(v::SparseCore{T,N,d}) where {T<:Number,N,d}
  return sqrt(norm2(v))
end

function ⊕(A::SparseCore{T,N,d}, b::Number) where {T<:Number,N,d}

  if d>1
    b = [T(b);;]
    if k==1
      row_ranks = A.row_ranks
      col_ranks = A.col_ranks .+ 1

      unoccupied = hcat.(A.unoccupied, b)
      occupied   = hcat.(A.occupied,   b  )
    elseif 1<k<d
      row_ranks = A.row_ranks .+ 1
      col_ranks = A.col_ranks .+ 1

      unoccupied =  cat.(A.unoccupied, b, dims=(1,2))
      occupied   =  cat.(A.occupied,   b, dims=(1,2))
    else # k == d
      row_ranks = A.row_ranks .+ 1
      col_ranks = A.col_ranks

      unoccupied = vcat.(A.unoccupied, b)
      occupied   = vcat.(A.occupied,   b)
    end
  else #d==1
    unoccupied = A.unoccupied .+ T(b)
    occupied   = A.occupied   .+T(b)
  end

  return SparseCore{T,N,d}(k,size(A,1),size(A,3),axes(A,1),axes(A,3),row_ranks,col_ranks,unoccupied,occupied)
end

⊕(b::Number, A::SparseCore) = A⊕b

function ⊗(A::SparseCore{T,N,d},B::SparseCore{T,N,d}) where {T<:Number,N,d}
  @assert A.k == B.k
  k = A.k
  row_ranks = A.row_ranks .* B.row_ranks
  col_ranks = A.col_ranks .* B.col_ranks

  unoccupied = [A.unoccupied[n]⊗B.unoccupied[n] for n in axes(A.unoccupied,1)]
  occupied = [A.occupied[n]⊗B.occupied[n] for n in axes(A.occupied,1)]

  return SparseCore{T,N,d}(k,size(A,1),size(A,3),axes(A,1),axes(A,3),row_ranks,col_ranks,unoccupied,occupied)
end

function LinearAlgebra.qr!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  R = [zeros(T,0,0) for n in axes(A,3)]

  for n in axes(A,3)
    An = A[n,:vertical]
    if size(An,1)>0 && size(An,2)>0
      F = qr!(An)

      r = sum(A.row_ranks[m] for m∈(n-1:n)∩axes(A,1))
      rank = min(r, A.col_ranks[n])
      Q = reshape(Matrix(F.Q), (:, rank))
      R[n] = F.R
      
      if rank < A.col_ranks[n]
        # We make sure all ranks are the same, even if we have to introduce redundant zeros.
        A[n,:vertical] = hcat(Q, zeros(T,size(Q,1), A.col_ranks[n]-rank))
        R[n] = vcat(R[n], zeros(T,A.col_ranks[n]-rank, A.col_ranks[n]))
      else
        A[n,:vertical] = Q
      end
    else
      # Keep A[n,:vertical] as is. It's empty anyway
      R[n] = zeros(T,A.col_ranks[n],A.col_ranks[n])
    end
  end

  return R
end

function LinearAlgebra.lq!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  L = [zeros(T,0,0) for n in axes(A,1)]

  for n in axes(A,1)
    An = A[n,:horizontal]
    if size(An,1) > 0 && size(An,2) > 0
      F = lq!(An)

      r = sum(A.col_ranks[m] for m∈(n:n+1)∩axes(A,3))
      rank = min(A.row_ranks[n],r)
      Q = reshape(Matrix(F.Q), (rank,:))
      L[n] = F.L

      if rank < A.row_ranks[n]
        # We make sure all ranks are the same, even if we have to introduce redundant zeros.
        A[n,:horizontal] = vcat(Q, zeros(T,A.row_ranks[n]-rank, size(Q,2)))
        L[n] = hcat(L[n], zeros(T,A.row_ranks[n], A.row_ranks[n]-rank))
      else
        A[n,:horizontal] = Q
      end
    else
    # Keep A[n,:horizontal] as is - it's empty anyway
      L[n] = zeros(T,A.row_ranks[n],A.row_ranks[n])
    end
  end

  return L
end


function my_qc!(A::Matrix{T}) where {T<:Number}
  m = size(A,1)
  n = size(A,2)
  if m>0 && n>0
    # Lapack in-place pivoted QR factorization
    A, tau, jpvt = LinearAlgebra.LAPACK.geqp3!(A)
    # Search for effective rank
    ϵ = 16 * A[1,1] * eps()
    rank = min(m,n) - searchsortedlast(view(A, reverse(diagind(A))), ϵ, by=abs)
    # Extract C = R*P' factor
    C = zeros(T, rank, n)
    for j=1:n, i=1:min(j,rank)
      C[i, jpvt[j]] = A[i,j]
    end
    # Extract Q factor into A
    LinearAlgebra.LAPACK.orgqr!(A, tau)
    Q = A[:,1:rank]
  else  # n = 0
    rank = 0
    Q = Matrix{T}(undef,m,rank)
    C = Matrix{T}(undef,rank,n)
  end

  return Q, C, rank
end

function qc!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  C = [zeros(T,0,0) for n in axes(A,3)]
  ranks = [0        for n in axes(A,3)]
  for n in axes(A,3)
      A[n,:vertical], C[n], ranks[n] = my_qc!(A[n,:vertical])
  end
  return C, ranks
end

function my_cq(A::Matrix{T}) where {T<:Number}
  m = size(A,1)
  n = size(A,2)
  tA = copy(transpose(A))
  Q,C,rank = my_qc!(tA)
  return copy(transpose(Q)), copy(transpose(C)), rank
end

function cq!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  C = [zeros(T,0,0) for n in axes(A,1)]
  ranks = [0        for n in axes(A,1)]
  for n in axes(A,1)
      A[n,:horizontal], C[n], ranks[n] = my_cq(A[n,:horizontal])
  end
  return C, ranks
end

function LinearAlgebra.svd(A::SparseCore{T,N,d}, unfolding::Symbol) where {T<:Number,N,d}
  @assert unfolding == :horizontal || unfolding == :vertical ||
          unfolding == :R          || unfolding == :L

  if unfolding == :R || unfolding == :horizontal

    U  = [zeros(T,0,0) for n in axes(A,1)]
    S  = [zeros(T,0  ) for n in axes(A,1)]
    Vt = [zeros(T,0,0) for n in axes(A,1)]

    for n in axes(A,1)
      An = A[n, :horizontal]
      if size(An,1) > 0
        F = svd(An)
        U[n] = F.U
        S[n] = F.S
        Vt[n] = F.Vt
      else # An is 0x? - fix Julia's thin svd inconsistency
        # U[n] = zeros(T,0,0)
        # S[n] = zeros(T,0  )
        Vt[n] = zeros(T,size(An))
      end
    end

    return U, S, Vt
  elseif unfolding == :L || unfolding == :vertical

    U  = [zeros(T,0,0) for n in axes(A,3)]
    S  = [zeros(T,0  ) for n in axes(A,3)]
    Vt = [zeros(T,0,0) for n in axes(A,3)]

    for n ∈ axes(A,3)
      An = A[n, :vertical]
      if size(An,1) > 0
        F = svd(An)
        U[n]  = F.U
        S[n]  = F.S
        Vt[n] = F.Vt
      else # An is 0x? - fix Julia's thin svd inconsistency
        # U[n] = zeros(T,0,0)
        # S[n] = zeros(T,0  )
        Vt[n] = zeros(T,size(An))
      end
    end
  end

  return U, S, Vt
end

function LinearAlgebra.svd!(A::SparseCore{T,N,d}, unfolding::Symbol) where {T<:Number,N,d}
  @assert unfolding == :horizontal || unfolding == :vertical ||
          unfolding == :R          || unfolding == :L

  if unfolding == :R || unfolding == :horizontal

    U = [zeros(T,0,0) for n in axes(A,1)]
    S = [zeros(T,0  ) for n in axes(A,1)]

    for n in axes(A,1)
      An = A[n, :horizontal]
      if size(An,1) > 0
        F = svd(An)
        U[n] = F.U
        S[n] = F.S
        A[n,:horizontal] = F.Vt
      else # An is 0x? - fix Julia's thin svd inconsistency
        # U[n] = zeros(T,0,0)
        # S[n] = zeros(T,0  )
        A[n,:horizontal] = zeros(T,size(An))
      end
    end

    return U, S, A
  elseif unfolding == :L || unfolding == :vertical

    S  = [zeros(T,0  ) for n in axes(A,3)]
    Vt = [zeros(T,0,0) for n in axes(A,3)]

    for n ∈ axes(A,3)
      An = A[n, :vertical]
      if size(An,1) > 0
        F = svd(An)
        A[n,:vertical]  = F.U
        S[n]  = F.S
        Vt[n] = F.Vt
      else # An is 0x? - fix Julia's thin svd inconsistency
        # A[n,:vertical] = zeros(T,0,0)
        # S[n] = zeros(T,0)
        Vt[n] = zeros(T,size(An))
      end
    end

    return A, S, Vt
  end
end

function reduce_ranks!(A::SparseCore{T,N,d}, 
                       row_ranks::OffsetVector{Int, Vector{Int}}, 
                       col_ranks::OffsetVector{Int, Vector{Int}}) where {T<:Number,N,d}
  @assert axes(row_ranks) == axes(A.row_ranks)
  @assert axes(col_ranks) == axes(A.col_ranks)

  for n in axes(A.unoccupied,1)
    A.unoccupied[n] = A.unoccupied[n][1:row_ranks[n],1:col_ranks[n]]
  end
  for n in axes(A.occupied,1)
    A.occupied[n] = A.occupied[n][1:row_ranks[n],1:col_ranks[n+1]]
  end
  A.row_ranks .= row_ranks
  A.col_ranks .= col_ranks

  return A
end

function Base.Array(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  array = zeros(T,sum(A.row_ranks),2,sum(A.col_ranks))
  row_block = [0;cumsum(OffsetArrays.no_offset_view(A.row_ranks))]
  col_block = [0;cumsum(OffsetArrays.no_offset_view(A.col_ranks))]

  for (i,n) in enumerate(axes(A,1))
    if n ∈ axes(A,3)
      j = findfirst(isequal(n), OffsetArrays.no_offset_view(axes(A,3)))
      array[row_block[i]+1:row_block[i+1],1,col_block[j]+1:col_block[j+1]] = A[n,1,n]
    end
    if n+1 ∈ axes(A,3)
      j = findfirst(isequal(n+1), OffsetArrays.no_offset_view(axes(A,3)))
      array[row_block[i]+1:row_block[i+1],2,col_block[j]+1:col_block[j+1]] = A[n,2,n+1]
    end
  end

  return array
end