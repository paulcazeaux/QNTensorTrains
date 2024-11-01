"""
  SparseCore{T<:Number,N,d,k}

Special bidiagonal sparse structure,
where diagonal  correspond to modal index = 1, 
and  just above the diagonal correspond to modal index 2

N is the total number of electrons and d is the overall tensor order; dictates structure
"""
struct SparseCore{T<:Number,N,d,S<:AbstractMatrix{T}} <: AbstractArray{S,3}
  k::Int        # Core index

  m::Int        # row size
  n::Int        # column size

  row_qn::OffsetArrays.IdOffsetRange{Int64,UnitRange{Int64}}
  col_qn::OffsetArrays.IdOffsetRange{Int64,UnitRange{Int64}}

  row_ranks::OffsetVector{Int, Vector{Int}}
  col_ranks::OffsetVector{Int, Vector{Int}}

  unoccupied::OffsetVector{S, Vector{S}}
  occupied::OffsetVector{S, Vector{S}}

  mem::Memory{T}

  # Main constructor - single memory allocation
  function SparseCore{T,N,d}(k::Int, row_ranks::OffsetVector{Int, Vector{Int}}, 
                                     col_ranks::OffsetVector{Int, Vector{Int}},
                                     mem::Memory{T}, offset::Int=0) where {T<:Number,N,d}

    @boundscheck begin
      N ≤ d                       || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
      1 ≤ k ≤ d                   || throw(BoundsError())
    end
    m = min(k-1, N) - min(max(N+k-1-d, 0), N) + 1
    n = min(k  , N) - min(max(N+k  -d, 0), N) + 1

    row_qn = occupation_qn(N,d,k)
    col_qn = occupation_qn(N,d,k+1)
    @boundscheck begin
      axes(row_ranks,1) == row_qn || throw(DimensionMismatch("Unexpected quantum number indices for given row ranks array"))
      axes(col_ranks,1) == col_qn || throw(DimensionMismatch("Unexpected quantum number indices for given column ranks array"))
    end

    sz = sum(row_ranks[n]*col_ranks[n]   for n ∈ row_qn ∩  col_qn    ) +
         sum(row_ranks[n]*col_ranks[n+1] for n ∈ row_qn ∩ (col_qn.-1))

    @boundscheck begin
      @assert length(mem) ≥ offset+sz
    end

    unoccupied = Vector{Matrix{T}}(undef, length(row_qn ∩ col_qn    ))
    occupied   = Vector{Matrix{T}}(undef, length(row_qn ∩(col_qn.-1)))
    unoccupied = OffsetVector(unoccupied, row_qn ∩ col_qn    )
    occupied   = OffsetVector(  occupied, row_qn ∩(col_qn.-1))

    mem[1+offset:sz+offset] .= T(0)
    idx = 1+offset
    for n in row_qn ∩ col_qn
      unoccupied[n] = Block(row_ranks[n],col_ranks[n],mem,idx)
      idx += length(unoccupied[n])
    end
    for n in row_qn ∩ (col_qn.-1)
      occupied[n] = Block(row_ranks[n],col_ranks[n+1],mem,idx)
      idx += length(occupied[n])
    end

    return new{T,N,d,Matrix{T}}(k,m,n,row_qn,col_qn,
                        deepcopy(row_ranks),deepcopy(col_ranks),
                        unoccupied, occupied, mem)
  end

  # Partial initialization without memory field when fully populated block arrays are provided
  function SparseCore{T,N,d}(k::Int, unoccupied::OffsetVector{S, Vector{S}}, occupied::OffsetVector{S, Vector{S}}) where {T<:Number,N,d,S<:AbstractMatrix{T}}
    @boundscheck begin
      N ≤ d                       || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
      1 ≤ k ≤ d                   || throw(BoundsError())
    end
    m = min(k-1, N) - min(max(N+k-1-d, 0), N) + 1
    n = min(k  , N) - min(max(N+k  -d, 0), N) + 1

    row_qn = occupation_qn(N,d,k)
    col_qn = occupation_qn(N,d,k+1)
    @boundscheck begin
      @assert axes(unoccupied,1) == row_qn ∩ col_qn
      @assert axes(occupied,  1) == row_qn ∩(col_qn.-1)
    end
    row_ranks = OffsetVector(Vector{Int}(undef, length(row_qn)), row_qn)
    col_ranks = OffsetVector(Vector{Int}(undef, length(col_qn)), col_qn)
    for l in row_qn
      if l ∈ col_qn
        @boundscheck l+1 ∈ col_qn && @assert size(unoccupied[l],1) == size(occupied[l],1)
        row_ranks[l] = size(unoccupied[l],1)
      else # l+1 ∈ col_qn
        row_ranks[l] = size(occupied[l],1)
      end
    end
    for r in col_qn
      if r ∈ row_qn
        @boundscheck r-1 ∈ row_qn && @assert size(unoccupied[r],2) == size(occupied[r-1],2)
        col_ranks[r] = size(unoccupied[r],2)
      else # r-1 ∈ row_qn
        col_ranks[r] = size(occupied[r-1],2)
      end
    end
    return new{T,N,d,S}(k, m, n, row_qn, col_qn, row_ranks, col_ranks, unoccupied, occupied)
  end
end

function SparseCore{T,N,d}(k::Int, row_ranks::OffsetVector{Int, Vector{Int}}, 
                                   col_ranks::OffsetVector{Int, Vector{Int}}) where {T<:Number,N,d}
  row_qn = occupation_qn(N,d,k)
  col_qn = occupation_qn(N,d,k+1)

  sz = sum(row_ranks[n]*col_ranks[n]   for n ∈ row_qn ∩  col_qn    ) +
       sum(row_ranks[n]*col_ranks[n+1] for n ∈ row_qn ∩ (col_qn.-1))
  mem = Memory{T}(undef,sz)
  return SparseCore{T,N,d}(k,row_ranks,col_ranks,mem)
end

function occupation_qn(N::Int, d::Int, k::Int)
  @boundscheck begin
    N ≤ d || throw(DimensionMismatch("Total number of electrons $N cannot be larger than dimension $d"))
    1 ≤ k ≤ d+1 || throw(BoundsError())
  end
  qn = min(max(N+k-1-d, 0), N):min(k-1, N)
  return OffsetArrays.IdOffsetRange(values=qn, indices=qn)
end

@inline function Block(row_rank::Int,col_rank::Int,mem::Memory{T},idx::Int) where T<:Number
  return row_rank>0 && col_rank>0 ? Base.wrap(Array, memoryref(mem,idx), (row_rank, col_rank)) : zeros(T, row_rank, col_rank)
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

  row_qn = occupation_qn(N,d,k)
  col_qn = occupation_qn(N,d,k+1)

  row_ranks = OffsetVector(Int.(row_qn .== Nl  ), row_qn)
  col_ranks = OffsetVector(Int.(col_qn .== Nl+s), col_qn)

  new_core = SparseCore{T,N,d}(k, row_ranks, col_ranks)
  fill!.(new_core.unoccupied, 1)
  fill!.(new_core.occupied, 1)

  return new_core
end

@inline function Base.similar(A::SparseCore{T,N,d,Matrix{T}}) where {T<:Number,N,d}
  return SparseCore{T,N,d}(site(A),A.row_ranks,A.col_ranks)
end

@inline function Base.size(A::SparseCore)
  return (A.m, 2, A.n)
end

@inline function Base.axes(A::SparseCore)
  return (A.row_qn, 1:2, A.col_qn)
end

@inline function Base.length(A::SparseCore)
  return A.m*2*A.n
end

@inline function site(A::SparseCore)
  return A.k
end

@propagate_inbounds function Base.getindex(A::SparseCore{T,N,d,S}, l::Int, s::Int, r::Int) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  @boundscheck checkbounds(A, l,s,r)

  if s==1 && r == l # Unoccupied state
    return A.unoccupied[l]
  elseif s==2 && l+1 == r # Occupied state
    return A.occupied[l]
  else # Forbidden
    zero_block = S(undef,A.row_ranks[l],A.col_ranks[r])
    fill!(zero_block, T(0))
    return zero_block
  end
end

@propagate_inbounds function Base.getindex(A::SparseCore, l::Int, r::Int)
  return A[l,r-l+1,r]
end

@propagate_inbounds function Base.getindex(A::SparseCore{T,N,d,Matrix{T}}, n::Int, unfolding::Symbol) where {T<:Number,N,d}
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

## Views
@propagate_inbounds function Base.getindex(
    A::SparseCore{T,N,d,S},  
    ::Colon, 
    ::Colon) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  row_ranks  = A.col_ranks
  col_ranks  = A.col_ranks
  unoccupied = [ view(A.unoccupied[l],:,:) for l in axes(A.unoccupied,1) ] 
  occupied   = [ view(A.occupied[l],  :,:) for l in axes(A.occupied,  1) ] 
  return SparseCore{T,N,d}(A.k, unoccupied,occupied)
end

@propagate_inbounds function Base.getindex(
    A::SparseCore{T,N,d,S}, 
    I::OffsetVector{UnitRange{Int},Vector{UnitRange{Int}}},
    ::Colon) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  @boundscheck begin
    @assert axes(A,1) == axes(I,1)
    @assert all(I[l] ⊆ 1:A.row_ranks[l] for l in A.row_qn)
  end
  
  row_ranks  = [ length(I[l]) for l in axes(A.row_ranks,1) ]
  col_ranks  = A.col_ranks
  unoccupied = [ view(A.unoccupied[l],I[l],:) for l in axes(A.unoccupied,1) ] 
  occupied   = [ view(A.occupied[l],  I[l],:) for l in axes(A.occupied,  1) ]
  return SparseCore{T,N,d}(A.k, unoccupied,occupied)
end

@propagate_inbounds function Base.getindex(
    A::SparseCore{T,N,d,S}, 
    ::Colon,
    J::OffsetVector{UnitRange{Int},Vector{UnitRange{Int}}}) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  @boundscheck begin
    @assert axes(A,3) == axes(J,1)
    @assert all(J[r] ⊆ 1:A.col_ranks[r] for r in A.col_qn)
  end
  row_ranks  = A.row_ranks
  col_ranks  = [ length(J[r]) for r in axes(A.col_ranks,1) ]
  unoccupied = [ view(A.unoccupied[l],:,J[l  ]) for l in axes(A.unoccupied,1) ] 
  occupied   = [ view(A.occupied[l],  :,J[l+1]) for l in axes(A.occupied,  1) ]
  return SparseCore{T,N,d}(A.k, unoccupied,occupied)
end


@propagate_inbounds function Base.getindex(
    A::SparseCore{T,N,d,S}, 
    I::OffsetVector{UnitRange{Int},Vector{UnitRange{Int}}},
    J::OffsetVector{UnitRange{Int},Vector{UnitRange{Int}}}) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  @boundscheck begin
    @assert axes(A,1) == axes(I,1)
    @assert all(J[r] ⊆ 1:A.col_ranks[r] for r in A.col_qn)
  end
  row_ranks  = [ length(I[l]) for l in axes(A.row_ranks,1) ]
  col_ranks  = [ length(J[r]) for r in axes(A.col_ranks,1) ]
  unoccupied = [ view(A.unoccupied[l],I[l],J[l  ]) for l in axes(A.unoccupied,1) ] 
  occupied   = [ view(A.occupied[l],  I[l],J[l+1]) for l in axes(A.occupied,  1) ]
  return SparseCore{T,N,d}(A.k, unoccupied,occupied)
end

@propagate_inbounds function Base.setindex!(A::SparseCore{T,N,d,S}, X::S, l::Int,r::Int) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  Base.setindex!(A,X,l,r-l+1,r)
end

@propagate_inbounds function Base.setindex!(A::SparseCore{T,N,d,S}, X::S, l::Int,s::Int,r::Int) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  @boundscheck begin
    checkbounds(A, l,s,r)
    size(X) == (A.row_ranks[l],A.col_ranks[r]) || 
      throw(DimensionMismatch("Trying to assign block of size $(size(X)) to a block of prescribed ranks $((A.row_ranks[l], A.col_ranks[r]))"))
  end

  if s==1 && r == l # Unoccupied state
    copyto!(A.unoccupied[l], X)
  elseif s==2 && l+1 == r # Occupied state
    copyto!(A.occupied[l], X)
  else # Forbidden
    throw(BoundsError(A, (l,s,r)))
  end
end

@propagate_inbounds function Base.setindex!(A::SparseCore{T,N,d,Matrix{T}}, X::S, n::Int, unfolding::Symbol) where {T<:Number,N,d,S<:AbstractMatrix{T}}
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
      copyto!(A.unoccupied[n], view(X,:,1:A.col_ranks[n]))
      copyto!(A.occupied[n],   view(X,:,A.col_ranks[n]+1:A.col_ranks[n]+A.col_ranks[n+1]))
    elseif n ∈ axes(A,3)
      copyto!(A.unoccupied[n], X)
    elseif n+1 ∈ axes(A,3)
      copyto!(A.occupied[n], X)
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
      copyto!(A.occupied[n-1], view(X,1:A.row_ranks[n-1],:))
      copyto!(A.unoccupied[n], view(X,A.row_ranks[n-1]+1:A.row_ranks[n-1]+A.row_ranks[n],:))
    elseif n-1 ∈ axes(A,1)
      copyto!(A.occupied[n-1], X)
    elseif n ∈ axes(A,1)
      copyto!(A.unoccupied[n], X)
    else
      throw(BoundsError(A, (n-1:n,:,n)))
    end
  end
end

@propagate_inbounds function Base.copyto!(dest::SparseCore{T,N,d}, src::SparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert dest.parent.k == src.k
    @assert dest.row_ranks == src.row_ranks && dest.col_ranks == src.col_ranks
  end
  for (a,b) in zip(dest.unoccupied, src.unoccupied)
    copyto!(a,b)
  end
  for (a,b) in zip(dest.occupied, src.occupied)
    copyto!(a,b)
  end
  return dest
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

    str = string("SparseCore{$T,$N,$d} with $(A.m)x2x$(A.n) block shape and index $(site(A)).",
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

struct AdjointSparseCore{T<:Number,N,d,S<:AbstractMatrix{T}}
  parent::SparseCore{T,N,d,S}
end

@inline function LinearAlgebra.adjoint(A::SparseCore{T,N,d,S}) where {T<:Number,N,d,S<:AbstractMatrix{T}}
  return AdjointSparseCore{T,N,d,S}(A)
end

@inline function LinearAlgebra.adjoint(A::AdjointSparseCore)
  return parent(A)
end

@inline function Base.parent(A::AdjointSparseCore)
  return A.parent
end

@inline function Base.size(A::AdjointSparseCore)
  return reverse(size(parent(A)))
end

@inline function Base.size(A::AdjointSparseCore, i)
  return size(A)[i]
end

@inline function Base.length(A::AdjointSparseCore)
  return length(parent(A))
end

@inline function Base.axes(A::AdjointSparseCore)
  return reverse(axes(parent(A)))
end

@inline function Base.axes(A::AdjointSparseCore, i)
  return axes(A)[i]
end

@inline function site(A::AdjointSparseCore)
  return site(parent(A))
end

@inline @propagate_inbounds function Base.getindex(A::AdjointSparseCore, I, J)
  return adjoint(parent(A)[J,I])
end

@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::SparseCore{T,N,d}, b::Number) where {T<:Number,N,d}
  @boundscheck @assert C.row_ranks == A.row_ranks
  @boundscheck @assert C.col_ranks == A.col_ranks

  for n in axes(A.unoccupied,1)
    mul!(C.unoccupied[n], T(b), A.unoccupied[n])
  end
  for n in axes(A.occupied,1)
    mul!(C.occupied[n], T(b), A.occupied[n])
  end
  return C
end

@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,N,d}, a::Number, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  return mul!(C,B,a)
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

@propagate_inbounds function LinearAlgebra.lmul!(a::Number, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  for n in axes(B.unoccupied,1)
    lmul!(T(a), B.unoccupied[n])
  end
  for n in axes(B.occupied,1)
    lmul!(T(a), B.occupied[n])
  end
  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(B::SparseCore{T,N,d}, a::Number) where {T<:Number,N,d}
  return lmul!(a,B)
end

@propagate_inbounds function LinearAlgebra.lmul!(A::AbstractMatrix{T}, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(B) == 1
    @assert size(A,1) == size(A,2) == B.row_ranks[0]
  end

  for n in axes(B.unoccupied, 1)
    lmul!(A, B.unoccupied[n])
  end
  for n in axes(B.occupied, 1)
    lmul!(A, B.occupied[n])
  end
  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(A::SparseCore{T,N,d}, B::Mat) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @assert site(A) == d
  @assert A.col_ranks[N] = size(B,1) == size(B,2)

  for n in axes(A.unoccupied, 1)
    rmul!(A.unoccupied[n], B)
  end
  for n in axes(A.occupied, 1)
    rmul!(A.occupied[n], B)
  end
  return A
end

@propagate_inbounds function LinearAlgebra.lmul!(A::Frame{T,N,d,Mat}, B::SparseCore{T,N,d}) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @boundscheck begin
    site(A) == site(B) && axes(A,1) == axes(B,1) || throw(DimensionMismatch("Axes mismatch between matrices $(summary(axis(A,1))) and core row indices $(summary(axis(B,1)))"))
    @assert A.row_ranks == A.col_ranks == B.row_ranks
  end

  for n in axes(B,1), m in axes(B,3) ∩ (n:n+1)
    lmul!(A[n], B[n,m-n+1,m])
  end

  return B
end

@propagate_inbounds function LinearAlgebra.rmul!(A::SparseCore{T,N,d}, B::Frame{T,N,d,Mat}) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  @boundscheck begin
    axes(A,3) == axes(B,1)|| throw(DimensionMismatch("Axes mismatch between core column indices $(summary(axis(B,3))) and matrices $(summary(axis(A,1)))"))
    for n in axes(A,3)
      @assert A.col_ranks[n] == size(B[n],1) == size(B[n],2)
    end
  end

  for n in axes(A,3), m in axes(A,1) ∩ (n-1:n)
    rmul!(A[m,n-m+1,n], B[n])
  end

  return A
end

"""
LinearAlgebra.mul!(C::SparseCore, A::Frame, B::SparseCore, α=1, β=0)
      --
      |
C =   A
      |    |
      -- --B--
"""
@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::Frame{T,N,d,M}, B::SparseCore{T,N,d}, α::Number=1, β::Number=0) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck begin
    @assert site(A) == site(B)
    @assert C.row_ranks == A.row_ranks
    @assert A.col_ranks == B.row_ranks
    @assert C.col_ranks == B.col_ranks
  end

  for n in axes(C.unoccupied,1)
    mul!(C.unoccupied[n], A.blocks[n], B.unoccupied[n], α, β)
  end
  for n in axes(C.occupied,1)
    mul!(C.occupied[n], A.blocks[n], B.occupied[n], α, β)
  end
  return C
end


"""
LinearAlgebra.mul!(C::SparseCore, A::SparseCore, B::Frame, α=1, β=0)
           --
            | 
C =         B 
       |    | 
     --A-- --
"""
@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::SparseCore{T,N,d}, B::Frame{T,N,d,M}, α::Number=1, β::Number=0) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck begin
    @assert site(A)+1 == site(B)
    @assert C.row_ranks == A.row_ranks
    @assert A.col_ranks == B.row_ranks
    @assert C.col_ranks == B.col_ranks
  end
  for n in axes(C.unoccupied,1)
    mul!(C.unoccupied[n], A.unoccupied[n], B.blocks[n], α, β)
  end
  for n in axes(C.occupied,1)
    mul!(C.occupied[n], A.occupied[n], B.blocks[n+1], α, β)
  end
  return C
end

"""
LinearAlgebra.mul!(C::AdjointSparseCore, A::Frame, B::AdjointSparseCore, α=1, β=0)
    -- --A--
    |    |   
C = B 
    | 
    --
"""
@propagate_inbounds function LinearAlgebra.mul!(C::AdjointSparseCore{T,N,d}, A::AdjointSparseCore{T,N,d}, B::Frame{T,N,d,Mat}, α::Number=1, β::Number=0) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  tA = parent(A)
  tC = parent(C)
  @boundscheck begin
    @assert site(A) == site(B) == site(C)
    @assert tC.row_ranks == B.col_ranks
    @assert B.row_ranks == tA.row_ranks
    @assert tC.col_ranks == tA.col_ranks
  end
  for n in axes(tC.unoccupied,1)
    mul!(tC.unoccupied[n], adjoint(B.blocks[n]), tA.unoccupied[n], α, β)
  end
  for n in axes(tC.occupied,1)
    mul!(tC.occupied[n], adjoint(B.blocks[n]), tA.occupied[n], α, β)
  end
  return C
end

"""
LinearAlgebra.mul!(C::AdjointSparseCore, A::Frame, B::AdjointSparseCore, α=1, β=0)
   --B-- -
     |   |
C =      A
         |
        --
"""
@propagate_inbounds function LinearAlgebra.mul!(C::AdjointSparseCore{T,N,d}, A::Frame{T,N,d,M}, B::AdjointSparseCore{T,N,d}, α::Number=1, β::Number=0) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  tB = parent(B)
  tC = parent(C)
  @boundscheck begin
    @assert site(A)-1 == site(B) == site(C)
    @assert tC.row_ranks == tB.row_ranks
    @assert tB.col_ranks == A.col_ranks
    @assert tC.col_ranks == A.row_ranks
  end
  for n in axes(tC.unoccupied,1)
    mul!(tC.unoccupied[n], tB.unoccupied[n], adjoint(A.blocks[n]), α, β)
  end
  for n in axes(tC.occupied,1)
    mul!(tC.occupied[n], tB.occupied[n], adjoint(A.blocks[n+1]), α, β)
  end
  return C
end


"""
LinearAlgebra.mul!(C::Frame, A::AdjointSparseCore, B::SparseCore, α, β)
     - --A--
     |   |
C =  |   
     |   |
     ----B--
"""
@propagate_inbounds function LinearAlgebra.mul!(C::Frame{T,N,d,Mat}, A::AdjointSparseCore{T,N,d}, B::SparseCore{T,N,d}, α::Number, β::Number) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  tA = parent(A)
  @boundscheck begin
    @assert site(A) == site(B) == site(C)-1
    @assert C.row_ranks == tA.col_ranks
    @assert B.row_ranks == tA.row_ranks
    @assert C.col_ranks == B.col_ranks
  end
  for n in axes(C,1)
    if n ∈ axes(tA.unoccupied,1) && n-1 ∈ axes(tA.occupied,1)
      mul!(C.blocks[n],adjoint(tA.unoccupied[n]),B.unoccupied[n],α,β)
      mul!(C.blocks[n],adjoint(tA.occupied[n-1]),B.occupied[n-1],α,1)
    elseif n ∈ axes(tA.unoccupied,1)
      mul!(C.blocks[n],adjoint(tA.unoccupied[n]),B.unoccupied[n],α,β)
    else #if n-1 ∈ axes(A.occupied,1)
      mul!(C.blocks[n],adjoint(tA.occupied[n-1]),B.occupied[n-1],α,β)
    end
  end
  return C
end

"""
LinearAlgebra.mul!(C::Frame, A::SparseCore, B::AdjointSparseCore, α, β)
     --B-- -
       |   |
C =        |
       |   |
     --A----
"""
@propagate_inbounds function LinearAlgebra.mul!(C::Frame{T,N,d,Mat}, A::SparseCore{T,N,d}, B::AdjointSparseCore{T,N,d}, α::Number, β::Number) where {T<:Number,N,d,Mat<:AbstractMatrix{T}}
  tB = parent(B)
  @boundscheck begin
    @assert site(C) == site(B) == site(A)
    @assert C.row_ranks == A.row_ranks
    @assert A.col_ranks == tB.col_ranks
    @assert C.col_ranks == tB.row_ranks
  end
  for n in axes(C,1)
    if n ∈ axes(A.unoccupied,1) && n ∈ axes(A.occupied,1)
      mul!(C.blocks[n],A.unoccupied[n],adjoint(tB.unoccupied[n]),α,β)
      mul!(C.blocks[n],A.occupied[n],  adjoint(tB.occupied[n]),  α,1)
    elseif n ∈ axes(A.unoccupied,1)
      mul!(C.blocks[n],A.unoccupied[n],adjoint(tB.unoccupied[n]),α,β)
    else #if n ∈ axes(A.occupied,1)
      mul!(C.blocks[n],A.occupied[n],  adjoint(tB.occupied[n]),  α,β)
    end
  end
  return C
end

@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::M, B::SparseCore{T,N,d}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck begin
    @assert site(C) == site(B) == 1
    @assert C.row_ranks == B.row_ranks && C.col_ranks == B.col_ranks
    @assert size(A,1) == size(A,2) == B.row_ranks[0]
  end

  for (b,c) in zip(B.unoccupied, C.unoccupied)
    mul!(c,A,b)
  end
  for (b,c) in zip(B.occupied, C.occupied)
    mul!(c,A,b)
  end

  return C
end

@propagate_inbounds function LinearAlgebra.mul!(C::SparseCore{T,N,d}, A::SparseCore{T,N,d}, B::M, α::Number=1, β::Number=0) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck begin
    @assert site(A) == site(C) == d
    @assert A.col_ranks[N] == size(B,1)
    @assert C.col_ranks[N] == size(B,2)
    @assert C.row_ranks == A.row_ranks
  end

  for n in axes(A.unoccupied, 1)
    mul!(C.unoccupied[n], A.unoccupied[n], B, α, β)
  end
  for n in axes(A.occupied, 1)
    mul!(C.occupied[n], A.occupied[n], B, α, β)
  end

  return C
end

function Base.:*(A::Frame{T,N,d,M}, B::SparseCore{T,N,d}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  return mul!(SparseCore{T,N,d}(site(B), A.row_ranks, B.col_ranks), A, B)
end

function Base.:*(A::SparseCore{T,N,d}, B::Frame{T,N,d,M}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  return mul!(SparseCore{T,N,d}(site(A), A.row_ranks, B.col_ranks), A, B)
end

function Base.:*(A::AdjointSparseCore{T,N,d}, B::Frame{T,N,d,M}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  return mul!(adjoint(SparseCore{T,N,d}(site(A), B.col_ranks, parent(A).col_ranks)), A, B)
end

function Base.:*(A::Frame{T,N,d,M}, B::AdjointSparseCore{T,N,d}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  return mul!(adjoint(SparseCore{T,N,d}(site(B), parent(B).row_ranks, A.row_ranks)), A, B)
end

function Base.:*(A::AdjointSparseCore{T,N,d}, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  return mul!(Frame{T,N,d}(site(A)+1,parent(A).col_ranks,B.col_ranks), A, B)
end

function Base.:*(A::SparseCore{T,N,d}, B::AdjointSparseCore{T,N,d}) where {T<:Number,N,d}
  return mul!(Frame{T,N,d}(site(A), A.row_ranks, parent(B).row_ranks), A, B)
end

function Base.:*(A::M, B::SparseCore{T,N,d}) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck @assert site(B) == 1 && size(A,1) == size(A,2) == B.row_ranks[0]
  return mul!(similar(B), A, B)
end

function Base.:*(A::SparseCore{T,N,d}, B::M) where {T<:Number,N,d,M<:AbstractMatrix{T}}
  @boundscheck @assert site(A) == d
  return mul!(SparseCore{T,N,d}(d,A.row_ranks, [size(B,2) for n in occupation_qn(N,d,d)]), A, B)
end


function ⊕(A::SparseCore{T,N,d}, B::SparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck @assert site(A) == site(B)
  k = site(A)
  if d>1
    if k==1
      @boundscheck @assert A.row_ranks == B.row_ranks
      row_ranks = A.row_ranks
      col_ranks = A.col_ranks + B.col_ranks
    elseif 1<k<d
      row_ranks = A.row_ranks + B.row_ranks
      col_ranks = A.col_ranks + B.col_ranks
    else # k == d
      @boundscheck @assert A.col_ranks == B.col_ranks
      row_ranks = A.row_ranks + B.row_ranks
      col_ranks = A.col_ranks
    end
  else #d==1
    @boundscheck @assert A.row_ranks == B.row_ranks
    @boundscheck @assert A.col_ranks == B.col_ranks
    row_ranks = A.row_ranks
    col_ranks = A.col_ranks
  end
  C = SparseCore{T,N,d}(k,row_ranks,col_ranks)

  if d>1
    if k==1
      for (a,b,c) in zip(A.unoccupied, B.unoccupied, C.unoccupied)
        copyto!(view(c,axes(a,1), axes(a,2))     , a)
        copyto!(view(c,axes(a,1), size(a,2).+axes(b,2)), b)
      end
      for (a,b,c) in zip(A.occupied, B.occupied, C.occupied)
        copyto!(view(c, axes(a,1), axes(a,2))     , a)
        copyto!(view(c, axes(a,1), size(a,2).+axes(b,2)), b)
      end

    elseif 1<k<d
      for (a,b,c) in zip(A.unoccupied, B.unoccupied, C.unoccupied)
        copyto!(view(c, axes(a,1)           , axes(a,2)           ), a)
        copyto!(view(c, size(a,1).+axes(b,1), size(a,2).+axes(b,2)), b)
      end
      for (a,b,c) in zip(A.occupied, B.occupied, C.occupied)
        copyto!(view(c, axes(a,1)           , axes(a,2)           ), a)
        copyto!(view(c, size(a,1).+axes(b,1), size(a,2).+axes(b,2)), b)
      end
    else # k == d
      for (a,b,c) in zip(A.unoccupied, B.unoccupied, C.unoccupied)
        copyto!(view(c, axes(a,1)           , axes(a,2)), a)
        copyto!(view(c, size(a,1).+axes(b,1), axes(a,2)), b)
      end
      for (a,b,c) in zip(A.occupied, B.occupied, C.occupied)
        copyto!(view(c, axes(a,1)           , axes(a,2)), a)
        copyto!(view(c, size(a,1).+axes(b,1), axes(a,2)), b)
      end
    end
  else #d==1
    for (a,b,c) in zip(A.unoccupied, B.unoccupied, C.unoccupied)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
    for (a,b,c) in zip(A.occupied, B.occupied, C.occupied)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
  end

  return C
end

@propagate_inbounds function LinearAlgebra.axpy!(α, v::SparseCore{T,N,d}, w::SparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck @assert v.row_ranks == w.row_ranks
  @boundscheck @assert v.col_ranks == w.col_ranks

  axpy!.(α, v.unoccupied, w.unoccupied)
  axpy!.(α, v.occupied, w.occupied)
  return w
end

@propagate_inbounds function LinearAlgebra.axpby!(α, v::SparseCore{T,N,d}, β, w::SparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck @assert v.row_ranks == w.row_ranks
  @boundscheck @assert v.col_ranks == w.col_ranks

  axpby!.(α, v.unoccupied, β, w.unoccupied)
  axpby!.(α, v.occupied, β, w.occupied)
  return w
end

@propagate_inbounds function LinearAlgebra.dot(V::SparseCore{T,N,d}, W::SparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck @assert V.row_ranks == W.row_ranks
  @boundscheck @assert V.col_ranks == W.col_ranks

  s = T(0)
  for (v,w) in zip(V.unoccupied, W.unoccupied)
    s += dot(v,w)
  end
  for (v,w) in zip(V.occupied, W.occupied)
    s += dot(v,w)
  end
  return s
end

function Base.abs2(V::SparseCore{T,N,d}) where {T<:Number,N,d}
  return sum(block->sum(abs2, block), V.unoccupied) + sum(block->sum(abs2, block), V.occupied)
end

function LinearAlgebra.norm(V::SparseCore{T,N,d}) where {T<:Number,N,d}
  return sqrt(abs2(V))
end

function ⊕(A::SparseCore{T,N,d}, b::Number) where {T<:Number,N,d}

  if d>1
    if k==1
      row_ranks = A.row_ranks
      col_ranks = A.col_ranks .+ 1
    elseif 1<k<d
      row_ranks = A.row_ranks .+ 1
      col_ranks = A.col_ranks .+ 1
    else # k == d
      row_ranks = A.row_ranks .+ 1
      col_ranks = A.col_ranks
    end
  else #d==1
    row_ranks = A.row_ranks
    col_ranks = A.col_ranks
  end
  C = SparseCore{T,N,d}(site(A),row_ranks,col_ranks)

  if d>1
    if k==1
      for (a,c) in zip(A.unoccupied, C.unoccupied)
        copyto!(view(c, axes(a,1), axes(a,2)  ), a   )
        fill!(  view(c, axes(a,1), size(a,2)+1), T(b))
      end
      for (a,c) in zip(A.occupied, C.occupied)
        copyto!(view(c, axes(a,1), axes(a,2)  ), a   )
        fill!(  view(c, axes(a,1), size(a,2)+1), T(b))
      end

    elseif 1<k<d
      for (a,c) in zip(A.unoccupied, C.unoccupied)
        copyto!(view(c, axes(a,1)  , axes(a,2)  ), a   )
        fill!(  view(c, axes(a,1)  , size(a,2)+1), T(0))
        fill!(  view(c, size(a,1)+1, axes(a,2)  ), T(0))
        fill!(  view(c, size(A,1)+1, size(A,2)+1), T(b))
      end
      for (a,c) in zip(A.occupied, C.occupied)
        copyto!(view(c, axes(a,1)  , axes(a,2)  ), a   )
        fill!(  view(c, axes(a,1)  , size(a,2)+1), T(0))
        fill!(  view(c, size(a,1)+1, axes(a,2)  ), T(0))
        fill!(  view(c, size(A,1)+1, size(A,2)+1), T(b))
      end
    else # k == d
      for (a,c) in zip(A.unoccupied, C.unoccupied)
        copyto!(view(c, axes(a,1)  , axes(a,2)), a   )
        fill!(  view(c, size(a,1)+1, axes(a,2)), T(b))
      end
      for (a,c) in zip(A.occupied, C.occupied)
        copyto!(view(c, axes(a,1)  , axes(a,2)), a   )
        fill!(  view(c, size(a,1)+1, axes(a,2)), T(b))
      end
    end
  else #d==1
    for (a,c) in zip(A.unoccupied, C.unoccupied)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
    for (a,c) in zip(A.occupied, C.occupied)
      copyto!(c, a)
      axpy!(T(1),b,c)
    end
  end

  return C
end

⊕(b::Number, A::SparseCore) = A⊕b

function ⊗(A::SparseCore{T,N,d},B::SparseCore{T,N,d}) where {T<:Number,N,d}
  @assert site(A) == site(B)
  k = site(A)
  row_ranks = A.row_ranks .* B.row_ranks
  col_ranks = A.col_ranks .* B.col_ranks

  C = SparseCore{T,N,d}(k, row_ranks, col_ranks)
  for (a,b,c) in zip(A.unoccupied, B.unoccupied, C.unoccupied)
    c .= reshape( 
            reshape(a, (size(a,1),1,size(a,2),1)) .* 
            reshape(b, (1,size(b,1),1,size(b,2))),
            (size(a,1)*size(b,1),size(a,2)*size(b,2))
                )
  end
  for (a,b,c) in zip(A.occupied, B.occupied, C.occupied)
    c .= reshape( 
                  reshape(a, (size(a,1),1,size(a,2),1)) .* 
                  reshape(b, (1,size(b,1),1,size(b,2))),
                  (size(a,1)*size(b,1),size(a,2)*size(b,2))
                      )
  end

  return C
end

function LinearAlgebra.qr!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  R = [UpperTriangular(zeros(T,A.col_ranks[n],A.col_ranks[n])) for n in axes(A,3)]

  for n in axes(A,3)
    An = A[n,:vertical]
    if size(An,1)>0 && size(An,2)>0
      F = qr!(An)

      r = sum(A.row_ranks[m] for m∈(n-1:n)∩axes(A,1))
      rank = min(r, A.col_ranks[n])
      Q = Matrix(F.Q)
      copyto!(R[n], 1:rank, 1:A.col_ranks[n], 
              F.R,       1:rank, 1:A.col_ranks[n])
      
      # We make sure all ranks are the same, even if we have to introduce redundant zeros.
      if n-1 ∈ axes(A,1) && n ∈ axes(A,1)
        copyto!( A.occupied[n-1], 1:A.row_ranks[n-1], 1:rank,
                 Q,                    1:A.row_ranks[n-1], 1:rank)
        fill!(  view(A.occupied[n-1], 1:A.row_ranks[n-1], rank+1:A.col_ranks[n]), T(0))
        copyto!( A.unoccupied[n], 1:A.row_ranks[n], 1:rank,
                 Q,                   (1:A.row_ranks[n]).+A.row_ranks[n-1],1:rank)
        fill!(  view(A.unoccupied[n], 1:A.row_ranks[n], rank+1:A.col_ranks[n]), T(0))
      elseif n-1 ∈ axes(A,1)
        copyto!( A.occupied[n-1], 1:A.row_ranks[n-1], 1:rank,
                 Q,                    1:A.row_ranks[n-1], 1:rank)
        fill!(  view(A.occupied[n-1], 1:A.row_ranks[n-1], rank+1:A.col_ranks[n]), T(0))
      elseif n ∈ axes(A,1)
        copyto!( A.unoccupied[n], 1:A.row_ranks[n], 1:rank,
                 Q,                    1:A.row_ranks[n], 1:rank)
        fill!(  view(A.unoccupied[n], 1:A.row_ranks[n], rank+1:A.col_ranks[n]), T(0))
      end
    end
  end

  return Frame{T,N,d}(site(A)+1, R)
end

function LinearAlgebra.lq!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  L = [LowerTriangular(zeros(T,A.row_ranks[n],A.row_ranks[n])) for n in axes(A,1)]

  for n in axes(A,1)
    An = A[n,:horizontal]
    if size(An,1) > 0 && size(An,2) > 0
      F = lq!(An)

      r = sum(A.col_ranks[m] for m∈(n:n+1)∩axes(A,3))
      rank = min(A.row_ranks[n],r)
      Q = Matrix(F.Q)
      copyto!(L[n], 1:A.row_ranks[n], 1:rank,
              F.L,       1:A.row_ranks[n], 1:rank)

      # We make sure all ranks are the same, even if we have to introduce redundant zeros.
      if n ∈ axes(A,3) && n+1 ∈ axes(A,3)
        copyto!(A.unoccupied[n], 1:rank, 1:A.col_ranks[n], 
                Q,                    1:rank, 1:A.col_ranks[n])
        fill!( view(A.unoccupied[n], rank+1:A.row_ranks[n], 1:A.col_ranks[n  ]), T(0))
        copyto!(A.occupied[n], 1:rank, 1:A.col_ranks[n+1],  
                Q,                  1:rank,(1:A.col_ranks[n+1]).+A.col_ranks[n])
        fill!( view(A.occupied[n],   rank+1:A.row_ranks[n], 1:A.col_ranks[n+1]), T(0))
      elseif n ∈ axes(A,3)
        copyto!(A.unoccupied[n], 1:rank, 1:A.col_ranks[n], 
                Q,                    1:rank, 1:A.col_ranks[n])
        fill!( view(A.unoccupied[n], rank+1:A.row_ranks[n], 1:A.col_ranks[n  ]), T(0))
      elseif n+1 ∈ axes(A,3)
        copyto!(A.occupied[n], 1:rank, 1:A.col_ranks[n+1], 
                Q,                  1:rank, 1:A.col_ranks[n+1])
        fill!( view(A.occupied[n],   rank+1:A.row_ranks[n], 1:A.col_ranks[n+1]), T(0))
      end
    end
  end

  return Frame{T,N,d}(site(A), L)
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
    Q = view(A,:,1:rank)
  else  # n = 0
    rank = 0
    Q = view(Matrix{T}(undef,m,rank),:,1:rank)
    C = Matrix{T}(undef,rank,n)
  end

  return Q, C, rank
end

function qc!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  col_qn = axes(A,3)
  Qblocks   = OffsetArray( Vector{Matrix{T}}(undef, length(col_qn)), col_qn)
  C         = OffsetArray( Vector{Matrix{T}}(undef, length(col_qn)), col_qn)
  col_ranks = OffsetArray( Vector{Int}(      undef, length(col_qn)), col_qn)
  for n in col_qn
    Qblocks[n], C[n], col_ranks[n] = my_qc!(A[n,:vertical])
  end

  Q = SparseCore{T,N,d}(site(A), A.row_ranks, col_ranks)
  for n in col_qn
      Q[n,:vertical] = Qblocks[n]
  end
  return Q, Frame{T,N,d}(site(A)+1, C), col_ranks
end

function my_cq(A::Matrix{T}) where {T<:Number}
  m = size(A,1)
  n = size(A,2)
  tA = copy(transpose(A))
  Q,C,rank = my_qc!(tA)
  return copy(transpose(Q)), copy(transpose(C)), rank
end

function cq!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  row_qn = axes(A,1)
  Qblocks   = OffsetArray( Vector{Matrix{T}}(undef, length(row_qn)), row_qn)
  C         = OffsetArray( Vector{Matrix{T}}(undef, length(row_qn)), row_qn)
  row_ranks = OffsetArray( Vector{Int}(      undef, length(row_qn)), row_qn)
  for n in row_qn
    Qblocks[n], C[n], row_ranks[n] = my_cq(A[n,:horizontal])
  end

  Q = SparseCore{T,N,d}(site(A), row_ranks, A.col_ranks)
  for n in row_qn
      Q[n,:horizontal] = Qblocks[n]
  end
  return Q, Frame{T,N,d}(site(A), C), row_ranks
end

function svd_horizontal(A::SparseCore{T,N,d}) where {T<:Number,N,d}  
  U  = OffsetArray( Vector{Matrix{T}}(undef, size(A,1)), axes(A,1))
  S  = OffsetArray( Vector{Vector{T}}(undef, size(A,1)), axes(A,1))
  Vt = OffsetArray( Vector{Matrix{T}}(undef, size(A,1)), axes(A,1))
  for n in axes(A,1)
    if A.row_ranks[n] > 0
      F = svd!(A[n,:horizontal])
      U[n] = F.U
      S[n] = F.S
      Vt[n]= F.Vt
    else
      U[n] = zeros(T,0,0)
      S[n] = zeros(T,0)
      Vt[n]= zeros(T,0,sum(A.col_ranks[m] for m in axes(A,3)∩(n:n+1)))
    end
  end
  return Frame{T,N,d}(site(A), U), Frame{T,N,d}(site(A), Diagonal.(S)), Vt
end

function svd_vertical(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  F  = [ A.row_ranks[n] > 0 ? [unpack_svd!(A[n, :horizontal])...] : [zeros(T,0,0),zeros(T,0),zeros(T,0,sum(A.col_ranks[m] for m in axes(A,3)∩(n:n+1)))] for n in axes(A,1)]
  
  U  = OffsetArray( Vector{Matrix{T}}(undef, size(A,3)), axes(A,3))
  S  = OffsetArray( Vector{Vector{T}}(undef, size(A,3)), axes(A,3))
  Vt = OffsetArray( Vector{Matrix{T}}(undef, size(A,3)), axes(A,3))
  for n in axes(A,3)
    if A.col_ranks[n] > 0
      F = svd!(A[n,:vertical])
      U[n] = F.U
      S[n] = F.S
      Vt[n]= F.Vt
    else
      U[n] = zeros(T,sum(A.row_ranks[m] for m in axes(A,1)∩(n-1:n)),0)
      S[n] = zeros(T,0)
      Vt[n]= zeros(T,0,0)
    end
  end
  return U, Frame{T,N,d}(site(A)+1, Diagonal.(S)), Frame{T,N,d}(site(A)+1, Vt)
end

function svd_horizontal!(A::SparseCore{T,N,d}) where {T<:Number,N,d}  
  U  = OffsetArray( Vector{Matrix{T}}(undef, size(A,1)), axes(A,1))
  S  = OffsetArray( Vector{Vector{T}}(undef, size(A,1)), axes(A,1))
  Vt = OffsetArray( Vector{Matrix{T}}(undef, size(A,1)), axes(A,1))
  for n in axes(A,1)
    if A.row_ranks[n] > 0
      F = svd!(A[n,:horizontal])
      U[n] = F.U
      S[n] = F.S
      Vt[n]= F.Vt
    else
      U[n] = zeros(T,0,0)
      S[n] = zeros(T,0)
      Vt[n]= zeros(T,0,sum(A.col_ranks[m] for m in axes(A,3)∩(n:n+1)))
    end
  end
  C = SparseCore{T,N,d}(site(A), length.(S), A.col_ranks)
  for n in axes(C,1)
    C[n,:horizontal] = Vt[n]
  end
  return Frame{T,N,d}(site(A), U), Frame{T,N,d}(site(A), Diagonal.(S)), C
end

function svd_vertical!(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  F  = [ A.row_ranks[n] > 0 ? [unpack_svd!(A[n, :horizontal])...] : [zeros(T,0,0),zeros(T,0),zeros(T,0,sum(A.col_ranks[m] for m in axes(A,3)∩(n:n+1)))] for n in axes(A,1)]
  
  U  = OffsetArray( Vector{Matrix{T}}(undef, size(A,3)), axes(A,3))
  S  = OffsetArray( Vector{Vector{T}}(undef, size(A,3)), axes(A,3))
  Vt = OffsetArray( Vector{Matrix{T}}(undef, size(A,3)), axes(A,3))
  for n in axes(A,3)
    if A.col_ranks[n] > 0
      F = svd!(A[n,:vertical])
      U[n] = F.U
      S[n] = F.S
      Vt[n]= F.Vt
    else
      U[n] = zeros(T,sum(A.row_ranks[m] for m in axes(A,1)∩(n-1:n)),0)
      S[n] = zeros(T,0)
      Vt[n]= zeros(T,0,0)
    end
  end
  C = SparseCore{T,N,d}(site(A), A.row_ranks, length.(S))
  for n in axes(C,3)
    C[n,:vertical] = U[n]
  end
  return C, Frame{T,N,d}(site(A)+1, Diagonal.(S)), Frame{T,N,d}(site(A)+1, Vt)
end

function svdvals_horizontal(A::SparseCore{T,N,d}) where {T<:Number,N,d}
  return [svdvals!(A[n, :horizontal]) for n in axes(A,1)]
end

function svdvals_vertical(A::SparseCore{T,N,d}, unfolding::Symbol) where {T<:Number,N,d}
  return [svdvals!(A[n, :vertical]) for n in axes(A,3)]
end

function reduce_ranks(A::SparseCore{T,N,d}, 
                       row_ranks::OffsetVector{Int, Vector{Int}}, 
                       col_ranks::OffsetVector{Int, Vector{Int}}) where {T<:Number,N,d}
  @boundscheck begin
    @assert axes(row_ranks) == axes(A.row_ranks)
    @assert axes(col_ranks) == axes(A.col_ranks)
    @assert all(row_ranks .≤ A.row_ranks)
    @assert all(col_ranks .≤ A.col_ranks)
  end
  
  B = SparseCore{T,N,d}(site(A), row_ranks, col_ranks)
  
  for n in axes(A.unoccupied,1)
    copyto!(B.unoccupied[n], view(A.unoccupied[n],1:row_ranks[n],1:col_ranks[n]))
  end
  for n in axes(A.occupied,1)
    copyto!(B.occupied[n], view(A.occupied[n],1:row_ranks[n],1:col_ranks[n+1]))
  end

  return B
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

function VectorInterface.zerovector(x::SparseCore{T,N,d}) where {T<:Number,N,d}
  return SparseCore{T,N,d}(x.k,x.row_ranks,x.col_ranks)
end

function VectorInterface.zerovector(x::SparseCore{S,N,d}, T::Type{<:Number}) where {S<:Number,N,d}
  return SparseCore{T,N,d}(x.k,x.row_ranks,x.col_ranks)
end

function VectorInterface.add!!(y::SparseCore{T,N,d}, x::SparseCore{T,N,d}, α::Number, β::Number) where {T<:Number,N,d}
  return axpby!(α,x,β,y)
end

function VectorInterface.scale(x::SparseCore{T,N,d}, α::Number) where {T<:Number,N,d}
    return VectorInterface.scale!!(deepcopy(x), α)
end

function VectorInterface.scale!!(x::SparseCore{T,N,d}, α::Number) where {T<:Number,N,d}
    α === VectorInterface.One() && return x
    return lmul!(α,x)
end
function VectorInterface.scale!!(y::SparseCore{T,N,d}, x::SparseCore{T,N,d}, α::Number) where {T<:Number,N,d}
    return mul!(y,x,α)
end

function VectorInterface.inner(x::SparseCore{T,N,d}, y::SparseCore{T,N,d}) where {T<:Number,N,d}
  return LinearAlgebra.dot(x,y)
end
