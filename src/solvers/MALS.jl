"""
  x = MALS(t::Matrix{T}, v::Array{T,4}, x0::TTtensor{T,d}, ε::Float64)

Implementation of the Modified Alternative Least Squares,
approximately solving for the ground state `Hx = λx` where `H` is 
a two-body Hamiltonian given in second quantization format as:
H = Σ t_ij a†_i a_j + Σ v_ijkl a†_i a†_j a_k a_l
The algorithm adapts ranks: the result will not necessarily have the
same ranks as the initial guess `x0`.
"""
function MALS(H::SparseHamiltonian{T,N,d},x0::TTvector{T,N,d}, ε::Float64 = 1e-4, maxIter::Int = 20) where {T<:Number,N,d}
  x = deepcopy(x0)
  λ, x = MALS!(H,x,ε,maxIter)

  return λ, x
end

function MALS!(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}, ε::Float64, maxIter::Int) where {T<:Number,N,d}
  # Right-orthogonalize the tensor x if necessary
  x.corePosition == 1 || rightOrthogonalize!(x, keepRank=true) 

  λ = RayleighQuotient(H,x)
  for it in 1:maxIter
    _ = MALSForwardSweep!(H,x,ε)
    λn = MALSBackSweep!(H,x,ε)
    r = abs(λ-λn)/abs(λn)
    λ = λn
    r < ε && break
  end

  return λ, x
end


function MALSForwardSweep!(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}, ε::Float64, inner_tol::T = 1e-6) where {T<:Number,N,d}
  move_core!(x, 1; keepRank=true) # Right-orthogonalize the tensor x
  Fᴿ = RightToLeftFraming(H,x)
  Fᴸ = IdFrame(Val(N), Val(d), 1)

  λ = T(0)
  ep = ε/sqrt(d-1)
  for k=1:d-1
    # Assumption: `x` is orthogonal with core at `k`
    @boundscheck @assert x.corePosition == k
    # Compute new cores.
    vals, vecs, info = KrylovKit.eigsolve(
                          core_pair -> FramedHamiltonian(H, core_pair,Fᴸ,Fᴿ[k+1]), 
                          contract(x.cores[k],x.cores[k+1]), 1, :SR;
                          krylovdim=5,
                          issymmetric=true, tol=inner_tol, verbosity=0)

    λ = vals[1]
    Xₖ, S, Xₖ₊₁ = factor_svd!(vecs[1], ep)
    lmul!(S,Xₖ₊₁) 
    set_core!(x, Xₖ)
    set_core!(x, Xₖ₊₁)
    rank(x,k+1) .= x.cores[k].col_ranks

    x.corePosition = k+1
    if k < d-1 
      # Compute the new left frame matrix Fᴸ
      Fᴸ = FramingStepRight(H,x,Fᴸ)
    end
  end

  return λ
end

function MALSBackSweep!(H::SparseHamiltonian{T,N,d}, x::TTvector{T,N,d}, ε::Float64, inner_tol::T = 1e-6) where {T<:Number,N,d}
  move_core!(x, d-1; keepRank=false) # Right-orthogonalize the tensor x
  Fᴸ = LeftToRightFraming(H,x)
  Fᴿ = IdFrame(Val(N), Val(d), d+1)
  Fᴿ = FramingStepLeft(H,x,Fᴿ)

  λ = T(0)
  ep = ε/sqrt(d-1)
  for k=d-2:-1:1
    # Assumption: `x` is orthogonal with core at `k+1`
    @boundscheck @assert x.corePosition == k+1

    # Compute new cores.
    vals, vecs, info = KrylovKit.eigsolve(
                          core_pair -> FramedHamiltonian(H,core_pair,Fᴸ[k],Fᴿ), 
                          contract(x.cores[k],x.cores[k+1]), 1, :SR;
                          krylovdim=5,
                          issymmetric=true, tol=inner_tol, verbosity=0)

    λ = vals[1]
    Xₖ, S, Xₖ₊₁ = factor_svd!(vecs[1], ep)
    rmul!(Xₖ, S)
    set_core!(x, Xₖ)
    set_core!(x, Xₖ₊₁)
    rank(x,k+1) .= x.cores[k].col_ranks

    x.corePosition = k
    if k>1 
      # Compute the new right frame matrix Fᴿ
      Fᴿ = FramingStepLeft(H,x,Fᴿ)
    end
  end

  return λ
end

# Special structure for the contraction of two successive cores
struct ContractedSparseCores{T<:Number,N,d} <: AbstractArray{Matrix{T},4}
  k::Int        # Core indices should be k and k+1

  m::Int        # row size
  n::Int        # column size

  row_qn::OffsetArrays.IdOffsetRange{Int64,UnitRange{Int64}}
  col_qn::OffsetArrays.IdOffsetRange{Int64,UnitRange{Int64}}

  row_ranks::OffsetVector{Int, Vector{Int}}
  col_ranks::OffsetVector{Int, Vector{Int}}

  blocks::OffsetVector{Matrix{T}, Vector{Matrix{T}}}

  mem::Memory{T}
end

function ContractedSparseCores{T,N,d}( k::Int,
              row_ranks::OffsetVector{Int, Vector{Int}},
              col_ranks::OffsetVector{Int, Vector{Int}}) where {T<:Number,N,d}
  row_qn = occupation_qn(N,d,k)
  mid_qn = occupation_qn(N,d,k+1)
  col_qn = occupation_qn(N,d,k+2)

  m = length(row_qn)
  n = length(col_qn)

  @boundscheck @assert row_qn == axes(row_ranks,1)
  @boundscheck @assert col_qn == axes(col_ranks,1)

  shapes = OffsetVector([ (sum(row_ranks[(m-1:m)∩row_qn]),  sum(col_ranks[(m:m+1)∩col_qn])) for m in mid_qn], mid_qn)

  mem = Memory{T}(undef, sum(prod.(shapes)))
  fill!(mem, T(0))
  idx = 1

  blocks = OffsetVector( Vector{Matrix{T}}(undef, size(mid_qn,1)), mid_qn )
  for m in mid_qn
    shape = shapes[m]
    if prod(shape)>0
      blocks[m] = Base.wrap(Array, memoryref(mem, idx), shape)
    else
      blocks[m] = zeros(T,shape)
    end
    idx += length(blocks[m])
  end

  return ContractedSparseCores{T,N,d}( k, m, n, 
                                      row_qn, col_qn, 
                                      deepcopy(row_ranks), 
                                      deepcopy(col_ranks),
                                      blocks, mem)
end

@inline function Base.size(v::ContractedSparseCores)
  return (v.m,v.n)
end

function Base.similar(v::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  return ContractedSparseCores{T,N,d}(v.k, v.row_ranks, v.col_ranks)
end

@inline function site(v::ContractedSparseCores)
  return v.k
end

function Base.:*(α::Number, v::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  w = similar(v)
  w.blocks .*= α
  return w
end

function LinearAlgebra.mul!(w::ContractedSparseCores{T,N,d}, v::ContractedSparseCores{T,N,d}, α::Number) where {T<:Number,N,d}
  @boundscheck begin
    @assert w.k == v.k
    @assert w.m == v.m
    @assert w.n == v.n
    @assert w.row_qn == v.row_qn
    @assert w.col_qn == v.col_qn
    @assert w.row_ranks == v.row_ranks
    @assert w.col_ranks == v.col_ranks
  end

  mul!.(w.blocks, α, v.blocks)
  return w
end

function LinearAlgebra.lmul!(α::Number, v::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  lmul!.(α, v.blocks)
  return v
end

function LinearAlgebra.rmul!(v::ContractedSparseCores{T,N,d}, α::Number) where {T<:Number,N,d}
  rmul!.(v.blocks, α)
  return v
end

function LinearAlgebra.axpy!(α::Number, v::ContractedSparseCores{T,N,d}, w::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert w.k == v.k
    @assert w.m == v.m
    @assert w.n == v.n
    @assert w.row_qn == v.row_qn
    @assert w.col_qn == v.col_qn
    @assert w.row_ranks == v.row_ranks
    @assert w.col_ranks == v.col_ranks
  end

  axpy!.(α,v.blocks,w.blocks)
  return w
end

function LinearAlgebra.axpby!(α::Number, v::ContractedSparseCores{T,N,d}, β::Number, w::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert w.k == v.k
    @assert w.m == v.m
    @assert w.n == v.n
    @assert w.row_qn == v.row_qn
    @assert w.col_qn == v.col_qn
    @assert w.row_ranks == v.row_ranks
    @assert w.col_ranks == v.col_ranks
  end

  axpby!.(α,v.blocks,β,w.blocks)
  return w
end

function LinearAlgebra.norm(x::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  s = sum(b -> sum(abs.(b).^2), x.blocks)
  return sqrt(s)
end

function LinearAlgebra.dot(x::ContractedSparseCores{T,N,d}, y::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert x.row_ranks == y.row_ranks
    @assert x.col_ranks == y.col_ranks
  end
  return sum(dot(x.blocks[m],y.blocks[m]) for m in axes(x.blocks,1))
end

function contract(Xₖ::SparseCore{T,N,d}, Xₖ₊₁::SparseCore{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert Xₖ.k+1 == Xₖ₊₁.k
    @assert Xₖ.col_ranks == Xₖ₊₁.row_ranks
  end
  k = Xₖ.k
  row_ranks = Xₖ.row_ranks
  col_ranks = Xₖ₊₁.col_ranks
  twocores = ContractedSparseCores{T,N,d}(k,row_ranks,col_ranks)

  mid_qn = occupation_qn(N,d,k+1)
  for m in mid_qn
    for l in (m-1:m)∩axes(Xₖ,1), r in (m:m+1)∩axes(Xₖ₊₁,3)
      Nr, Nc = size(twocores.blocks[m])
      rows = (l == m-1 ? (1:row_ranks[l]) : (Nr-row_ranks[l]+1:Nr))
      cols = (r == m   ? (1:col_ranks[r]) : (Nc-col_ranks[r]+1:Nc))
      mul!(view(twocores.blocks[m],rows,cols), Xₖ[l,m], Xₖ₊₁[m,r])
    end
  end

  return twocores
end

function factor_qc(twocores::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  k = twocores.k
  mid_qn = occupation_qn(N,d,k+1)

  Q = OffsetVector(   Vector{Matrix{T}}(undef, length(mid_qn)), mid_qn)
  C = OffsetVector(   Vector{Matrix{T}}(undef, length(mid_qn)), mid_qn)
  mid_rank = OffsetVector( Vector{Int}( undef, length(mid_qn)), mid_qn)
  for m in mid_qn
    Q[m],C[m],mid_rank[m] = my_qc!(copy(twocores.blocks[m]))
  end
  Xₖ = SparseCore{T,N,d}(k, twocores.row_ranks, mid_rank)
  Xₖ₊₁ = SparseCore{T,N,d}(k+1, mid_rank, twocores.col_ranks)

  for m in mid_qn
    Xₖ[m,:vertical] = Q[m]
    Xₖ₊₁[m,:horizontal] = C[m]
  end

  return Xₖ, Xₖ₊₁
end

function factor_svd!(twocores::ContractedSparseCores{T,N,d}, ep::Float64) where {T<:Number,N,d}
  k = twocores.k
  mid_qn = occupation_qn(N,d,k+1)

  U  = OffsetVector(  Vector{Matrix{T}}(undef, length(mid_qn)), mid_qn)
  S  = OffsetVector(  Vector{Vector{T}}(undef, length(mid_qn)), mid_qn)
  Vt = OffsetVector(  Vector{Matrix{T}}(undef, length(mid_qn)), mid_qn)
  mid_rank = OffsetVector( Vector{Int}( undef, length(mid_qn)), mid_qn)

  for m in mid_qn
    F = svd!(twocores.blocks[m])
    U[m],S[m],Vt[m] = F.U, F.S, F.Vt
  end
  S = Frame{T,N,d}(k+1,Diagonal.(S))
  mid_rank = chop(S, ep)
  s = norm(S)

  Xₖ   = SparseCore{T,N,d}(k,   twocores.row_ranks, mid_rank)
  Xₖ₊₁ = SparseCore{T,N,d}(k+1, mid_rank, twocores.col_ranks)
  for m in mid_qn
    r = mid_rank[m]
    S[m] = Diagonal(diag(S[m])[1:r] ./ s)
    Xₖ[m,:vertical]     = U[m][:,1:r]
    Xₖ₊₁[m,:horizontal] = Vt[m][1:r,:]
  end

  return Xₖ, S, Xₖ₊₁
end

function VectorInterface.zerovector(x::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  y = similar(x)
  fill!.(y.blocks, T(0))
  return y
end

function VectorInterface.zerovector(x::ContractedSparseCores{S,N,d}, T::Type{<:Number}) where {S<:Number,N,d}
  return ContractedSparseCores{T,N,d}( x.k, x.row_ranks, x.col_ranks)
end

function VectorInterface.add!!(y::ContractedSparseCores{T,N,d}, x::ContractedSparseCores{T,N,d}, α::Number, β::Number) where {T<:Number,N,d}
  axpby!(α,x,β,y)
end

function VectorInterface.scale(x::ContractedSparseCores{T,N,d}, α::Number) where {T<:Number,N,d}
    return VectorInterface.scale!!(deepcopy(x), α)
end

function VectorInterface.scale!!(x::ContractedSparseCores{T,N,d}, α::Number) where {T<:Number,N,d}
    α === VectorInterface.One() && return x
    return lmul!(α,x)
end

function VectorInterface.scale!!(y::ContractedSparseCores{T,N,d}, x::ContractedSparseCores{T,N,d}, α::Number) where {T<:Number,N,d}
    return mul!(y,x,α)
end

@propagate_inbounds function VectorInterface.inner(x::ContractedSparseCores{T,N,d}, y::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert x.row_ranks == y.row_ranks
    @assert x.col_ranks == y.col_ranks
  end

  s = T(0)
  for (X,Y) in zip(x.blocks, y.blocks)
    s += dot(X,Y)
  end
  return s
end

function FramedHamiltonian(H::SparseHamiltonian{T,N,d},
                               twocores_in::ContractedSparseCores{T,N,d},
                                    Fᴸ::Frame{T,N,d}, 
                                    Fᴿ::Frame{T,N,d}) where {T<:Number,N,d}
  @boundscheck begin
    @assert site(Fᴸ) == site(twocores_in) == site(Fᴿ)-2
  end
  Xₖ, Xₖ₊₁ = factor_qc(twocores_in)

  return contract( Fᴸ*H*Xₖ, H*Xₖ₊₁*Fᴿ )
  
  # row_ranks = Xₖ.row_ranks
  # col_ranks = Xₖ₊₁.col_ranks


  # # Frame matrix-free Hamiltonian operation applied to both cores
  # HXₖ = H_matvec_core(H, Xₖ)
  # HXₖ₊₁ = H_matvec_core(H, Xₖ₊₁)
  # mid_qn = Xₖ.col_qn

  # twocores_out = similar(twocores_in)

  # for m in mid_qn
  #   ql = (m-1:m)∩Xₖ.row_qn
  #   qr = (m:m+1)∩Xₖ₊₁.col_qn
  #   Nr, Nc = size(twocores_out.blocks[m])

  #   L = zeros(T, Nr, HXₖ[2][m])
  #   for l in ql
  #     rows = (l == m-1 ? (1:row_ranks[l]) : (Nr-row_ranks[l]+1:Nr))
  #     for (I,J,V) in zip(HXₖ[3],HXₖ[4],HXₖ[5])
  #       if isnonzero(V,l,m)
  #         @views mul!(L[rows,J[m]], Fᴸ[l][:,I[l]], V[l,m], 1., 1.)
  #       end
  #     end
  #   end

  #   R = zeros(T, HXₖ₊₁[1][m], Nc)
  #   for r in qr
  #     cols = (r == m   ? (1:col_ranks[r]) : (Nc-col_ranks[r]+1:Nc))
  #     for (I,J,V) in zip(HXₖ₊₁[3],HXₖ₊₁[4],HXₖ₊₁[5])
  #       if isnonzero(V,m,r)
  #         @views mul!(R[I[m],cols], V[m,r], Fᴿ[r][J[r],:], 1., 1.)
  #       end
  #     end
  #   end

  #   mul!(twocores_out.blocks[m], L, R)
  # end
  # return twocores_out
end
