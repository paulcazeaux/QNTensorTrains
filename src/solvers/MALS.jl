"""
  x = MALS(H::SparseHamiltonian{T,Nup,Ndn,d}, x0::TTtensor{T,d}, ε::Float64)

Implementation of the Modified Alternative Least Squares,
approximately solving for the ground state `Hx = λx` where `H` is 
a two-body Hamiltonian given in second quantization format as:
H = Σ t_ij a†_i a_j + Σ v_ijkl a†_i a†_j a_k a_l
The algorithm adapts ranks: the result will not necessarily have the
same ranks as the initial guess `x0`.
"""
function MALS(H::SparseHamiltonian{T,Nup,Ndn,d}, x0::TTvector{T,Nup,Ndn,d}, ε::Float64 = 1e-4, maxIter::Int = 20) where {T<:Number,Nup,Ndn,d}
  x = deepcopy(x0)
  λ, x = MALS!(H,x,ε,maxIter)

  return λ, x
end

function MALS!(H::SparseHamiltonian{T,Nup,Ndn,d}, x::TTvector{T,Nup,Ndn,d}, ε::Float64, maxIter::Int) where {T<:Number,Nup,Ndn,d}
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


function MALSForwardSweep!(H::SparseHamiltonian{T,Nup,Ndn,d}, x::TTvector{T,Nup,Ndn,d}, ε::Float64, inner_tol::T = 1e-6) where {T<:Number,Nup,Ndn,d}
  move_core!(x, 1; keepRank=true) # Right-orthogonalize the tensor x
  Fᴿ = RightToLeftFraming(H,x)
  Fᴸ = IdFrame(Val(d), Val(Nup), Val(Ndn), 1)

  λ = T(0)
  ep = ε/sqrt(d-1)
  for k=1:d-1
    # Assumption: `x` is orthogonal with core at `k`
    @boundscheck @assert x.corePosition == k
    # Compute new cores.
    x₀ = contract(x.cores[k],x.cores[k+1])
    vals, vecs, info = KrylovKit.eigsolve(
                          core_pair -> FramedHamiltonian(H, core_pair,Fᴸ,Fᴿ[k+1]), 
                          contract(x.cores[k],x.cores[k+1]), 1, :SR;
                          krylovdim=5,
                          issymmetric=true, tol=inner_tol, verbosity=0)

    λ = vals[1]
    Xₖ, S, Xₖ₊₁ = factor_svd!(vecs[1], ep)
    lmul!(S,Xₖ₊₁) 
    set_cores!(x, Xₖ, Xₖ₊₁)

    x.corePosition = k+1
    if k < d-1 
      # Compute the new left frame matrix Fᴸ
      Fᴸ = FramingStepRight(H,x,Fᴸ)
    end
  end

  return λ
end

function MALSBackSweep!(H::SparseHamiltonian{T,Nup,Ndn,d}, x::TTvector{T,Nup,Ndn,d}, ε::Float64, inner_tol::T = 1e-6) where {T<:Number,Nup,Ndn,d}
  move_core!(x, d-1; keepRank=false) # Right-orthogonalize the tensor x
  Fᴸ = LeftToRightFraming(H,x)
  Fᴿ = IdFrame(Val(d), Val(Nup), Val(Ndn), d+1)
  Fᴿ = FramingStepLeft(H,x,Fᴿ)

  λ = T(0)
  ep = ε/sqrt(d-1)
  for k=d-2:-1:1
    # Assumption: `x` is orthogonal with core at `k+1`
    @boundscheck @assert x.corePosition == k+1

    # Compute new cores.
    x₀ = contract(x.cores[k],x.cores[k+1])
    vals, vecs, info = KrylovKit.eigsolve(
                          core_pair -> FramedHamiltonian(H,core_pair,Fᴸ[k],Fᴿ), 
                          contract(x.cores[k],x.cores[k+1]), 1, :SR;
                          krylovdim=5,
                          issymmetric=true, tol=inner_tol, verbosity=0)

    λ = vals[1]
    Xₖ, S, Xₖ₊₁ = factor_svd!(vecs[1], ep)
    rmul!(Xₖ, S)
    set_cores!(x, Xₖ , Xₖ₊₁)

    x.corePosition = k
    if k>1 
      # Compute the new right frame matrix Fᴿ
      Fᴿ = FramingStepLeft(H,x,Fᴿ)
    end
  end

  return λ
end

# Special structure for the contraction of two successive cores
struct ContractedSparseCores{T<:Number,Nup,Ndn,d} <: AbstractArray{Matrix{T},4}
  k         :: Int        # Core indices should be k and k+1

  m         :: Int        # row size
  n         :: Int        # column size

  row_qn    :: Vector{Tuple{Int,Int}}
  mid_qn    :: Vector{Tuple{Int,Int}}
  col_qn    :: Vector{Tuple{Int,Int}}

  row_ranks :: Matrix{Int}
  col_ranks :: Matrix{Int}

  blocks    :: Matrix{Matrix{T}}

  mem       :: Memory{T}
end

function ContractedSparseCores{T,Nup,Ndn,d}( k::Int,
              row_ranks::Matrix{Int},
              col_ranks::Matrix{Int}) where {T<:Number,Nup,Ndn,d}
  row_qn = state_qn(Nup,Ndn,d,k)
  mid_qn = state_qn(Nup,Ndn,d,k+1)
  col_qn = state_qn(Nup,Ndn,d,k+2)

  m = length(row_qn)
  n = length(col_qn)

  shapes = Matrix{NTuple{2,Int}}(undef, Nup+1,Ndn+1)
  for (mup,mdn) in mid_qn
    m₋ = ( (mup-1,mdn-1), (mup,mdn-1), (mup-1,mdn), (mup,mdn))
    m₊ = ( (mup,mdn), (mup+1,mdn), (mup,mdn+1), (mup+1,mdn+1))

    shapes[mup,mdn] = (sum(row_ranks[lup,ldn] for (lup,ldn) in m₋ if (lup,ldn) in row_qn),  
                       sum(col_ranks[rup,rdn] for (rup,rdn) in m₊ if (rup,rdn) in col_qn)) 
  end

  mem = Memory{T}(undef, sum(prod(shapes[mup,mdn]) for (mup,mdn) in mid_qn))
  mem .= T(0)
  idx = 1

  blocks = Matrix{Matrix{T}}(undef, Nup+1,Ndn+1)
  for (mup,mdn) in mid_qn
    (row_rank, col_rank) = shapes[mup,mdn]
    blocks[mup,mdn] = Block(row_rank,col_rank,mem,idx)
    idx += length(blocks[mup,mdn])
  end

  return ContractedSparseCores{T,Nup,Ndn,d}( k, m, n, 
                                      row_qn, mid_qn, col_qn, 
                                      deepcopy(row_ranks), 
                                      deepcopy(col_ranks),
                                      blocks, mem)
end

@inline function Base.size(v::ContractedSparseCores)
  return (v.m,v.n)
end

function Base.similar(v::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return ContractedSparseCores{T,Nup,Ndn,d}(v.k, v.row_ranks, v.col_ranks)
end

@inline function site(v::ContractedSparseCores)
  return v.k
end

function Base.:*(α::Number, v::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  w = similar(v)
  for (mup,mdn) in w.mid_qn
    w.blocks[mup,mdn] .*= α
  end
  return w
end

function LinearAlgebra.mul!(w::ContractedSparseCores{T,Nup,Ndn,d}, v::ContractedSparseCores{T,Nup,Ndn,d}, α::Number) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert w.k == v.k
    @assert w.m == v.m
    @assert w.n == v.n
    @assert w.row_qn == v.row_qn
    @assert w.col_qn == v.col_qn
    @assert w.row_ranks == v.row_ranks
    @assert w.col_ranks == v.col_ranks
  end

  for (mup,mdn) in w.mid_qn
    mul!(w.blocks[mup,mdn], α, v.blocks[mup,mdn])
  end
  return w
end

function LinearAlgebra.lmul!(α::Number, v::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  for (mup,mdn) in v.mid_qn
    lmul!(α, v.blocks[mup,mdn])
  end
  return v
end

function LinearAlgebra.rmul!(v::ContractedSparseCores{T,Nup,Ndn,d}, α::Number) where {T<:Number,Nup,Ndn,d}
  for (mup,mdn) in v.mid_qn
    rmul!(v.blocks[mup,mdn], α)
  end
  return v
end

function LinearAlgebra.axpy!(α::Number, v::ContractedSparseCores{T,Nup,Ndn,d}, w::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert w.k == v.k
    @assert w.m == v.m
    @assert w.n == v.n
    @assert w.row_qn == v.row_qn
    @assert w.col_qn == v.col_qn
    @assert w.row_ranks == v.row_ranks
    @assert w.col_ranks == v.col_ranks
  end
  for (mup,mdn) in v.mid_qn
    axpy!(α,v.blocks[mup,mdn],w.blocks[mup,mdn])
  end
  return w
end

function LinearAlgebra.axpby!(α::Number, v::ContractedSparseCores{T,Nup,Ndn,d}, β::Number, w::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert w.k == v.k
    @assert w.m == v.m
    @assert w.n == v.n
    @assert w.row_qn == v.row_qn
    @assert w.col_qn == v.col_qn
    @assert w.row_ranks == v.row_ranks
    @assert w.col_ranks == v.col_ranks
  end

  for (mup,mdn) in v.mid_qn
    axpby!(α,v.blocks[mup,mdn],β,w.blocks[mup,mdn])
  end
  return w
end

function LinearAlgebra.norm(x::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  s = T(0)
  for (mup,mdn) in x.mid_qn
    s += sum(abs2, x.blocks[mup,mdn])
  end
  return sqrt(s)
end

function LinearAlgebra.dot(x::ContractedSparseCores{T,Nup,Ndn,d}, y::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert x.row_ranks == y.row_ranks
    @assert x.col_ranks == y.col_ranks
  end
  s = T(0)
  for (mup,mdn) in x.mid_qn
    s += dot(x.blocks[mup,mdn], y.blocks[mup,mdn])
  end
  return s
end

function contract(Xₖ::SparseCore{T,Nup,Ndn,d}, Xₖ₊₁::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert Xₖ.k+1 == Xₖ₊₁.k
    @assert Xₖ.col_ranks == Xₖ₊₁.row_ranks
  end
  k = Xₖ.k
  row_ranks = Xₖ.row_ranks
  col_ranks = Xₖ₊₁.col_ranks
  twocores = ContractedSparseCores{T,Nup,Ndn,d}(k,row_ranks,col_ranks)
  @assert twocores.row_ranks == row_ranks
  @assert twocores.col_ranks == col_ranks

  for (mup,mdn) in col_qn(Xₖ)
    l = ( (mup,mdn), (mup-1,mdn), (mup,mdn-1), (mup-1,mdn-1)) ∩ row_qn(Xₖ)
    r = ( (mup,mdn), (mup+1,mdn), (mup,mdn+1), (mup+1,mdn+1)) ∩ col_qn(Xₖ₊₁)

    rows = cumsum(row_ranks[lup,ldn] for (lup,ldn) in l)
    cols = cumsum(col_ranks[rup,rdn] for (rup,rdn) in r)
    for (i, (lup,ldn)) in enumerate(l), (j, (rup,rdn)) in enumerate(r)
      mul!(view(twocores.blocks[mup,mdn],(i==1 ? 1 : rows[i-1]+1):rows[i],
                                         (j==1 ? 1 : cols[j-1]+1):cols[j]), 
              Xₖ[(lup,ldn),(mup,mdn)], Xₖ₊₁[(mup,mdn),(rup,rdn)])
    end
  end

  return twocores
end

function factor_qc(twocores::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  k = twocores.k

  Q = Matrix{Matrix{T}}(undef,   Nup+1,Ndn+1)
  C = Matrix{Matrix{T}}(undef,   Nup+1,Ndn+1)
  mid_rank = zeros(Int,Nup+1,Ndn+1)
  for (mup,mdn) in twocores.mid_qn
    Q[mup,mdn],C[mup,mdn],mid_rank[mup,mdn] = my_qc!(copy(twocores.blocks[mup,mdn]))
  end
  Xₖ = SparseCore{T,Nup,Ndn,d}(k, twocores.row_ranks, mid_rank)
  Xₖ₊₁ = SparseCore{T,Nup,Ndn,d}(k+1, mid_rank, twocores.col_ranks)

  for (mup,mdn) in twocores.mid_qn
    Xₖ[(mup,mdn),:vertical] = Q[mup,mdn]
    Xₖ₊₁[(mup,mdn),:horizontal] = C[mup,mdn]
  end

  y = contract(Xₖ, Xₖ₊₁)
  for (mup,mdn) in y.mid_qn
    y.blocks[mup,mdn] -= twocores.blocks[mup,mdn]
  end

  return Xₖ, Xₖ₊₁
end

function factor_svd!(twocores::ContractedSparseCores{T,Nup,Ndn,d}, ep::Float64) where {T<:Number,Nup,Ndn,d}
  k = twocores.k

  U  = Matrix{Matrix{T}}(undef,             Nup+1,Ndn+1)
  S  = Matrix{Diagonal{T,Vector{T}}}(undef, Nup+1,Ndn+1)
  Vt = Matrix{Matrix{T}}(undef,             Nup+1,Ndn+1)
  mid_rank = zeros(Int,Nup+1,Ndn+1)

  for (mup,mdn) in twocores.mid_qn
    F = svd!(twocores.blocks[mup,mdn])
    U[mup,mdn],S[mup,mdn],Vt[mup,mdn] = F.U, Diagonal(F.S), F.Vt
  end
  S = Frame{T,Nup,Ndn,d}(k+1,S)
  mid_rank = chop(S, ep)
  s = norm(S)

  Xₖ   = SparseCore{T,Nup,Ndn,d}(k,   twocores.row_ranks, mid_rank)
  Xₖ₊₁ = SparseCore{T,Nup,Ndn,d}(k+1, mid_rank, twocores.col_ranks)
  for (mup,mdn) in twocores.mid_qn
    r = mid_rank[mup,mdn]
    S[(mup,mdn)] = Diagonal(diag(block(S,mup,mdn))[1:r] ./ s)
    Xₖ[(mup,mdn),:vertical]     = U[mup,mdn][:,1:r]
    Xₖ₊₁[(mup,mdn),:horizontal] = Vt[mup,mdn][1:r,:]
  end

  return Xₖ, S, Xₖ₊₁
end

function VectorInterface.zerovector(x::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  y = similar(x)
  for (mup,mdn) in x.mid_qn
    fill!(y.blocks[mup,mdn], T(0))
  end
  return y
end

function VectorInterface.zerovector(x::ContractedSparseCores{S,Nup,Ndn,d}, T::Type{<:Number}) where {S<:Number,Nup,Ndn,d}
  return ContractedSparseCores{T,Nup,Ndn,d}( x.k, x.row_ranks, x.col_ranks)
end

function VectorInterface.add!!(y::ContractedSparseCores{T,Nup,Ndn,d}, x::ContractedSparseCores{T,Nup,Ndn,d}, α::Number, β::Number) where {T<:Number,Nup,Ndn,d}
  axpby!(α,x,β,y)
end

function VectorInterface.scale(x::ContractedSparseCores{T,Nup,Ndn,d}, α::Number) where {T<:Number,Nup,Ndn,d}
    return VectorInterface.scale!!(deepcopy(x), α)
end

function VectorInterface.scale!!(x::ContractedSparseCores{T,Nup,Ndn,d}, α::Number) where {T<:Number,Nup,Ndn,d}
    α === VectorInterface.One() && return x
    return lmul!(α,x)
end

function VectorInterface.scale!!(y::ContractedSparseCores{T,Nup,Ndn,d}, x::ContractedSparseCores{T,Nup,Ndn,d}, α::Number) where {T<:Number,Nup,Ndn,d}
    return mul!(y,x,α)
end

@propagate_inbounds function VectorInterface.inner(x::ContractedSparseCores{T,Nup,Ndn,d}, y::ContractedSparseCores{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  @boundscheck begin
    @assert x.row_ranks == y.row_ranks
    @assert x.col_ranks == y.col_ranks
  end

  s = T(0)
  for (mup,mdn) in x.mid_qn
    s += dot(x.blocks[mup,mdn],y.blocks[mup,mdn])
  end
  return s
end

function FramedHamiltonian(H::SparseHamiltonian{T,Nup,Ndn,d},
                               twocores_in::ContractedSparseCores{T,Nup,Ndn,d},
                                    Fᴸ::Frame{T,Nup,Ndn,d}, 
                                    Fᴿ::Frame{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
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
