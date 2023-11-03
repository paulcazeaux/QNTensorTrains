"""
  x = MALS(t::Matrix{T}, v::Array{T,4}, x0::TTtensor{T,d}, ε::Float64)

Implementation of the Modified Alternative Least Squares,
approximately solving for the ground state `Hx = λx` where `H` is 
a two-body Hamiltonian given in second quantization format as:
H = Σ t_ij a†_i a_j + Σ v_ijkl a†_i a†_j a_k a_l
The algorithm adapts ranks: the result will not necessarily have the
same ranks as the initial guess `x0`.
"""
function MALS(t::Matrix{T}, v::Array{T,4}, x0::TTvector{T,N,d}, ε::Float64 = 1e-4, maxIter::Int = 20; reduced::Bool=false) where {T<:Number,N,d}
  @boundscheck @assert size(t) == (d,d) && size(v) == (d,d,d,d)

  x = deepcopy(x0)
  λ, x = MALS!(t,v,x,ε,maxIter)

  return λ, x
end

function MALS!(t::Matrix{T}, v::Array{T,4}, x::TTvector{T,N,d}, ε::Float64, maxIter::Int; reduced::Bool=false) where {T<:Number,N,d}
  reduced || (v = Hamiltonian.reduce(v))
  # Right-orthogonalize the tensor x if necessary
  x.corePosition == 1 || rightOrthogonalize!(x, keepRank=true) 

  λ = RayleighQuotient(x,t,v)
  for it in 1:maxIter
    _ = MALSForwardSweep!(t,v,x,ε)
    λn = MALSBackSweep!(t,v,x,ε)
    r = abs(λ-λn)/abs(λn)
    λ = λn
    r < ε && break
  end

  return λ, x
end


function MALSForwardSweep!(t::Matrix{T}, v::Array{T,4}, x::TTvector{T,N,d}, ε::Float64, inner_tol::T = 1e-6) where {T<:Number,N,d}
  move_core!(x, 1; keepRank=true) # Right-orthogonalize the tensor x
  Wᴿ = RightToLeftFraming(t,v,x)
  Wᴸ = OffsetVector([n == 0 ? ones(T,1,1) : zeros(T,0,0) for n in 0:N], 0:N)

  λ = T(0)
  ep = ε/sqrt(d-1)
  for k=1:d-1
    # Assumption: `x` is orthogonal with core at `k`
    @boundscheck @assert x.corePosition == k
    # Compute new cores.
    vals, vecs, info = KrylovKit.eigsolve(
                          core_pair -> FramedHamiltonian(core_pair,t,v,Wᴸ,Wᴿ[k+1]), 
                          contract(x.cores[k],x.cores[k+1]), 1, :SR;
                          krylovdim=5,
                          issymmetric=true, tol=inner_tol, verbosity=0)

    λ = vals[1]
    S = factor_svd!(x.cores[k],x.cores[k+1],vecs[1], ep)
    lmul!(Diagonal.(S),x.cores[k+1]) 
    rank(x,k+1) .= x.cores[k].col_ranks

    x.corePosition = k+1
    if k < d-1 
      # Compute the new left frame matrix Wᴸ
      Wᴸ = FramingStepRight(t,v,x,k,Wᴸ)
    end
  end

  return λ
end

function MALSBackSweep!(t::Matrix{T}, v::Array{T,4}, x::TTvector{T,N,d}, ε::Float64, inner_tol::T = 1e-6) where {T<:Number,N,d}
  move_core!(x, d-1; keepRank=false) # Right-orthogonalize the tensor x
  Wᴸ = LeftToRightFraming(t,v,x)
  Wᴿ = OffsetVector([n == N ? ones(T,1,1) : zeros(T,0,0) for n in 0:N], 0:N)
  Wᴿ = FramingStepLeft(t,v,x,d,Wᴿ)

  λ = T(0)
  ep = ε/sqrt(d-1)
  for k=d-2:-1:1
    # Assumption: `x` is orthogonal with core at `k+1`
    @boundscheck @assert x.corePosition == k+1

    # Compute new cores.
    vals, vecs, info = KrylovKit.eigsolve(
                          core_pair -> FramedHamiltonian(core_pair,t,v,Wᴸ[k],Wᴿ), 
                          contract(x.cores[k],x.cores[k+1]), 1, :SR;
                          krylovdim=5,
                          issymmetric=true, tol=inner_tol, verbosity=0)

    λ = vals[1]
    S = factor_svd!(x.cores[k],x.cores[k+1],vecs[1], ep)
    rmul!(x.cores[k], Diagonal.(S)) 
    rank(x,k+1) .= x.cores[k].col_ranks

    x.corePosition = k
    if k>1 
      # Compute the new right frame matrix Wᴿ
      Wᴿ = FramingStepLeft(t,v,x,k+1,Wᴿ)
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
end

function Base.size(v::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  return (v.m,v.n)
end

function Base.similar(v::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  return ContractedSparseCores{T,N,d}(v.k, v.m, v.n, 
                                      v.row_qn, v.col_qn, 
                                      v.row_ranks, 
                                      v.col_ranks,
                                      similar.(v.blocks))
end

function Base.:*(α::Number, v::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  return ContractedSparseCores{T,N,d}(v.k, v.m, v.n, 
                                      v.row_qn, v.col_qn, 
                                      v.row_ranks, 
                                      v.col_ranks,
                                      α .* v.blocks)
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

  axpy!.(α,v.blocks,β,w.blocks)
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
  m = Xₖ.m
  n = Xₖ₊₁.n

  row_qn = Xₖ.row_qn
  mid_qn = Xₖ.col_qn
  col_qn = Xₖ₊₁.col_qn

  row_ranks = Xₖ.row_ranks
  col_ranks = Xₖ₊₁.col_ranks

  blocks = [Matrix{T}(undef,0,0) for m in mid_qn]
  for m in mid_qn
    ql = (m-1:m)∩Xₖ.row_qn
    qr = (m:m+1)∩Xₖ₊₁.col_qn

    Nr, Nc = sum(row_ranks[ql]), sum(col_ranks[qr])

    blocks[m] = zeros(T,Nr,Nc)
    for l in ql, r in qr
      rows = (l == m-1 ? (1:row_ranks[l]) : (Nr-row_ranks[l]+1:Nr))
        cols = (r == m   ? (1:col_ranks[r]) : (Nc-col_ranks[r]+1:Nc))
        contract!(view(blocks[m],rows,cols), Xₖ[l,m], Xₖ₊₁[m,r])
    end
  end

  twocores = ContractedSparseCores{T,N,d}(k,m,n,row_qn,col_qn,row_ranks,col_ranks,blocks)
  return twocores
end

function factor_qc(twocores::ContractedSparseCores{T,N,d}) where {T<:Number,N,d}
  k = twocores.k
  Xₖ = SparseCore{T,N,d}(k)
  Xₖ₊₁ = SparseCore{T,N,d}(k+1)
  mid_qn = Xₖ.col_qn
  @boundscheck @assert Xₖ₊₁.row_qn == mid_qn
  Xₖ.row_ranks .= twocores.row_ranks
  Xₖ₊₁.col_ranks .= twocores.col_ranks

  for m in mid_qn
    Q,C,rank = my_qc!(copy(twocores.blocks[m]))
    Xₖ.col_ranks[m] = rank
    Xₖ₊₁.row_ranks[m] = rank
    Xₖ[m,:vertical] = Q
    Xₖ₊₁[m,:horizontal] = C
  end

  return Xₖ, Xₖ₊₁
end

function factor_svd!(Xₖ::SparseCore{T,N,d}, Xₖ₊₁::SparseCore{T,N,d}, twocores::ContractedSparseCores{T,N,d}, ep::Float64) where {T<:Number,N,d}
  k = twocores.k
  mid_qn = Xₖ.col_qn

  @boundscheck begin
    @assert Xₖ.k == k && Xₖ₊₁.k == k+1
    @assert Xₖ.row_ranks == twocores.row_ranks
    @assert Xₖ₊₁.col_ranks == twocores.col_ranks
    @assert Xₖ₊₁.row_qn == mid_qn
  end

  U   = [zeros(T,0,0) for m in mid_qn]
  S   = [zeros(T,0  ) for m in mid_qn]
  Vt  = [zeros(T,0,0) for m in mid_qn]
  for m in mid_qn
    F = svd!(twocores.blocks[m])
    U[m],S[m],Vt[m] = F.U, F.S, F.Vt
  end
  ranks = chop(S, ep)
  s = norm(norm.(S))

  for m in mid_qn
    r = ranks[m]
    S[m] = S[m][1:r] ./ s
    Xₖ.col_ranks[m] = r
    Xₖ[m,:vertical] = U[m][:,1:r]
    Xₖ₊₁.row_ranks[m] = r
    Xₖ₊₁[m,:horizontal] = Vt[m][1:r,:]
  end

  return S
end

function FramedHamiltonian(twocores_in::ContractedSparseCores{T,N,d},
                                     t::Matrix{T}, 
                                     v::Array{T,4},
                                    Wᴸ::OffsetVector{Matrix{T},Vector{Matrix{T}}}, 
                                    Wᴿ::OffsetVector{Matrix{T},Vector{Matrix{T}}}) where {T<:Number,N,d}
  Xₖ, Xₖ₊₁ = factor_qc(twocores_in)
  row_ranks = Xₖ.row_ranks
  col_ranks = Xₖ₊₁.col_ranks


  # Frame matrix-free Hamiltonian operation applied to both cores
  HXₖ = sparse_H_matvec(Xₖ, t, v)
  HXₖ₊₁ = sparse_H_matvec(Xₖ₊₁, t, v)
  mid_qn = Xₖ.col_qn

  twocores_out = similar(twocores_in)

  for m in mid_qn
    ql = (m-1:m)∩Xₖ.row_qn
    qr = (m:m+1)∩Xₖ₊₁.col_qn
    Nr, Nc = size(twocores_out.blocks[m])

    L = zeros(T, Nr, HXₖ[2][m])
    for l in ql
      rows = (l == m-1 ? (1:row_ranks[l]) : (Nr-row_ranks[l]+1:Nr))
      for (I,J,V) in zip(HXₖ[3],HXₖ[4],HXₖ[5])
        if isnonzero(V,l,m)
          @views mul!(L[rows,J[m]], Wᴸ[l][:,I[l]], V[l,m], 1., 1.)
        end
      end
    end

    R = zeros(T, HXₖ₊₁[1][m], Nc)
    for r in qr
      cols = (r == m   ? (1:col_ranks[r]) : (Nc-col_ranks[r]+1:Nc))
      for (I,J,V) in zip(HXₖ₊₁[3],HXₖ₊₁[4],HXₖ₊₁[5])
        if isnonzero(V,m,r)
          @views mul!(R[I[m],cols], V[m,r], Wᴿ[r][J[r],:], 1., 1.)
        end
      end
    end

    mul!(twocores_out.blocks[m], L, R)
  end
  return twocores_out
end
