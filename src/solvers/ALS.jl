using LinearAlgebra, KrylovKit

"""
  x = ALS(t::Matrix{T}, v::Array{T,4}, x0::TTvector{T,N,d}, ε::Float64)

Implementation of the Alternative Least Squares algorithm for 
approximately solving for the ground state `Hx = λx` where `H` is 
a two-body Hamiltonian given in second quantization format as:
H = Σ t_ij a†_i a_j + Σ v_ijkl a†_i a†_j a_k a_l
The result will have the same ranks as the initial guess `x0`.
"""
function ALS(t::Matrix{T}, v::Array{T,4}, x0::TTvector{T,N,d}, ε::Float64 = 1e-4, maxIter::Int = 20; reduced::Bool=false) where {T<:Number,N,d}
  @assert size(t) == (d,d) && size(v) == (d,d,d,d)

  x = deepcopy(x0)
  return ALS!(t,v,x,ε,maxIter; reduced=reduced)
end

function ALS!(t::Matrix{T}, v::Array{T,4}, x::TTvector{T,N,d}, ε::Float64, maxIter::Int; reduced::Bool=false) where {T<:Number,N,d}
  reduced || (v = Hamiltonian.reduce(v))

  # Right-orthogonalize the tensor x if necessary
  x.corePosition == 1 || rightOrthogonalize!(x, keepRank=true)

  λ = RayleighQuotient(x,t,v)
  for _ in 1:maxIter
    _ = ALSForwardSweep!(t,v,x)
    λn = ALSBackSweep!(t,v,x)
    r = abs(λ-λn)/abs(λn)
    λ = λn

    r < ε && break
  end

  return  λ, x
end

function ALSForwardSweep!(t::Matrix{T}, v::Array{T,4}, x::TTvector{T,N,d}, inner_tol::T = 1e-8) where {T<:Number,N,d}
  x.corePosition == 1 || rightOrthogonalize!(x, keepRank=true) # Right-orthogonalize the tensor x
  Wᴿ = RightToLeftFraming(t,v,x)

  λ = T(0)
  Wᴸ = OffsetVector([n == 0 ? ones(T,1,1) : zeros(T,0,0) for n in 0:N], 0:N)
  for k=1:d
    # Compute new core with iterative Lanczos
    vals, vecs, info = KrylovKit.eigsolve(
                          Xk -> FramedHamiltonian(Xk,t,v,Wᴸ,Wᴿ[k]), 
                          x.cores[k], 1, :SR, 
                          issymmetric=true, tol=inner_tol, verbosity=0)
    λ = vals[1]
    x.cores[k] = vecs[1]

    if k < d 
      # orthogonalize the new core, moving the tensor core to position i+1.
      move_core!(x, k+1, keepRank=true)
      # Compute the new left frame matrix Wᴸ
      Wᴸ = FramingStepRight(t,v,x,k,Wᴸ)
    end
  end
  return λ
end


function ALSBackSweep!(t::Matrix{T}, v::Array{T,4}, x::TTvector{T,N,d}, inner_tol::T = 1e-8) where {T<:Number,N,d}
  x.corePosition == d || leftOrthogonalize!(x, keepRank=true) # Right-orthogonalize the tensor x
  Wᴸ = LeftToRightFraming(t,v,x)

  λ = T(0)
  Wᴿ = OffsetVector([n == N ? ones(T,1,1) : zeros(T,0,0) for n in 0:N], 0:N)
  for k=d:-1:1
    # Compute new core with iterative Lanczos
    vals, vecs, info = KrylovKit.eigsolve(
                          Xk -> FramedHamiltonian(Xk,t,v,Wᴸ[k],Wᴿ), 
                          x.cores[k], 1, :SR, 
                          issymmetric=true, tol=inner_tol, verbosity=0)
    λ = vals[1]
    x.cores[k] = vecs[1]
    if k > 1
      # orthogonalize the new core, moving the tensor core to position i-1.
      move_core!(x, k-1, keepRank=true)
      # Compute the new left frame matrix Wᴿ
      Wᴿ = FramingStepLeft(t,v,x,k,Wᴿ)
    end
  end
  return λ
end

function FramedHamiltonian( Xₖ::SparseCore{T,N,d},
                             t::Matrix{T}, 
                             v::Array{T,4},
                            Wᴸ::OffsetVector{Matrix{T},Vector{Matrix{T}}}, 
                            Wᴿ::OffsetVector{Matrix{T},Vector{Matrix{T}}}) where {T<:Number,N,d}
  HXₖ = sparse_H_matvec(Xₖ, t, v)
  Yₖ = similar(Xₖ)
  for l in axes(Yₖ,1), r in axes(Yₖ,3) ∩ (l:l+1)
    Yₖ[l,r] = zeros(T,Yₖ.row_ranks[l],Yₖ.col_ranks[r])
    for (I,J,V) in zip(HXₖ[3],HXₖ[4],HXₖ[5])
      if isnonzero(V,l,r)
        W = Wᴸ[l][:,I[l]] * V[l,r] * Wᴿ[r][J[r],:]
        Yₖ[l,r] += W
      end
    end
  end
  return Yₖ
end

function LeftToRightFraming(t::Matrix{T}, v::Array{T,4}, x::TTvector{T,N,d}) where {T<:Number,N,d}
  # Recursive contractions to compute frame matrices and vectors from the left
  Wᴸ = Vector{OffsetVector{Matrix{T},Vector{Matrix{T}}}}(undef, d)
  Wᴸ[1] = OffsetVector([n == 0 ? ones(T,1,1) : zeros(T,0,0) for n in 0:N], 0:N)
  for k=2:d
    Wᴸ[k] = FramingStepRight(t,v,x,k-1,Wᴸ[k-1])
  end

  return Wᴸ
end

function RightToLeftFraming(t::Matrix{T}, v::Array{T,4}, x::TTvector{T,N,d}) where {T<:Number,N,d}
  # Recursive contractions to compute frame matrices and vectors from the right
  Wᴿ = Vector{OffsetVector{Matrix{T},Vector{Matrix{T}}}}(undef, d)
  Wᴿ[d] = OffsetVector([n == N ? ones(T,1,1) : zeros(T,0,0) for n in 0:N], 0:N)
  for k=d-1:-1:1
    Wᴿ[k] = FramingStepLeft(t,v,x,k+1,Wᴿ[k+1])
  end

  return Wᴿ
end

function FramingStepRight(t::Matrix{T}, v::Array{T,4}, 
                          x::TTvector{T,N,d}, k::Int, 
                          Wᴸ::OffsetVector{Matrix{T},Vector{Matrix{T}}}) where {T<:Number,N,d}
    Xₖ = x.cores[k]
    HXₖ = sparse_H_matvec(Xₖ,t,v)

    Wᴸ⁺¹ = OffsetVector([Matrix{T}(undef,0,0) for l in 0:N], 0:N)
    for r in axes(Xₖ,3)
      Wᴸ⁺¹[r] = zeros(T, Xₖ.col_ranks[r], HXₖ[2][r])
      for l in axes(Xₖ,1) ∩ (r-1:r)
        if isnonzero(Xₖ,l,r)
          for (I,J,V) in zip(HXₖ[3],HXₖ[4],HXₖ[5])
            if isnonzero(V,l,r)
              mul!( view(Wᴸ⁺¹[r], :, J[r]), 
                    adjoint(data(Xₖ[l,r])) * Wᴸ[l][:,I[l]],
                    data(V[l,r]),
                    conj(factor(Xₖ[l,r])) * factor(V[l,r]),
                    T(1)
                  )
            end
          end
        end
      end
    end

    return Wᴸ⁺¹
end

function FramingStepLeft(t::Matrix{T}, v::Array{T,4}, 
                          x::TTvector{T,N,d}, k::Int, 
                          Wᴿ::OffsetVector{Matrix{T},Vector{Matrix{T}}}) where {T<:Number,N,d}
    Xₖ = x.cores[k]
    HXₖ = sparse_H_matvec(Xₖ,t,v)

    Wᴿ⁻¹ = OffsetVector([Matrix{T}(undef,0,0) for l in 0:N], 0:N)
    for l in axes(Xₖ,1)
      Wᴿ⁻¹[l] = zeros(T, HXₖ[1][l], Xₖ.row_ranks[l])
      for r in axes(Xₖ,3) ∩ (l:l+1)
        if isnonzero(Xₖ,l,r)
          for (I,J,V) in zip(HXₖ[3],HXₖ[4],HXₖ[5])
            if isnonzero(V,l,r)
              mul!( view(Wᴿ⁻¹[l], I[l], :),
                    data(V[l,r]), 
                    Wᴿ[r][J[r],:] * adjoint(data(Xₖ[l,r])),
                    factor(V[l,r]) * conj(factor(Xₖ[l,r])),
                    T(1)
                  )
            end
          end
        end
      end
    end

    return Wᴿ⁻¹
end
