using LinearAlgebra, KrylovKit

"""
  x = ALS(H::SparseHamiltonian{T,Nup,Ndn,d}, x0::TTvector{T,Nup,Ndn,d}, ε::Float64)

Implementation of the Alternative Least Squares algorithm for 
approximately solving for the ground state `Hx = λx` where `H` is 
a two-body Hamiltonian given in second quantization format as:
H = Σ t_ij a†_i a_j + Σ v_ijkl a†_i a†_j a_k a_l
The result will have the same ranks as the initial guess `x0`.
"""
function ALS(H::SparseHamiltonian{T,Nup,Ndn,d}, x0::TTvector{T,Nup,Ndn,d}, ε::Float64 = 1e-4, maxIter::Int = 20) where {T<:Number,Nup,Ndn,d}
  x = deepcopy(x0)
  return ALS!(H,x,ε,maxIter)
end

function ALS!(H::SparseHamiltonian{T,Nup,Ndn,d}, x::TTvector{T,Nup,Ndn,d}, ε::Float64, maxIter::Int) where {T<:Number,Nup,Ndn,d}
  # Right-orthogonalize the tensor x if necessary
  x.corePosition == 1 || rightOrthogonalize!(x, keepRank=true)

  λ = RayleighQuotient(H,x)
  for _ in 1:maxIter
    _ = ALSForwardSweep!(H,x)
    λn = ALSBackSweep!(H,x)
    r = abs(λ-λn)/abs(λn)
    λ = λn

    r < ε && break
  end

  return  λ, x
end

function ALSForwardSweep!(H::SparseHamiltonian{T,Nup,Ndn,d}, x::TTvector{T,Nup,Ndn,d}, inner_tol::T = 1e-8) where {T<:Number,Nup,Ndn,d}
  x.corePosition == 1 || rightOrthogonalize!(x, keepRank=true) # Right-orthogonalize the tensor x
  Fᴿ = RightToLeftFraming(H,x)

  λ = T(0)
  Fᴸ = IdFrame(Val(d), Val(Nup), Val(Ndn), 1)
  for k=1:d
    # Compute new core with iterative Lanczos
    vals, vecs, info = KrylovKit.eigsolve(
                          Xk -> FramedHamiltonian(H,Xk,Fᴸ,Fᴿ[k]), 
                          x.cores[k], 1, :SR, 
                          issymmetric=true, tol=inner_tol, verbosity=0)
    λ = vals[1]
    x.cores[k] = vecs[1]

    if k < d 
      # orthogonalize the new core, moving the tensor core to position i+1.
      move_core!(x, k+1, keepRank=true)
      # Compute the new left frame matrix Fᴸ
      Fᴸ = FramingStepRight(H,x,Fᴸ)
    end
  end
  return λ
end


function ALSBackSweep!(H::SparseHamiltonian{T,Nup,Ndn,d}, x::TTvector{T,Nup,Ndn,d}, inner_tol::T = 1e-8) where {T<:Number,Nup,Ndn,d}
  x.corePosition == d || leftOrthogonalize!(x, keepRank=true) # Right-orthogonalize the tensor x
  Fᴸ = LeftToRightFraming(H,x)

  λ = T(0)
  Fᴿ = IdFrame(Val(d), Val(Nup), Val(Ndn), d+1)
  for k=d:-1:1
    # Compute new core with iterative Lanczos
    vals, vecs, info = KrylovKit.eigsolve(
                          Xk -> FramedHamiltonian(H, Xk, Fᴸ[k],Fᴿ), 
                          x.cores[k], 1, :SR, 
                          issymmetric=true, tol=inner_tol, verbosity=0)
    λ = vals[1]
    x.cores[k] = vecs[1]
    if k > 1
      # orthogonalize the new core, moving the tensor core to position i-1.
      move_core!(x, k-1, keepRank=true)
      # Compute the new left frame matrix Fᴿ
      Fᴿ = FramingStepLeft(H,x,Fᴿ)
    end
  end
  return λ
end

function FramedHamiltonian( H::SparseHamiltonian{T,Nup,Ndn,d},
                            Xₖ::SparseCore{T,Nup,Ndn,d},
                            Fᴸ::Frame{T,Nup,Ndn,d}, 
                            Fᴿ::Frame{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  return (Fᴸ*H*Xₖ) * Fᴿ
end

function LeftToRightFraming(H::SparseHamiltonian{T,Nup,Ndn,d}, x::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  # Recursive contractions to compute frame matrices and vectors from the left
  Fᴸ = Vector{Frame{T,Nup,Ndn,d,Matrix{T}}}(undef, d)
  Fᴸ[1] = IdFrame(Val(d), Val(Nup), Val(Ndn), 1)
  for k=2:d
    Fᴸ[k] = FramingStepRight(H,x,Fᴸ[k-1])
  end

  return Fᴸ
end

function RightToLeftFraming(H::SparseHamiltonian{T,Nup,Ndn,d}, x::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  # Recursive contractions to compute frame matrices and vectors from the right
  Fᴿ = Vector{Frame{T,Nup,Ndn,d,Matrix{T}}}(undef, d)
  Fᴿ[d] = IdFrame(Val(d), Val(Nup), Val(Ndn), d+1)
  for k=d-1:-1:1
    Fᴿ[k] = FramingStepLeft(H,x,Fᴿ[k+1])
  end

  return Fᴿ
end

function FramingStepRight(H::SparseHamiltonian{T,Nup,Ndn,d}, 
                          x::TTvector{T,Nup,Ndn,d},
                          Fᴸ::Frame{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
    k = site(Fᴸ)
    Xₖ = x.cores[k]
    return (adjoint(Xₖ) * Fᴸ) * H * Xₖ
end

function FramingStepLeft(H::SparseHamiltonian{T,Nup,Ndn,d}, 
                          x::TTvector{T,Nup,Ndn,d},
                          Fᴿ::Frame{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
    k = site(Fᴿ)-1
    Xₖ = x.cores[k]
  return H * Xₖ * (Fᴿ * adjoint(Xₖ))
end