using LinearAlgebra, TimerOutputs


function Lanczos( H::SparseHamiltonian{T,N,d}, x0::TTvector{T,N,d};
                  tol::Float64 = 1e-4, maxIter::Int = 20) where {T<:Number,N,d}
  to = TimerOutput()
  
@timeit to "Orthonormalization" begin
  q0, = leftOrthonormalize!!(deepcopy(x0))
end
@timeit to "RayleighQuotient" begin
  α = RayleighQuotient(H, q0)
end
  Q  = [q0]
  dv = [α ]
  ev = zeros(T,0)
  res = zeros(T,0)
  ranks = [maximum(maximum.(rank(q0)))]
  round!(q0, tol)
  rounded_ranks = [maximum(maximum.(rank(q0)))]
  breakdown = false

  # k=1
@timeit to "MatVec" begin
  q̃ = H*q0 - α*q0
  push!(ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "Orthonormalization" begin
  q, nrm = leftOrthonormalize!!(q̃)
  round!(q, tol)
  push!(rounded_ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "RayleighQuotient" begin
  α = RayleighQuotient(H, q)
end
  β = prod(nrm)
@timeit to "Eigenvalue estimation" begin
  F = eigen(SymTridiagonal(dv, ev))
  λ = [F.values[1]]
  γ = F.vectors[:,1]
end

  if β*abs(γ[end]) > tol
  for k=2:maxIter-1
    push!(Q,  q)
    push!(dv, α)
    push!(ev, β)
    push!(res, β*abs(γ[end]))

@timeit to "MatVec" begin
    q̃ = H*Q[end] - α*Q[end] - β*Q[end-1]
    push!(ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "Orthonormalization" begin
    q, nrm = leftOrthonormalize!!(q̃)
    round!(q, tol)
    push!(rounded_ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "RayleighQuotient" begin
    α = RayleighQuotient(H, q)
end
    β = prod(nrm)

@timeit to "Eigenvalue estimation" begin
    F = eigen(SymTridiagonal(dv, ev))
    push!(λ, F.values[1])
    γ = F.vectors[:,1]
end

    # @show α, β, β*abs(γ[end])
    if β*abs(γ[end]) < tol
      break
    elseif β*abs(γ[end]) > 2minimum(res)
      breakdown = true
      @warn "Terminating Lanczos iterations - inexact breakdown"
      break
    end
  end
  end

@timeit to "Assemble eigenvector" begin
  w, = leftOrthonormalize!!(roundSum(γ, Q, tol))
end
  display(to)
  return λ[end], w, breakdown, λ, res, ranks, rounded_ranks
end

function randLanczos( H::SparseHamiltonian{T,N,d}, x0::TTvector{T,N,d};
                  tol::Float64 = 1e-4, rmax=100, over = 5, maxIter::Int = 20, reduced::Bool=false) where {T<:Number,N,d}
  to = TimerOutput()

@timeit to "Orthonormalization" begin
  q0, = leftOrthonormalize!!(deepcopy(x0))
end
@timeit to "RayleighQuotient" begin
  α = RayleighQuotient(H, q0)
end
  Q  = [q0]
  dv = [α ]
  ev = zeros(T,0)
  res = zeros(T,0)
  ranks = [maximum(maximum.(rank(q0)))]
  round!(q0, tol)
  rounded_ranks = [maximum(maximum.(rank(q0)))]
  breakdown = false

@timeit to "Randomized MatVec" begin
  q̃ = randRound_H_MatVecAdd([1,-α], H, [q0,q0], rmax, over)
  # q̃ = randRound_H_MatVecAdd2([1,-α], H, [q0,q0], rmax+over, to)
  push!(ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "Orthonormalization" begin
  q, nrm = leftOrthonormalize!!(q̃)
  round!(q, tol, rmax=rmax)
  push!(rounded_ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "RayleighQuotient" begin
  α = RayleighQuotient(H, q)
end
  β = prod(nrm)
@timeit to "Eigenvalue estimation" begin
  F = eigen(SymTridiagonal(dv, ev))
  λ = [F.values[1]]
  γ = F.vectors[:,1]
end

  if β*abs(γ[end]) > tol
  for k=2:maxIter-1
    push!(Q,  q)
    push!(dv, α)
    push!(ev, β)
    push!(res, β*abs(γ[end]))

@timeit to "Randomized MatVec" begin
    q̃ = randRound_H_MatVecAdd([1,-α,-β], H, [Q[end],Q[end],Q[end-1]], rmax, over)
    # q̃ = randRound_H_MatVecAdd2([1,-α,-β], H, [Q[end],Q[end],Q[end-1]], rmax+over, to)
    push!(ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "Orthonormalization" begin
    q, nrm = leftOrthonormalize!!(q̃)
    round!(q, tol, rmax=rmax)
    push!(rounded_ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "RayleighQuotient" begin
    α = RayleighQuotient(H, q)
end
    β = prod(nrm)

@timeit to "Eigenvalue estimation" begin
    F = eigen(SymTridiagonal(dv, ev))
    push!(λ, F.values[1])
    γ = F.vectors[:,1]
end

    # @show α, β, β*abs(γ[end])
    if β*abs(γ[end]) < tol
      break
    elseif β*abs(γ[end]) > 2minimum(res)
      @warn "Terminating Lanczos iterations - inexact breakdown"
      breakdown = true
      break
    end
  end
  end

@timeit to "Assemble eigenvector" begin
  w, = leftOrthonormalize!!(roundRandSum(γ, Q, rmax, over))
  # w, = leftOrthonormalize!!(roundRandSum2(γ, Q, rmax, over))
end
  display(to)
  return λ[end], w, breakdown, λ, res, ranks, rounded_ranks
end