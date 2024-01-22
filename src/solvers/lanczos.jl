using LinearAlgebra, TimerOutputs


function Lanczos( t::Matrix{T}, v::Array{T,4}, x0::TTvector{T,N,d};
                  tol::Float64 = 1e-4, maxIter::Int = 20, reduced::Bool=false) where {T<:Number,N,d}
  to = TimerOutput()
  reduced || (v = Hamiltonian.reduce(v))

@timeit to "Orthonormalization" begin
  q0, = leftOrthonormalize!!(deepcopy(x0))
end
@timeit to "RayleighQuotient" begin
  α = RayleighQuotient(q0, t, v; reduced=true)
end
  Q  = [q0]
  dv = [α ]
  ev = zeros(T,0)
  res = zeros(T,0)
  ranks = [maximum(maximum.(rank(q0)))]
  round!(q0, tol)
  rounded_ranks = [maximum(maximum.(rank(q0)))]

  # k=1
@timeit to "MatVec" begin
  q̃ = H_matvec(q0, t, v) - α*q0
  push!(ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "Orthonormalization" begin
  q, nrm = leftOrthonormalize!!(q̃)
  round!(q, tol)
  push!(rounded_ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "RayleighQuotient" begin
  α = RayleighQuotient(q, t, v; reduced=true)
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
    q̃ = H_matvec(Q[end], t, v) - α*Q[end] - β*Q[end-1]
    push!(ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "Orthonormalization" begin
    q, nrm = leftOrthonormalize!!(q̃)
    round!(q, tol)
    push!(rounded_ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "RayleighQuotient" begin
    α = RayleighQuotient(q, t, v; reduced=true)
end
    β = prod(nrm)

@timeit to "Eigenvalue estimation" begin
    F = eigen(SymTridiagonal(dv, ev))
    push!(λ, F.values[1])
    γ = F.vectors[:,1]
end

    @show α, β, β*abs(γ[end])
    if β*abs(γ[end]) < tol
      break
    elseif β*abs(γ[end]) > 2minimum(res)
      @warn "Terminating Lanczos iterations - inexact breakdown"
      break
    end
  end
  end

  display(to)
@timeit to "Assemble eigenvector" begin
  w, = leftOrthonormalize!!(roundSum(γ, Q, tol))
end
  return λ[end], w, λ, res, ranks, rounded_ranks
end

function randLanczos( t::Matrix{T}, v::Array{T,4}, x0::TTvector{T,N,d};
                  tol::Float64 = 1e-4, rmax=100, over = 5, maxIter::Int = 20, reduced::Bool=false) where {T<:Number,N,d}

  to = TimerOutput()
  reduced || (v = Hamiltonian.reduce(v))

@timeit to "Orthonormalization" begin
  q0, = leftOrthonormalize!!(deepcopy(x0))
end
@timeit to "RayleighQuotient" begin
  α = RayleighQuotient(q0, t, v; reduced=true)
end
  Q  = [q0]
  dv = [α ]
  ev = zeros(T,0)
  res = zeros(T,0)
  ranks = [maximum(maximum.(rank(q0)))]
  round!(q0, tol)
  rounded_ranks = [maximum(maximum.(rank(q0)))]

  # k=1
@timeit to "MatVec" begin
  q̃ = randRound_H_MatVecAdd([1,-α], [q0,q0], t, v, rmax, over)
  push!(ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "Orthonormalization" begin
  q, nrm = leftOrthonormalize!!(q̃)
  round!(q, tol, rmax=rmax)
  push!(rounded_ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "RayleighQuotient" begin
  α = RayleighQuotient(q, t, v; reduced=true)
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
    q̃ = randRound_H_MatVecAdd([1,-α,-β], [Q[end],Q[end],Q[end-1]], t, v, rmax, over)
    push!(ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "Orthonormalization" begin
    q, nrm = leftOrthonormalize!!(q̃)
    round!(q, tol, rmax=rmax)
    push!(rounded_ranks, maximum(maximum.(rank(q̃))))
end
@timeit to "RayleighQuotient" begin
    α = RayleighQuotient(q, t, v; reduced=true)
end
    β = prod(nrm)

@timeit to "Eigenvalue estimation" begin
    F = eigen(SymTridiagonal(dv, ev))
    push!(λ, F.values[1])
    γ = F.vectors[:,1]
end

    @show α, β, β*abs(γ[end])
    if β*abs(γ[end]) < tol
      break
    elseif β*abs(γ[end]) > 2minimum(res)
      @warn "Terminating Lanczos iterations - inexact breakdown"
      break
    end
  end
  end

  display(to)
@timeit to "Assemble eigenvector" begin
  w, = leftOrthonormalize!!(roundRandSum(γ, Q, rmax, over))
end
  return λ[end], w, λ, res, ranks, rounded_ranks
end