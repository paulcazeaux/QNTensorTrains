using LinearAlgebra, TimerOutputs


const over = 5

function Lanczos( t::Matrix{T}, v::Array{T,4}, x0::TTvector{T,N,d};
                  tol::Float64 = 1e-4, rmax=100, maxIter::Int = 20, reduced::Bool=false) where {T<:Number,N,d}
  reduced || (v = Hamiltonian.reduce(v))

  q0, = leftOrthonormalize!!(deepcopy(x0))
  α = RayleighQuotient(q0, t, v; reduced=true)

  Q  = [q0]
  dv = [α ]
  ev = zeros(T,0)

  # k=1
  Aq = H_matvec(q0, t, v, reduced=true)
  q̃ = Aq - α*q0
  q, nrm = leftOrthonormalize!!(q̃)
  α = RayleighQuotient(q, t, v; reduced=true)
  β = prod(nrm)

  push!(Q,  q)
  push!(dv, α)
  push!(ev, β)

  F = eigen(SymTridiagonal(dv, ev))
  λ, γ = F.values[1], F.vectors[:,1]
  w = roundSum(γ, Q, ϵ=tol)
  r = norm(H_matvec(w, t, v) - λ*w, :RL)
    
  if r > tol 
  for k=2:maxIter-1
    Aq = H_matvec(Q[end], t, v)
    q̃ = Aq - α*Q[end] - β*Q[end-1]
    q, nrm = leftOrthonormalize!!(q̃)
    α = RayleighQuotient(q, t, v; reduced=true)
    β = prod(nrm)

    push!(Q,  q)
    push!(dv, α)
    push!(ev, β)

    F = eigen(SymTridiagonal(dv, ev))
    λ, γ = F.values[1], F.vectors[:,1]
    w = roundSum(γ, Q, ϵ=tol)
    r = norm(H_matvec(w, t, v) - λ*w, :RL)

    O = zeros(k+1,k+1)
    for i=1:k+1, j=1:k+1
      O[i,j] = dot(Q[i],Q[j],orthogonalize=true)
    end

    @show λ, r, norm(w)
    @show γ
    display(O)
    display(SymTridiagonal(dv, ev))

    # w2 = roundRandSum(γ, Q, rmax, over)
    # @show norm(w-w2, :LR)
    # if norm(w-w2, :LR) > .1
    #   @warn "rank deficient sketch"
    # end
    if r < tol 
      break
    end
  end
  end

  w, = leftOrthonormalize!!(w)
  @show λ, norm(w), norm(H_matvec(w, t, v) - λ*w, :RL)
  return λ, w
end

function randLanczos( t::Matrix{T}, v::Array{T,4}, x0::TTvector{T,N,d};
                  tol::Float64 = 1e-4, rmax=100, maxIter::Int = 20, reduced::Bool=false) where {T<:Number,N,d}

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

  # k=1
@timeit to "MatVec" begin
  q̃ = randRound_H_MatVecAdd([1,-α], [q0,q0], t, v, rmax, over)
end
@timeit to "Orthonormalization" begin
  q, nrm = leftOrthonormalize!!(q̃)
  round!(q, tol^2)
end
@timeit to "RayleighQuotient" begin
  α = RayleighQuotient(q, t, v; reduced=true)
end
  β = prod(nrm)
@timeit to "Eigenvalue estimation" begin
  F = eigen(SymTridiagonal(dv, ev))
  λ = [F.values[1]]
  γ = F.vectors[:,1]
  res = [β*abs(γ[end])]
end

  push!(Q,  q)
  push!(dv, α)
  push!(ev, β)

@show res[end]
  if res[end] > tol
  for k=2:maxIter-1

@timeit to "MatVec" begin
    q̃ = randRound_H_MatVecAdd([1,-α,-β], [Q[end],Q[end],Q[end-1]], t, v, rmax, over)
end
# @timeit to "Orthogonalize" begin
#     Aq = H_matvec(Q[end], t, v, reduced=true)
#     q̃ = roundRandSum([1,-α,-β], [Aq, Q[end], Q[end-1]], rmax, over)
# end
@timeit to "Orthonormalization" begin
    q, nrm = leftOrthonormalize!!(q̃)
    round!(q, tol^2, rmax=rmax)
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

    push!(Q,  q)
    push!(dv, α)
    push!(ev, β)
    push!(res, β*abs(γ[end]))

    @show res[end]
    if res[end] < tol
      break
    end
  end
  end

  display(to)
  F = eigen(SymTridiagonal(dv, ev))
  push!(λ, F.values[1])
  γ = F.vectors[:,1]
@timeit to "Assemble eigenvector" begin
  w, = leftOrthonormalize!!(roundRandSum(γ, Q, rmax, over))
end
  return λ[end], w, λ, res
end