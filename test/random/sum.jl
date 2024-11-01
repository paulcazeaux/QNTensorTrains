# roundRandSum
@test begin
  T = Float64
  d = 25
  N = 7
  tol = 1e-3
  ϵ = 1e-4

  M = 10

  # Build summands with rank 5 but ϵ-perturbations of random rank 1 state
  r = [[(k∈(1,d+1) ? 1 : 5) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  x = [approx_state(Val(d),Val(N),r,ϵ) for _ in 1:M]
  α = [rand(T)               for _ in 1:M]

  # Explicit sum
  s = α[1]*x[1]
  for i=2:length(x)
    s += α[i]*x[i]
  end

  # Compute approximate randomized sum with max ranks M + oversampling
  y = roundRandSum(α, x, M, 10)

  isapprox( norm(y-s, :LR), 0, atol = tol * norm(s) )
end

# roundRandSum2
@test begin
  T = Float64
  d = 25
  N = 7
  tol = 1e-3
  ϵ = 1e-4

  M = 10

  # Build summands with rank 5 but ϵ-perturbations of random rank 1 state
  r = [[(k∈(1,d+1) ? 1 : 5) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  x = [approx_state(Val(d),Val(N),r,ϵ) for _ in 1:M]
  α = [rand(T)               for _ in 1:M]

  # Explicit sum
  s = α[1]*x[1]
  for i=2:length(x)
    s += α[i]*x[i]
  end

  # Compute approximate randomized sum with max ranks M + oversampling
  y = roundRandSum2(α, x, M, 10)

  isapprox( norm(y-s, :LR), 0, atol = tol * norm(s) )
end