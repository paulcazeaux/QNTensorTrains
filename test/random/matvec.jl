# randRound_H_MatVec
@test begin
  T = Float64
  d = 20
  N = 7
  tol = 1e-3
  ϵ = 1e-4

  t = zeros(d,d)
  for i=1:d, j=1:d
    if i!=j
      t[i,j] = 1/abs(i-j)
    end
  end
  v = zeros(d,d,d,d)
  for i=1:d-1,j=i+1:d,k=1:d-1,l=k+1:d
    v[i,j,k,l] = 2/abs((i+k)-(j+l))
  end

  # Build tensor with rank 5 but ϵ-perturbations of random rank 1 state
  r = [[(k∈(1,d+1) ? 1 : 5) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  x = approx_state(Val(d),Val(N),r,ϵ)

  # Explicit matvec
  H = SparseHamiltonian(t,v,Val(N),Val(d))
  yref = H*x

  # Compute approximate randomized sum with max ranks M + oversampling
  y = randRound_H_MatVec(H, x, 10, 5)
  isapprox( norm(y-yref, :LR), 0, atol = tol * norm(yref) )
end


# randRound_H_MatVec2
@test begin
  T = Float64
  d = 20
  N = 7
  tol = 1e-3
  ϵ = 1e-4

  t = zeros(d,d)
  for i=1:d, j=1:d
    if i!=j
      t[i,j] = 1/abs(i-j)
    end
  end
  v = zeros(d,d,d,d)
  for i=1:d-1,j=i+1:d,k=1:d-1,l=k+1:d
    v[i,j,k,l] = 2/abs((i+k)-(j+l))
  end

  # Build tensor with rank 5 but ϵ-perturbations of random rank 1 state
  r = [[(k∈(1,d+1) ? 1 : 5) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  x = approx_state(Val(d),Val(N),r,ϵ)

  # Explicit matvec
  H = SparseHamiltonian(t,v,Val(N),Val(d))
  yref = H*x
  # Compute approximate randomized sum with max ranks M + oversampling
  y = randRound_H_MatVec2(H, x, 10, 5)
  isapprox( norm(y-yref, :LR), 0, atol = tol * norm(yref) )
end
