# randRound_H_MatVec
@test begin
  T = Float64
  d = 10
  N = 7
  Sz = 1//2
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
  x = approx_state(Val(d),Val(N),Val(Sz),5:5,ϵ)

  # Explicit matvec
  H = SparseHamiltonian(t,v,N,Sz,d)
  yref = H*x

  # Compute approximate randomized sum with max ranks M + oversampling
  y = randRound_H_MatVec(H, x, 10, 5)
  isapprox( norm(y-yref, :LR), 0, atol = tol * norm(yref) )
end


# randRound_H_MatVec2
@test begin
  T = Float64
  d = 10
  N = 8
  Sz = 0//2
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
  x = approx_state(Val(d),Val(N),Val(Sz),5:5,ϵ)

  # Explicit matvec
  H = SparseHamiltonian(t,v,N,Sz,d)

  yref = H*x
  # Compute approximate randomized sum with max ranks M + oversampling
  y = randRound_H_MatVec2(H, x, 10, 5)
  isapprox( norm(y-yref, :LR), 0, atol = tol * norm(yref) )
end
