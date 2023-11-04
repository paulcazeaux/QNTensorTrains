
# Helper draw function
function draw!(ω)
  i = rand(ω)
  pop!(ω, i)
  return i
end

function approx_state(::Val{d},::Val{N},r,ϵ) where {N,d}
  ω = Set(1:d)
  s = ntuple(i->draw!(ω), Val(N))

  return round!(perturbation(tt_state(s, Val(d)),r,ϵ))
end

# roundRandSum
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
  yref = H_matvec(x,t,v)

  # Compute approximate randomized sum with max ranks M + oversampling
  y = randRound_H_MatVec(x, t, v, 6, 5)

  isapprox( norm(y-yref, :LR), 0, atol = tol * norm(yref) )
end
