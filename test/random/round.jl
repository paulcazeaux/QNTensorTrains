# Round
@test begin
  T = Float64
  d = 50
  N = 7
  tol = 1e-3

  x0 = round!(draw_state(Val(d),Val(N)) + draw_state(Val(d),Val(N)))

  r = [ [ (k==1 || k==d+1 ? 1 : rand(0:20)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  p = round(tt_randn(Val(d),Val(N),r,T))
  lmul!(tol/2 * norm(x0)/norm(p,:LR), p)

  x = round(x0 + p)

  target_r = [ [ (k==1 || k==d+1 ? 1 : 5) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  y = roundRandOrth(x, rank(x0), 5)
  y0 = round(y, rank(x0))

  isapprox( norm(y-x0, :LR), 0, atol = tol * norm(x0) ) && isapprox( norm(y0-x0, :LR), 0, atol = tol * norm(x0) )
end


# Round2
@test begin
  T = Float64
  d = 50
  N = 7
  tol = 1e-3

  x0 = round!(draw_state(Val(d),Val(N)) + draw_state(Val(d),Val(N)))

  r = [ [ (k==1 || k==d+1 ? 1 : rand(0:20)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  p = round(tt_randn(Val(d),Val(N),r,T))
  lmul!(tol/2 * norm(x0)/norm(p,:LR), p)

  x = round(x0 + p)

  y = roundRandOrth2(x, 2, 5)
  y0 = round(y, rank(x0))

  isapprox( norm(y-x0, :LR), 0, atol = tol * norm(x0) ) && isapprox( norm(y0-x0, :LR), 0, atol = tol * norm(x0) )
end