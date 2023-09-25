
# Helper draw function
function draw!(ω)
  i = rand(ω)
  pop!(ω, i)
  return i
end

function draw_state(d,N)
  ω = Set(1:d)
  s = ((draw!(ω) for i=1:N)..., )
  return tt_state(s, d)
end

# Round
@test begin
  T = Float64
  d = 50
  N = 7
  tol = 1e-3

  x0 = round!(draw_state(d,N) + draw_state(d,N))

  r = [ [ (k==1 || k==d+1 ? 1 : rand(0:20)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  p = round(tt_randn(d,N,r,T))
  lmul!(tol/2 * norm(x0)/norm(p,:LR), p)

  x = round(x0 + p)

  target_r = [ [ (k==1 || k==d+1 ? 1 : 5) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  y = roundRandOrth(x, target_r)
  y0 = round(y, rank(x0))

  isapprox( norm(y-x0, :LR), 0, atol = tol * norm(x0) ) && isapprox( norm(y0-x0, :LR), 0, atol = tol * norm(x0) )
end
