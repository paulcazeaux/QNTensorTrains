# Tangent to tt_state
@test begin
  T = Float64
  d = 10
  N = 6

  s = [2,4,6,7,8,10]
  x = tt_state([k∈s for k=1:d])

  r = [ [ (k in (1,d+1) ? 1 : rand(2:4)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  dir = tt_randn(Val(d),Val(N),r)
  dx = TTtangent(x, dir)

  ϵ = 1e-6
  ddx = 1/ϵ  * (round!(x+ ϵ*dir, rank(x)) - x)
  isapprox( norm(TTvector(dx) - ddx, :LR), 0.0, atol=10ϵ)
end


# Tangent to tt_randn. Note this test will fail with some small probability
@test begin
  T = Float64
  d = 10
  N = 6

  r = [ [ (k in (1,d+1) ? 1 : rand(1:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  x = round!(tt_randn(Val(d),Val(N),r))
  r = [ [ (k in (1,d+1) ? 1 : rand(2:4)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  dir = round!(tt_randn(Val(d),Val(N),r))
  x = x / norm(x)
  dir = dir / norm(dir)
  dx = TTtangent(x, dir)

  ϵ = 1e-6
  ddx1 = 1/ϵ  * (round!(x+ ϵ*dir, rank(x)) - x)
  ddx2 = 1/2ϵ * (round!(x+2ϵ*dir, rank(x)) - x)
  isapprox( norm(TTvector(dx) - ddx2, :LR)/norm(TTvector(dx) - ddx1, :LR), 2.0, atol=0.1)
end
