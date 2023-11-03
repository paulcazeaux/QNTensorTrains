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

  ϵ = 1e-4
  ddx = 1/ϵ * (round!(x+ϵ*dir, rank(x)) - x)
  norm(TTvector(dx) - ddx)/norm(dir) < ϵ
end


# Tangent to tt_randn
@test begin
  T = Float64
  d = 10
  N = 6

  r = [ [ (k in (1,d+1) ? 1 : rand(0:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  x = tt_randn(Val(d),Val(N),r)
  r = [ [ (k in (1,d+1) ? 1 : rand(2:4)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  dir = tt_randn(Val(d),Val(N),r)
  dx = TTtangent(x, dir)

  ϵ = 1e-4
  ddx = 1/ϵ * (round!(x+ϵ*dir, rank(x)) - x)
  norm(TTvector(dx) - ddx)/norm(dir) < 10ϵ
end
