# Tangent to tt_state
@test begin
  T = Float64
  d = 8
  N = 6
  Sz = 2//2

  s1 = [2,4,6,8]
  s2 = [3,6]
  x = tt_state([k∈s1 for k=1:d], [k∈s2 for k=1:d])
  dir = randomized_state(d, N, Sz, 2:4)
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
  Sz = 0//2

  x = round!(randomized_state(d, N, Sz, 1:5))
  dir = round!(randomized_state(d, N, Sz, 2:4))
  x = x / norm(x)
  dir = dir / norm(dir)
  dx = TTtangent(x, dir)

  ϵ = 1e-6
  ddx1 = 1/ϵ  * (round!(x+ ϵ*dir, rank(x)) - x)
  ddx2 = 1/2ϵ * (round!(x+2ϵ*dir, rank(x)) - x)
  isapprox( norm(TTvector(dx) - ddx2, :LR)/norm(TTvector(dx) - ddx1, :LR), 2.0, atol=0.1)
end
