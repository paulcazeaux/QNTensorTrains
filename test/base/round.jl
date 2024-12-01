# Left orthogonalize
@test begin
  T = Float64
  d = 10
  N = 3
  Sz = -1//2

  x = randomized_state(d,N,Sz,0:10)
  X = Array(x)
  QNTensorTrains.leftOrthogonalize!(x)
  isapprox( norm( X - Array(x) ), 0, atol = 1e-12 * norm(X) )
end

# Left orthogonalize - rank conserving
@test begin
  T = Float64
  d = 10
  N = 3
  Sz = 1//2

  x = randomized_state(d,N,Sz,0:10)
  r = rank(x)

  X = Array(x)
  QNTensorTrains.leftOrthogonalize!(x, keepRank=true)
  isapprox( norm( X - Array(x) ), 0.0,  atol=1e-12 * norm(X) ) && all(all(rank(x,k) .== r[k]) for k=1:d+1)
end

# Right orthogonalize
@test begin
  T = Float64
  d = 10
  N = 4
  Sz = 0//2

  x = randomized_state(d,N,Sz,0:10)
  X = Array(x)
  QNTensorTrains.rightOrthogonalize!(x)
  isapprox( norm( X - Array(x) ), 0, atol = 1e-12 * norm(X) )
end

# Right orthogonalize - rank conserving
@test begin
  T = Float64
  d = 10
  N = 4
  Sz = -2//2

  x = randomized_state(d,N,Sz,0:10)
  r = rank(x)
  X = Array(x)
  QNTensorTrains.rightOrthogonalize!(x, keepRank=true)
  isapprox( norm( X - Array(x) ), 0.0, atol = 1e-12 * norm(X) ) && all(all(rank(x,k) .== r[k]) for k=1:d+1)
end

# Round
@test begin
  tol = 1e-8
  T = Float64
  d = 10
  N = 7
  Sz = -1//2

  x = randomized_state(d,N,Sz,0:10)
  X = Array(x)
  y = round(x, tol)
  Y = Array(y)

  isapprox( norm( X-Y ), 0, atol = tol * norm(X) )
end


# Global round
@test begin
  tol = 1e-8
  T = Float64
  d = 10
  N = 6
  Sz = 0//2

  x = randomized_state(d,N,Sz,0:10)
  X = Array(x)
  y = round(x, tol; Global=true)
  Y = Array(y)

  isapprox( norm( X-Y ), 0, atol = tol * norm(X) )
end
